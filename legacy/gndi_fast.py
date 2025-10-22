#!/usr/bin/env python3

import os, json, math, argparse, random
from contextlib import nullcontext
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from fpdf import FPDF

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ------------------------------
# Device & seeds
# ------------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()
RESULTS_DIR = "results"; os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------
# Models
# ------------------------------
def get_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    elif name == "small_cnn":
        class SmallCNN(nn.Module):
            def __init__(self, nc=3, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(nc, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2,2)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(128, num_classes)
            def forward(self, x):
                x = F.relu(self.conv1(x)); x = self.pool(F.relu(self.conv2(x)))
                x = F.relu(self.conv3(x))
                x = self.gap(x).view(x.size(0), -1)
                return self.fc(x)
        m = SmallCNN()
    elif name == "mnist_mlp":
        class MLP(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.fc1 = nn.Linear(28*28, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, num_classes)
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        m = MLP(num_classes=10)
    else:
        raise ValueError("Unknown model")
    return m

# ------------------------------
# Data
# ------------------------------
def get_loaders(dataset="cifar10", bs=128, quick=False):
    pin = False  # MPS ignores pinned memory; leave False universally
    if dataset == "cifar10":
        tr = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        te = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        train = torchvision.datasets.CIFAR10("../data", True, download=True, transform=tr)
        test  = torchvision.datasets.CIFAR10("../data", False, download=True, transform=te)
        if quick:
            train = torch.utils.data.Subset(train, list(range(2000)))
            test  = torch.utils.data.Subset(test,  list(range(1000)))
        return DataLoader(train, bs, True,  num_workers=4, pin_memory=pin), \
               DataLoader(test,  bs, False, num_workers=4, pin_memory=pin)
    elif dataset == "mnist":
        tr = te = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train = torchvision.datasets.MNIST("../data", True, download=True, transform=tr)
        test  = torchvision.datasets.MNIST("../data", False, download=True, transform=te)
        if quick:
            train = torch.utils.data.Subset(train, list(range(2000)))
            test  = torch.utils.data.Subset(test,  list(range(1000)))
        return DataLoader(train, bs, True,  num_workers=4, pin_memory=pin), \
               DataLoader(test,  bs, False, num_workers=4, pin_memory=pin)
    else:
        raise ValueError("Unknown dataset")

# ------------------------------
# Train / Eval
# ------------------------------
def train(model, train_loader, val_loader, epochs=10, lr=1e-3, ckpt=None):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = -1.0
    # Use fp16 autocast on MPS for speed in Pass-A style loops
    use_amp = (DEVICE == "mps")
    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train ep{ep+1}/{epochs}", leave=False)
        for xb,yb in pbar:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
            else:
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        acc = eval_acc(model, val_loader)
        if acc > best:
            best = acc
            if ckpt: torch.save(model.state_dict(), ckpt)
    return model

def eval_acc(model, loader):
    model.eval(); c=t=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            pred = out.argmax(1)
            c += (pred==yb).sum().item()
            t += yb.size(0)
    return c/max(1,t)

# ------------------------------
# Layers
# ------------------------------
def collect_layers(model, take_every=1, only_linear=False):
    layers=[]; k=0
    for n,m in model.named_modules():
        if only_linear and isinstance(m, nn.Linear):
            layers.append((n,m)); continue
        if (not only_linear) and isinstance(m,(nn.Conv2d, nn.Linear)):
            if k % max(1,take_every) == 0:
                layers.append((n,m))
            k += 1
    return layers

# ------------------------------
# Safe JvP with fallback
# ------------------------------
def safe_jvp_with_fallback(g_func, a_point, v_dir, use_jvp=True, fd_eps=1e-3):
    """
    Try forward-mode JvP; on failure (e.g., MaxPool non-smooth), fall back to finite difference along v:
       Jv ≈ (g(a+eps v) - g(a)) / eps
    Scale eps by ||v|| to keep step size stable.
    """
    if use_jvp and hasattr(torch.autograd.functional, "jvp"):
        try:
            with nullcontext():  # fp32 in Pass-B
                _, jv = torch.autograd.functional.jvp(
                    func=g_func,
                    inputs=(a_point,),
                    v=(v_dir,),
                    create_graph=False,
                    strict=True
                )
            return jv
        except Exception:
            pass
    denom = (v_dir.norm().detach().item() + 1e-12)
    eps = fd_eps / denom
    y0 = g_func(a_point).detach()
    y1 = g_func(a_point + eps * v_dir).detach()
    return (y1 - y0) / eps

# ------------------------------
# Core metrics
# ------------------------------
def compute_all_metrics(model, val_loader, layer_list, quick=False, sample_cap=256, do_mean_baseline=True):
    """
    Returns:
      - results: dict[layer] -> metrics
      - stats: correlations
      - corr_df: correlation vs sample-size (zero & mean variants)
    """
    model.eval()
    layer_names=[n for n,_ in layer_list]
    layer_by={n:m for n,m in layer_list}

    # forward hooks capture activations (retain grad for Taylor)
    activations={}
    def hook_fn(name):
        def h(m,i,o):
            activations[name]=o
            if o.requires_grad:
                try: o.retain_grad()
                except: pass
            return o
        return h
    handles=[m.register_forward_hook(hook_fn(n)) for n,m in layer_list]

    # Baseline: weight L1
    weight_l1={n: float(m.weight.detach().abs().sum().cpu().item()) if hasattr(m,'weight') and m.weight is not None else 0.0
               for n,m in layer_list}

    pred_acc   = {n:0.0 for n in layer_names}
    pred_mean  = {n:0.0 for n in layer_names}
    actual_acc = {n:0.0 for n in layer_names}
    actnorm    = {n:0.0 for n in layer_names}
    taylorfo   = {n:0.0 for n in layer_names}
    count=0

    # running mean baseline buffers (dtype-safe)
    mean_buf = {n: None for n in layer_names}
    mean_cnt = 0

    # per-sample pairs for corr-vs-samples
    pair_zero = {n: [] for n in layer_names}
    pair_mean = {n: [] for n in layer_names}

    # helpers
    def forward_with_zero(name, x1):
        mod = layer_by[name]
        def _h(m, inp, out): return torch.zeros_like(out)
        h = mod.register_forward_hook(_h)
        try:
            with nullcontext():
                y = model(x1)
        finally:
            h.remove()
        return y

    def g_from_activation(name, a_repl, x1):
        mod = layer_by[name]
        def _h(m, inp, out): return a_repl.to(out.dtype)
        h = mod.register_forward_hook(_h)
        try:
            with nullcontext():
                y = model(x1)
        finally:
            h.remove()
        return y

    # --------- Pass A: batch stats (act_norm, Taylor-FO, mean baseline) ---------
    use_ampA = (DEVICE == "mps")
    for xb,yb in tqdm(val_loader, desc="pass A (batch stats)", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        activations.clear()
        model.zero_grad(set_to_none=True)
        if use_ampA:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

        B = xb.size(0)
        with torch.no_grad():
            for n in layer_names:
                a = activations.get(n, None)
                if a is not None:
                    actnorm[n] += a.detach().view(B,-1).abs().mean(1).sum().item()
                    # mean baseline buf; keep dtype of a to avoid mps broadcast issues
                    curm = a.mean(dim=0, keepdim=True, dtype=a.dtype).detach()
                    if mean_buf[n] is None: mean_buf[n] = curm
                    else: mean_buf[n] = (mean_buf[n]*mean_cnt + curm)/(mean_cnt+1)
        # Taylor-FO
        loss.backward()
        with torch.no_grad():
            for n in layer_names:
                a = activations.get(n, None)
                g = None if a is None else a.grad
                if (a is not None) and (g is not None):
                    taylorfo[n] += (a.detach().view(B,-1).abs() * g.detach().view(B,-1).abs()).sum(1).sum().item()

        mean_cnt += 1
        if quick: break

    # --------- Pass B: per-sample JvP/FD + actual ---------
    processed=0
    for xb,yb in tqdm(val_loader, desc="pass B (per-sample)", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        B = xb.size(0)
        for i in range(B):
            if processed >= sample_cap: break
            xi = xb[i:i+1].detach()
            # baseline logits for delta
            activations.clear()
            with nullcontext():
                f_base = model(xi)

            for n in layer_names:
                a = activations.get(n, None)
                if a is None: continue
                a_i = a[0:1].detach().requires_grad_(True)
                v_i = a_i

                # Closure g for this layer
                def g(a_rep): return g_from_activation(n, a_rep, xi)

                # JvP or FD along v=a (zero-baseline)
                jv = safe_jvp_with_fallback(g, a_point=a_i, v_dir=v_i, use_jvp=True, fd_eps=1e-3)
                pred = float(jv.norm().item())
                pred_acc[n] += pred

                # Mean-baseline variant
                if do_mean_baseline and (mean_buf[n] is not None):
                    a_bar = mean_buf[n].to(a_i.dtype)
                    v_m = (a_i - a_bar)
                    jv_m = safe_jvp_with_fallback(g, a_point=a_i, v_dir=v_m, use_jvp=True, fd_eps=1e-3)
                    pred_mean[n] += float(jv_m.norm().item())

                # Actual effect
                with nullcontext():
                    f_zero = forward_with_zero(n, xi)
                    delta = (f_base - f_zero).norm().item()
                actual_acc[n] += float(delta)

                pair_zero[n].append((pred, delta))
                if do_mean_baseline and (mean_buf[n] is not None):
                    pair_mean[n].append((float(jv_m.norm().item()), delta))

            processed += 1
        if processed >= sample_cap: break
        if quick and processed >= 256: break

    # aggregate
    N = max(1, processed)
    results={}
    for n in layer_names:
        results[n] = {
            "S_GNDI": pred_acc[n]/N,
            "S_GNDI_mean": pred_mean[n]/N if do_mean_baseline else float("nan"),
            "actual_delta": actual_acc[n]/N,
            "weight_l1": weight_l1[n],
            "act_norm": actnorm[n]/max(1, mean_cnt),
            "taylor_fo": taylorfo[n]/max(1, mean_cnt),
            "samples": N
        }

    # correlations (zero-baseline)
    preds = np.array([results[n]["S_GNDI"] for n in layer_names], float)
    acts  = np.array([results[n]["actual_delta"] for n in layer_names], float)
    if len(layer_names)>=3 and not np.allclose(preds,preds[0]) and not np.allclose(acts,acts[0]):
        pr = pearsonr(preds, acts); sr = spearmanr(preds, acts)
    else:
        pr = (np.nan, np.nan); sr = (np.nan, np.nan)

    # correlations (mean-baseline)
    mpreds = np.array([results[n]["S_GNDI_mean"] for n in layer_names], float) if do_mean_baseline else None
    if do_mean_baseline and len(layer_names)>=3 and not np.allclose(mpreds, mpreds[0]) and not np.allclose(acts, acts[0]):
        mpr = pearsonr(mpreds, acts); msr = spearmanr(mpreds, acts)
    else:
        mpr = (np.nan, np.nan); msr = (np.nan, np.nan)

    # sample-size correlation curves
    sizes=[32,64,128,256,512,1024]
    corr_rows=[]
    for Nsub in sizes:
        P,A = [],[]
        for n in layer_names:
            arr = np.array(pair_zero[n], float)
            if arr.shape[0]==0: continue
            arr = arr[:min(Nsub, arr.shape[0])]
            P.append(arr[:,0].mean()); A.append(arr[:,1].mean())
        if len(P)>=3:
            p = pearsonr(P,A)[0]; s = spearmanr(P,A)[0]
        else:
            p = np.nan; s = np.nan
        corr_rows.append({"samples":Nsub, "pearson_r":p, "spearman_rho":s, "variant":"zero"})
        if do_mean_baseline:
            P,A = [],[]
            for n in layer_names:
                arr = np.array(pair_mean[n], float)
                if arr.shape[0]==0: continue
                arr = arr[:min(Nsub, arr.shape[0])]
                P.append(arr[:,0].mean()); A.append(arr[:,1].mean())
            if len(P)>=3:
                p = pearsonr(P,A)[0]; s = spearmanr(P,A)[0]
            else:
                p = np.nan; s = np.nan
            corr_rows.append({"samples":Nsub, "pearson_r":p, "spearman_rho":s, "variant":"mean"})

    # cleanup
    for h in handles: h.remove()

    stats = {
        "pearson_r": float(pr[0]) if pr[0]==pr[0] else float("nan"),
        "pearson_p": float(pr[1]) if pr[1]==pr[1] else float("nan"),
        "spearman_rho": float(sr[0]) if sr[0]==sr[0] else float("nan"),
        "spearman_p": float(sr[1]) if sr[1]==sr[1] else float("nan"),
        "pearson_r_mean": float(mpr[0]) if mpr[0]==mpr[0] else float("nan"),
        "spearman_rho_mean": float(msr[0]) if msr[0]==msr[0] else float("nan"),
    }
    return results, stats, pd.DataFrame(corr_rows)

# ------------------------------
# Ranking (top-k curves)
# ------------------------------
def ranking_curve(model, val_loader, layer_list, results, metric_key):
    model.eval()
    base = eval_acc(model, val_loader)
    ranked = sorted(layer_list, key=lambda x: results[x[0]][metric_key], reverse=True)
    fracs, accs = [], []
    total = len(ranked)
    for k in range(1, total+1):
        zero_set = set([n for n,_ in ranked[:k]])
        handles=[]
        for n,m in layer_list:
            if n in zero_set:
                handles.append(m.register_forward_hook(lambda module, i, o: torch.zeros_like(o)))
        acc = eval_acc(model, val_loader)
        for h in handles: h.remove()
        fracs.append(k/total); accs.append(acc)
    return fracs, accs, base

# ------------------------------
# Plots-only PDF (no prose)
# ------------------------------
def plots_only_pdf(fig_paths, out_pdf):
    pdf = FPDF()
    for p in fig_paths:
        pdf.add_page()
        pdf.image(p, x=5, y=5, w=200)
    pdf.output(out_pdf)

# ------------------------------
# Single run
# ------------------------------
def run_single(args, seed=0, tag="run"):
    set_seed(seed)
    train_loader, val_loader = get_loaders(args.dataset, args.batch, args.quick)
    model_name = args.model or ("resnet18" if args.dataset=="cifar10" else "mnist_mlp")
    model = get_model(model_name, num_classes=10)
    ckpt = args.checkpoint or os.path.join(RESULTS_DIR, f"{model_name}_{args.dataset}_seed{seed}.pt")

    if args.train or not os.path.exists(ckpt):
        train(model, train_loader, val_loader, epochs=args.epochs, ckpt=ckpt)
    else:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    model.to(DEVICE); model.eval()
    base_acc = eval_acc(model, val_loader)

    only_linear = (model_name == "mnist_mlp")
    layer_list = collect_layers(model, take_every=args.take_every, only_linear=only_linear)

    results, stats, corr_df = compute_all_metrics(
        model, val_loader, layer_list,
        quick=args.quick, sample_cap=args.sample_cap, do_mean_baseline=args.mean_baseline
    )

    # Save metrics
    metrics_json = os.path.join(RESULTS_DIR, f"metrics_{tag}_seed{seed}.json")
    with open(metrics_json, "w") as f:
        json.dump({"stats":stats, "results":results, "base_acc":base_acc}, f, indent=2)

    rows=[]
    for n,_ in layer_list:
        r = results[n].copy(); r["layer"]=n
        rows.append(r)
    df = pd.DataFrame(rows).set_index("layer")
    metrics_csv = os.path.join(RESULTS_DIR, f"metrics_{tag}_seed{seed}.csv")
    df.to_csv(metrics_csv)

    corr_csv = os.path.join(RESULTS_DIR, f"correlations_{tag}_seed{seed}.csv")
    corr_df.to_csv(corr_csv, index=False)

    # Top-k curves plot
    figs=[]
    plt.figure(figsize=(7,5))
    rank_metrics = ["S_GNDI","S_GNDI_mean"] if args.mean_baseline else ["S_GNDI"]
    rank_metrics += ["weight_l1","act_norm","taylor_fo"]
    topk_rows=[]
    for mk in rank_metrics:
        fr, ac, base = ranking_curve(model, val_loader, layer_list, results, mk)
        plt.plot(fr, ac, label=mk)
        for f,a in zip(fr,ac):
            topk_rows.append({"metric":mk,"frac":f,"acc":a,"seed":seed})
    plt.axhline(base, linestyle="--", linewidth=1)
    plt.legend()
    p_topk = os.path.join(RESULTS_DIR, f"topk_{tag}_seed{seed}.png")
    plt.tight_layout(); plt.savefig(p_topk, dpi=150); plt.close(); figs.append(p_topk)
    pd.DataFrame(topk_rows).to_csv(os.path.join(RESULTS_DIR, f"topk_curves_{tag}_seed{seed}.csv"), index=False)

    # Scatter (zero & mean)
    names=[n for n,_ in layer_list]
    x = np.array([results[n]["S_GNDI"] for n in names]); y = np.array([results[n]["actual_delta"] for n in names])
    plt.figure(figsize=(6,5)); plt.scatter(x,y,s=22)
    for i,n in enumerate(names): plt.text(x[i], y[i], n, fontsize=6)
    plt.xlabel("predicted (G-NDI, zero)"); plt.ylabel("actual delta")
    p_sc0 = os.path.join(RESULTS_DIR, f"scatter_zero_{tag}_seed{seed}.png")
    plt.tight_layout(); plt.savefig(p_sc0, dpi=150); plt.close(); figs.append(p_sc0)

    if args.mean_baseline:
        x = np.array([results[n]["S_GNDI_mean"] for n in names]); y = np.array([results[n]["actual_delta"] for n in names])
        plt.figure(figsize=(6,5)); plt.scatter(x,y,s=22)
        for i,n in enumerate(names): plt.text(x[i], y[i], n, fontsize=6)
        plt.xlabel("predicted (G-NDI, mean)"); plt.ylabel("actual delta")
        p_sc1 = os.path.join(RESULTS_DIR, f"scatter_mean_{tag}_seed{seed}.png")
        plt.tight_layout(); plt.savefig(p_sc1, dpi=150); plt.close(); figs.append(p_sc1)

    # Correlation vs samples
    cdf = pd.read_csv(corr_csv)
    plt.figure(figsize=(7,5))
    for variant in sorted(cdf["variant"].unique()):
        sub = cdf[cdf["variant"]==variant]
        plt.plot(sub["samples"], sub["spearman_rho"], marker="o", label=variant)
    plt.xlabel("samples"); plt.ylabel("spearman rho")
    p_cvs = os.path.join(RESULTS_DIR, f"corr_vs_samples_{tag}_seed{seed}.png")
    plt.tight_layout(); plt.savefig(p_cvs, dpi=150); plt.close(); figs.append(p_cvs)

    # Normalized bar
    df_norm = (df - df.min())/(df.max()-df.min()+1e-12)
    sel_cols = ["S_GNDI","actual_delta","weight_l1","act_norm","taylor_fo"]
    if args.mean_baseline: sel_cols.insert(1,"S_GNDI_mean")
    df_norm[sel_cols].plot.bar(figsize=(10,4))
    p_bar = os.path.join(RESULTS_DIR, f"metrics_bar_{tag}_seed{seed}.png")
    plt.tight_layout(); plt.savefig(p_bar, dpi=150); plt.close(); figs.append(p_bar)

    # Plots-only PDF
    out_pdf = os.path.join(RESULTS_DIR, f"report_data_only_{tag}_seed{seed}.pdf")
    plots_only_pdf(figs, out_pdf)

    return {"pdf": out_pdf, "metrics_json": metrics_json, "metrics_csv": metrics_csv,
            "corr_csv": corr_csv, "topk_csv": os.path.join(RESULTS_DIR, f"topk_curves_{tag}_seed{seed}.csv"),
            "base_acc": float(base)}

# ------------------------------
# Aggregate seeds
# ------------------------------
def aggregate_seeds(arts):
    rows=[]
    for a in arts:
        jd = json.load(open(a["metrics_json"]))
        s = jd["stats"]; base = jd.get("base_acc", float("nan"))
        rows.append({"pearson_r":s.get("pearson_r",np.nan),
                     "spearman_rho":s.get("spearman_rho",np.nan),
                     "pearson_r_mean":s.get("pearson_r_mean",np.nan),
                     "spearman_rho_mean":s.get("spearman_rho_mean",np.nan),
                     "base_acc":base})
    df=pd.DataFrame(rows)
    agg = df.agg(["mean","std"])
    out=os.path.join(RESULTS_DIR, "seed_summary.csv")
    agg.to_csv(out)
    return out

# ------------------------------
# CLI
# ------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist","cifar10"], default="mnist")
    ap.add_argument("--model", choices=["mnist_mlp","small_cnn","resnet18"], default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--sample-cap", type=int, default=256, help="per-sample computations in Pass B")
    ap.add_argument("--take-every", type=int, default=2, help="take every Nth eligible layer")
    ap.add_argument("--mean-baseline", action="store_true", help="also compute mean-baseline variant")
    ap.add_argument("--seeds", type=int, default=1)
    args=ap.parse_args()

    if args.model is None:
        args.model = "resnet18" if args.dataset=="cifar10" else "mnist_mlp"

    arts=[]
    tag=f"{args.dataset}_{args.model}"
    for s in range(args.seeds):
        arts.append(run_single(args, seed=s, tag=tag))

    if args.seeds > 1:
        summary = aggregate_seeds(arts)
        print("seed summary:", summary)
    print("Artifacts written to:", RESULTS_DIR)
    print("Device:", DEVICE)

if __name__=="__main__":
    main()
