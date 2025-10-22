#!/usr/bin/env python3

import os, json, math, argparse, random, itertools
from collections import defaultdict, OrderedDict
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
from contextlib import nullcontext
from fpdf import FPDF
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ------------------------------
# Utils
# ------------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# optional: lower precision for MPS
USE_FP16 = (DEVICE == "mps")
def amp_ctx_A():
    return torch.autocast(device_type="mps", dtype=torch.float16) if torch.backends.mps.is_available() else nullcontext()
def amp_ctx_B():
    return nullcontext()

RESULTS_DIR = "results"; os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------
# Models
# ------------------------------
def get_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        m = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    elif name == "small_cnn":
        class SmallCNN(nn.Module):
            def __init__(self, nc=3, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(nc, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(128, num_classes)
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = self.pool(x).view(x.size(0), -1)
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
    if dataset == "cifar10":
        tr = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        te = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        train = torchvision.datasets.CIFAR10("../data", True, download=True, transform=tr)
        test  = torchvision.datasets.CIFAR10("../data", False, download=True, transform=te)
        if quick:
            train = torch.utils.data.Subset(train, list(range(2000)))
            test  = torch.utils.data.Subset(test,  list(range(1000)))
    elif dataset == "mnist":
        tr = te = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train = torchvision.datasets.MNIST("../data", True, download=True, transform=tr)
        test  = torchvision.datasets.MNIST("../data", False, download=True, transform=te)
        if quick:
            train = torch.utils.data.Subset(train, list(range(2000)))
            test  = torch.utils.data.Subset(test,  list(range(1000)))
    else:
        raise ValueError("Unknown dataset")
    return DataLoader(train, bs, True, num_workers=4, pin_memory=True), DataLoader(test, bs, False, num_workers=4, pin_memory=True)

# ------------------------------
# Train / Eval
# ------------------------------
def train(model, train_loader, val_loader, epochs=10, lr=1e-3, ckpt=None):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = -1
    for ep in range(epochs):
        model.train()
        for xb,yb in tqdm(train_loader, desc=f"train ep{ep+1}/{epochs}", leave=False):
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb); loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = eval_acc(model, val_loader)
        if acc > best:
            best = acc
            if ckpt: torch.save(model.state_dict(), ckpt)
    return model

def eval_acc(model, loader):
    model.eval(); c=t=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            c += (pred==yb).sum().item(); t += yb.size(0)
    return c/max(1,t)

# ------------------------------
# Layer utilities
# ------------------------------
def collect_layers(model):
    layers=[]
    for n,m in model.named_modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            layers.append((n,m))
    return layers

# ------------------------------
# Core G-NDI & baselines
# ------------------------------
def compute_all_metrics(model, val_loader, layer_list, quick=False, use_mean_baseline=True, sample_cap=None):
    model.eval()
    layer_names=[n for n,_ in layer_list]
    layer_by={n:m for n,m in layer_list}

    activations={}
    def hook_fn(name):
        def h(m,i,o):
            activations[name]=o
            if o.requires_grad:
                try:o.retain_grad()
                except:pass
            return o
        return h
    handles=[m.register_forward_hook(hook_fn(n)) for n,m in layer_list]

    weight_l1={n: float(m.weight.detach().abs().sum().cpu().item()) if hasattr(m,'weight') and m.weight is not None else 0.0 for n,m in layer_list}
    pred_acc = {n:0.0 for n in layer_names}
    pred_mean_acc = {n:0.0 for n in layer_names}
    actual_acc = {n:0.0 for n in layer_names}
    actnorm_acc = {n:0.0 for n in layer_names}
    taylor_acc = {n:0.0 for n in layer_names}
    count=0

    # running means for mean-baseline
    mean_buf = {n: None for n in layer_names}
    count_mean = 0

    # build per-sample pairs for corr-vs-samples
    pair_store = {n: [] for n in layer_names}
    pair_store_mean = {n: [] for n in layer_names}

    # helper closures
    def forward_with_zero(name, x1):
        mod = layer_by[name]

        def _hook(m, inp, out):
            return torch.zeros_like(out)  # zeros created with matching dtype/device

        h = mod.register_forward_hook(_hook)
        try:
            y = model(x1)
        finally:
            h.remove()
        return y

    def g_from_activation(name, a_repl, x1):
        mod = layer_by[name]

        def _hook(m, inp, out):
            return a_repl.to(out.dtype)  # <- ensure dtype matches (fp16 under MPS autocast)

        h = mod.register_forward_hook(_hook)
        try:
            y = model(x1)
        finally:
            h.remove()
        return y

    # Pass A: collect act_norm and Taylor
    for xb,yb in tqdm(val_loader, desc="pass A (batch stats)", leave=False):
        xb,yb=xb.to(DEVICE), yb.to(DEVICE); B=xb.size(0)
        activations.clear()
        model.zero_grad(set_to_none=True)
        with amp_ctx_A():
            logits = model(xb)

        # act norm
        with torch.no_grad():
            for n in layer_names:
                a = activations.get(n,None)
                if a is not None:
                    actnorm_acc[n]+= a.detach().view(B,-1).abs().mean(1).sum().item()

        # taylor
        loss = F.cross_entropy(logits, yb); loss.backward()
        with torch.no_grad():
            for n in layer_names:
                a=activations.get(n,None); g=None if a is None else a.grad
                if (a is not None) and (g is not None):
                    taylor_acc[n]+= (a.detach().view(B,-1).abs()*g.detach().view(B,-1).abs()).sum(1).sum().item()

        # mean baseline buffer
        with torch.no_grad():
            for n in layer_names:
                a=activations.get(n,None)
                if a is not None:
                    cur_mean = a.mean(dim=0, keepdim=True, dtype=a.dtype).detach()
                    if mean_buf[n] is None: mean_buf[n]=cur_mean
                    else: mean_buf[n] = (mean_buf[n]*count_mean + cur_mean)/(count_mean+1)
        count_mean += 1
        if quick: break

    # Pass B: per-sample JvP and actual deltas
    jvp_supported = hasattr(torch.autograd.functional, "jvp")
    processed = 0
    for xb,yb in tqdm(val_loader, desc="pass B (per-sample)", leave=False):
        xb,yb=xb.to(DEVICE), yb.to(DEVICE); B=xb.size(0)
        for i in range(B):
            xi = xb[i:i+1].detach()
            activations.clear()
            with amp_ctx_B(), torch.no_grad():
                f_base = model(xi)
            for n in layer_names:
                a = activations.get(n,None)
                if a is None: continue
                a_i = a[0:1].detach().requires_grad_(True)
                v_i = a_i
                def g(a_repl): return g_from_activation(n, a_repl, xi)

                # JvP for zero baseline (v=a)
                if jvp_supported:
                    _, jv = torch.autograd.functional.jvp(g, (a_i,), (v_i,), create_graph=False, strict=True)
                else:
                    eps=1e-3
                    jv = (g(a_i + eps*v_i).detach() - g(a_i).detach())/eps
                # positive alignment (larger predicted -> larger damage)
                pred = float(jv.norm().item())
                pred_acc[n]+= pred

                # JvP for mean-baseline
                if mean_buf[n] is not None:
                    a_bar = mean_buf[n].to(a_i.dtype)
                    v_mean = (a_i - a_bar)
                    if jvp_supported:
                        _, jv_m = torch.autograd.functional.jvp(g, (a_i,), (v_mean,), create_graph=False, strict=True)
                    else:
                        eps=1e-3
                        jv_m = (g(a_i + eps*v_mean).detach() - g(a_i).detach())/eps
                    pred_mean_acc[n]+= float(jv_m.norm().item())

                with torch.no_grad():
                    f_zero = forward_with_zero(n, xi)
                    delta = (f_base - f_zero).norm().item()
                actual_acc[n]+= float(delta)

                pair_store[n].append((pred, delta))
                if mean_buf[n] is not None:
                    pair_store_mean[n].append((float(jv_m.norm().item()), delta))

            count+=1; processed+=1
            if sample_cap and processed>=sample_cap: break
        if sample_cap and processed>=sample_cap: break
        if quick and processed>=512: break

    # Averages
    results={}
    for n in layer_names:
        results[n]={
            "S_GNDI": pred_acc[n]/max(1,count),
            "S_GNDI_mean": pred_mean_acc[n]/max(1,count),
            "actual_delta": actual_acc[n]/max(1,count),
            "weight_l1": weight_l1[n],
            "act_norm": actnorm_acc[n]/max(1,count),
            "taylor_fo": taylor_acc[n]/max(1,count),
            "samples": count
        }

    # correlations (zero-baseline)
    preds = np.array([results[n]["S_GNDI"] for n in layer_names], float)
    acts  = np.array([results[n]["actual_delta"] for n in layer_names], float)
    pr = pearsonr(preds, acts) if len(layer_names)>=3 and not np.allclose(preds,preds[0]) and not np.allclose(acts,acts[0]) else (np.nan,np.nan)
    sr = spearmanr(preds, acts) if len(layer_names)>=3 else (np.nan,np.nan)

    # correlations (mean-baseline)
    mpreds = np.array([results[n]["S_GNDI_mean"] for n in layer_names], float)
    mpr = pearsonr(mpreds, acts) if len(layer_names)>=3 and not np.allclose(mpreds,mpreds[0]) and not np.allclose(acts,acts[0]) else (np.nan,np.nan)
    msr = spearmanr(mpreds, acts) if len(layer_names)>=3 else (np.nan,np.nan)

    # sample-size curve
    sizes=[64,128,256,512,1024,2048]
    corr_rows=[]
    for N in sizes:
        P=[]; A=[]
        for n in layer_names:
            arr=np.array(pair_store[n],float)
            if len(arr)==0: continue
            arr=arr[:min(N,len(arr))]
            P.append(arr[:,0].mean()); A.append(arr[:,1].mean())
        if len(P)>=3:
            p=pearsonr(P,A)[0]; s=spearmanr(P,A)[0]
        else:
            p=np.nan; s=np.nan
        corr_rows.append({"samples":N,"pearson_r":p,"spearman_rho":s,"variant":"zero"})

        # mean variant
        P=[]; A=[]
        for n in layer_names:
            arr=np.array(pair_store_mean[n],float) if len(pair_store_mean[n])>0 else None
            if arr is None: continue
            arr=arr[:min(N,len(arr))]
            P.append(arr[:,0].mean()); A.append(arr[:,1].mean())
        if len(P)>=3:
            p=pearsonr(P,A)[0]; s=spearmanr(P,A)[0]
        else:
            p=np.nan; s=np.nan
        corr_rows.append({"samples":N,"pearson_r":p,"spearman_rho":s,"variant":"mean"})

    # Cleanup hooks
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
# Ranking experiment (top-k curves)
# ------------------------------
def ranking_curve(model, val_loader, layer_list, results, metric_key):
    model.eval()
    base = eval_acc(model, val_loader)
    layers_sorted = sorted(layer_list, key=lambda x: results[x[0]][metric_key], reverse=True)
    fracs=[]; accs=[]
    for k in range(1, len(layers_sorted)+1):
        zero_set=set([n for n,_ in layers_sorted[:k]])
        handles=[]
        for n,m in layer_list:
            if n in zero_set:
                handles.append(m.register_forward_hook(lambda mod,i,o: torch.zeros_like(o)))
        acc = eval_acc(model, val_loader)
        for h in handles: h.remove()
        fracs.append(k/len(layers_sorted)); accs.append(acc)
    return fracs, accs, base

# ------------------------------
# Plots-only PDF
# ------------------------------
def plots_only_pdf(fig_paths, out_pdf):
    pdf=FPDF()
    for p in fig_paths:
        pdf.add_page()
        pdf.image(p, x=5, y=5, w=200)  # no titles/prose
    pdf.output(out_pdf)

# ------------------------------
# Main experiment (single seed)
# ------------------------------
def run_single(args, seed=0, tag="run"):
    set_seed(seed)
    train_loader, val_loader = get_loaders(args.dataset, args.batch, args.quick)
    nc = 3 if args.dataset=="cifar10" else 1
    num_classes = 10
    model = get_model(args.model or ("resnet18" if args.dataset=="cifar10" else "mnist_mlp"), num_classes)
    ckpt = args.checkpoint or os.path.join(RESULTS_DIR, f"{args.model}_{args.dataset}_seed{seed}.pt")

    if args.train or not os.path.exists(ckpt):
        train(model, train_loader, val_loader, epochs=args.epochs, ckpt=ckpt)
    else:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE); model.eval()

    base_acc = eval_acc(model, val_loader)

    layer_list = collect_layers(model)

    results, stats, corr_df = compute_all_metrics(
        model, val_loader, layer_list,
        quick=args.quick, use_mean_baseline=True,
        sample_cap=None
    )

    # Save tabular metrics
    metrics_path_json = os.path.join(RESULTS_DIR, f"metrics_{tag}_seed{seed}.json")
    with open(metrics_path_json,"w") as f: json.dump({"stats":stats,"results":results,"base_acc":base_acc}, f, indent=2)

    # CSVs
    rows=[]
    for n,_ in layer_list:
        r=results[n].copy(); r["layer"]=n
        rows.append(r)
    df = pd.DataFrame(rows).set_index("layer")
    df.to_csv(os.path.join(RESULTS_DIR, f"metrics_{tag}_seed{seed}.csv"))

    corr_df.to_csv(os.path.join(RESULTS_DIR, f"correlations_{tag}_seed{seed}.csv"), index=False)

    # Top-k curves for all metrics
    figs=[]
    metrics_for_rank=["S_GNDI","S_GNDI_mean","weight_l1","act_norm","taylor_fo"]
    topk_rows=[]
    plt.figure(figsize=(7,5))
    for mk in metrics_for_rank:
        fracs, accs, base = ranking_curve(model, val_loader, layer_list, results, mk)
        plt.plot(fracs, accs, label=mk)
        for f,a in zip(fracs, accs):
            topk_rows.append({"metric":mk, "frac":f, "acc":a, "seed":seed})
    plt.axhline(base, linestyle="--", linewidth=1)
    plt.legend()
    path_topk=os.path.join(RESULTS_DIR, f"topk_{tag}_seed{seed}.png"); plt.tight_layout(); plt.savefig(path_topk, dpi=150); plt.close()
    pd.DataFrame(topk_rows).to_csv(os.path.join(RESULTS_DIR, f"topk_curves_{tag}_seed{seed}.csv"), index=False)
    figs.append(path_topk)

    # Scatter plots (zero & mean variants) — no titles/prose, only axes
    # zero variant
    names=[n for n,_ in layer_list]
    x=np.array([results[n]["S_GNDI"] for n in names]); y=np.array([results[n]["actual_delta"] for n in names])
    plt.figure(figsize=(6,5)); plt.scatter(x,y,s=20)
    for i,n in enumerate(names): plt.text(x[i],y[i],n,fontsize=6)
    plt.xlabel("predicted (G-NDI, zero)"); plt.ylabel("actual delta")
    pth=os.path.join(RESULTS_DIR, f"scatter_zero_{tag}_seed{seed}.png"); plt.tight_layout(); plt.savefig(pth, dpi=150); plt.close(); figs.append(pth)
    # mean variant
    x=np.array([results[n]["S_GNDI_mean"] for n in names]); y=np.array([results[n]["actual_delta"] for n in names])
    plt.figure(figsize=(6,5)); plt.scatter(x,y,s=20)
    for i,n in enumerate(names): plt.text(x[i],y[i],n,fontsize=6)
    plt.xlabel("predicted (G-NDI, mean)"); plt.ylabel("actual delta")
    pth=os.path.join(RESULTS_DIR, f"scatter_mean_{tag}_seed{seed}.png"); plt.tight_layout(); plt.savefig(pth, dpi=150); plt.close(); figs.append(pth)

    # Sample-size correlation curves
    corr_plot = pd.read_csv(os.path.join(RESULTS_DIR, f"correlations_{tag}_seed{seed}.csv"))
    plt.figure(figsize=(7,5))
    for variant in sorted(corr_plot["variant"].unique()):
        sub = corr_plot[corr_plot["variant"]==variant]
        plt.plot(sub["samples"], sub["spearman_rho"], marker="o", label=f"{variant}")
    plt.xlabel("samples"); plt.ylabel("spearman rho")
    pth=os.path.join(RESULTS_DIR, f"corr_vs_samples_{tag}_seed{seed}.png"); plt.tight_layout(); plt.savefig(pth, dpi=150); plt.close(); figs.append(pth)

    # Bar: normalized metrics (data-only)
    df_norm = (df - df.min())/(df.max()-df.min()+1e-12)
    df_norm[["S_GNDI","S_GNDI_mean","actual_delta","weight_l1","act_norm","taylor_fo"]].plot.bar(figsize=(10,4))
    pth=os.path.join(RESULTS_DIR, f"metrics_bar_{tag}_seed{seed}.png"); plt.tight_layout(); plt.savefig(pth, dpi=150); plt.close(); figs.append(pth)

    # Build plots-only PDF
    out_pdf=os.path.join(RESULTS_DIR, f"report_data_only_{tag}_seed{seed}.pdf")
    plots_only_pdf(figs, out_pdf)

    return {
        "metrics_json": metrics_path_json,
        "metrics_csv": os.path.join(RESULTS_DIR, f"metrics_{tag}_seed{seed}.csv"),
        "correlations_csv": os.path.join(RESULTS_DIR, f"correlations_{tag}_seed{seed}.csv"),
        "topk_csv": os.path.join(RESULTS_DIR, f"topk_curves_{tag}_seed{seed}.csv"),
        "pdf": out_pdf,
        "base_acc": float(base)
    }

# ------------------------------
# Aggregate multiple seeds (optional)
# ------------------------------
def aggregate_seeds(artifacts):
    # artifacts: list of per-seed outputs
    rows=[]
    for art in artifacts:
        jd = json.load(open(art["metrics_json"]))
        s = jd["stats"]; base = jd.get("base_acc", np.nan)
        rows.append({"seed_json": art["metrics_json"],
                     "pearson_r": s.get("pearson_r", np.nan),
                     "spearman_rho": s.get("spearman_rho", np.nan),
                     "pearson_r_mean": s.get("pearson_r_mean", np.nan),
                     "spearman_rho_mean": s.get("spearman_rho_mean", np.nan),
                     "base_acc": base})
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
    ap.add_argument("--dataset", choices=["mnist","cifar10"], default="cifar10")
    ap.add_argument("--model", choices=["mnist_mlp","small_cnn","resnet18"], default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=1)
    args=ap.parse_args()

    if args.model is None:
        args.model = "resnet18" if args.dataset=="cifar10" else "mnist_mlp"

    artifacts=[]
    for s in range(args.seeds):
        tag=f"{args.dataset}_{args.model}"
        artifacts.append(run_single(args, seed=s, tag=tag))

    if args.seeds>1:
        summary_csv = aggregate_seeds(artifacts)
        print("seed summary:", summary_csv)

    print("Artifacts written under:", RESULTS_DIR)

if __name__=="__main__":
    main()
