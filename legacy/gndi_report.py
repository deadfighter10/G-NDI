#!/usr/bin/env python3

import os, json, argparse, random, textwrap, platform, time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from fpdf import FPDF

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------- Device, seeding, paths ----------------------

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()
RESULTS_DIR = "results"; os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

# ---------------------- Models ----------------------

def get_model(name: str="resnet18", num_classes: int=10) -> nn.Module:
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    elif name == "small_cnn":
        class SmallCNN(nn.Module):
            def __init__(self, nc=3, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(nc, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2,2)
                self.block = nn.Sequential(
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                )
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(128, num_classes)
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool(F.relu(self.conv2(x)))
                x = F.relu(self.conv3(x))
                r = F.relu(self.block(x))
                x = x + r
                x = self.gap(x).view(x.size(0), -1)
                return self.fc(x)
        m = SmallCNN()
    else:
        raise ValueError(f"Unknown model '{name}'")
    return m

# ---------------------- Data ----------------------

def get_loaders(dataset: str="cifar10", bs: int=128, quick: bool=False) -> Tuple[DataLoader, DataLoader]:
    pin = False
    if dataset == "cifar10":
        tr = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                        T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        te = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        train = torchvision.datasets.CIFAR10("../data", True, download=True, transform=tr)
        test  = torchvision.datasets.CIFAR10("../data", False, download=True, transform=te)
        if quick:
            train = torch.utils.data.Subset(train, list(range(5000)))
            test  = torch.utils.data.Subset(test,  list(range(2000)))
        return DataLoader(train, bs, True,  num_workers=4, pin_memory=pin), \
               DataLoader(test,  bs, False, num_workers=4, pin_memory=pin)
    else:
        raise ValueError("Use cifar10 for deeper tests")

# ---------------------- Train / Eval ----------------------

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          epochs: int=10, lr: float=1e-3, ckpt: str=None) -> nn.Module:
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = -1.0
    use_amp = (DEVICE == "mps")
    for ep in range(epochs):
        model.train()
        for xb,yb in tqdm(train_loader, desc=f"train ep{ep+1}/{epochs}", leave=False):
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    out = model(xb); loss = F.cross_entropy(out, yb)
            else:
                out = model(xb); loss = F.cross_entropy(out, yb)
            loss.backward(); opt.step()
        acc = eval_acc(model, val_loader)
        best = max(best, acc)
        if ckpt and acc == best:
            torch.save(model.state_dict(), ckpt)
    return model

@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader) -> float:
    model.eval(); c=t=0
    for xb,yb in loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb).argmax(1)
        c += (pred==yb).sum().item(); t += yb.size(0)
    return c/max(1,t)

# ---------------------- Layers ----------------------

def collect_layers(model: nn.Module, take_every: int=1) -> List[Tuple[str, nn.Module]]:
    layers=[]; k=0
    for n,m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if k % max(1, take_every) == 0:
                layers.append((n,m))
            k += 1
    return layers

def param_count(module: nn.Module) -> int:
    total=0
    for p in module.parameters(recurse=False):
        total += p.numel()
    return total

# ---------------------- Safe JvP ----------------------

def safe_jvp_with_fallback(g_func, a_point, v_dir, use_jvp=True, fd_eps=1e-3):
    if use_jvp and hasattr(torch.autograd.functional, "jvp"):
        try:
            with nullcontext():
                _, jv = torch.autograd.functional.jvp(
                    func=g_func, inputs=(a_point,), v=(v_dir,),
                    create_graph=False, strict=True
                )
            return jv
        except Exception:
            pass
    denom = (v_dir.norm().detach().item() + 1e-12)
    eps = fd_eps / denom
    y0 = g_func(a_point).detach()
    y1 = g_func(a_point + eps * v_dir).detach()
    return (y1 - y0) / eps

# ---------------------- Baselines: SNIP, SynFlow ----------------------

def snip_score(model: nn.Module, data_loader: DataLoader, layer_list: List[Tuple[str, nn.Module]], max_batches: int=1) -> Dict[str, float]:
    model.zero_grad(set_to_none=True); model.eval()
    it=0
    for xb,yb in data_loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb); loss = F.cross_entropy(out, yb)
        loss.backward()
        it+=1
        if it>=max_batches: break
    scores={}
    for name,mod in layer_list:
        if hasattr(mod,"weight") and mod.weight is not None and (mod.weight.grad is not None):
            scores[name] = float((mod.weight.grad * mod.weight).abs().sum().detach().cpu().item())
        else:
            scores[name] = 0.0
    model.zero_grad(set_to_none=True)
    return scores

def synflow_score(model: nn.Module, layer_list: List[Tuple[str, nn.Module]], input_shape: Tuple[int,...]) -> Dict[str, float]:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
        if p.grad is not None: p.grad = None
    model.zero_grad(set_to_none=True)

    x = torch.ones(*input_shape, device=DEVICE, dtype=torch.float32)
    y = model(x); s = y.sum()

    params = [m.weight for _, m in layer_list if hasattr(m,"weight") and (m.weight is not None)]
    grads = torch.autograd.grad(outputs=s, inputs=params, retain_graph=False, create_graph=False, allow_unused=True)

    scores={}; j=0
    for name,mod in layer_list:
        if hasattr(mod,"weight") and mod.weight is not None:
            g = grads[j]; j += 1
            scores[name] = float((g * mod.weight).abs().sum().detach().cpu().item()) if g is not None else 0.0
        else:
            scores[name] = 0.0
    model.zero_grad(set_to_none=True)
    return scores

# ---------------------- Core metrics ----------------------

@dataclass
class MetricOptions:
    quick: bool=False
    sample_cap: int=256
    do_mean_baseline: bool=True
    compute_delta_acc: bool=True

def softmax_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p||q) with p=softmax(p_logits), q=softmax(q_logits)."""
    p = F.softmax(p_logits, dim=-1)
    logp = F.log_softmax(p_logits, dim=-1)
    logq = F.log_softmax(q_logits, dim=-1)
    kl = (p * (logp - logq)).sum(dim=-1)
    return kl

def compute_all_metrics(model: nn.Module, val_loader: DataLoader, layer_list: List[Tuple[str, nn.Module]],
                        options: MetricOptions):
    model.eval()
    layer_names=[n for n,_ in layer_list]
    layer_by={n:m for n,m in layer_list}

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

    weight_l1={n: float(m.weight.detach().abs().sum().cpu().item()) if hasattr(m,'weight') and m.weight is not None else 0.0
               for n,m in layer_list}
    params_count={n: param_count(m) for n,m in layer_list}

    pred_zero   = {n:0.0 for n in layer_names}
    pred_mean   = {n:0.0 for n in layer_names}
    act_logitsL2= {n:0.0 for n in layer_names}
    act_softKL  = {n:0.0 for n in layer_names}
    actnorm     = {n:0.0 for n in layer_names}
    taylorfo    = {n:0.0 for n in layer_names}
    mean_buf    = {n: None for n in layer_names}; mean_cnt=0

    pair_zero   = {n: [] for n in layer_names}  # (pred, actual KL)
    pair_mean   = {n: [] for n in layer_names}

    # helpers
    def forward_with_replacement(name, x1, replace_fn):
        mod = layer_by[name]
        h = mod.register_forward_hook(lambda m,i,o: replace_fn(o))
        try:
            y = model(x1)
        finally:
            h.remove()
        return y

    # Pass A: gather means, Taylor-FO, act-norm
    use_ampA = (DEVICE == "mps")
    for xb,yb in tqdm(val_loader, desc="pass A (batch stats)", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        activations.clear()
        model.zero_grad(set_to_none=True)
        if use_ampA:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                logits = model(xb); loss = F.cross_entropy(logits, yb)
        else:
            logits = model(xb); loss = F.cross_entropy(logits, yb)
        B = xb.size(0)
        with torch.no_grad():
            for n in layer_names:
                a = activations.get(n, None)
                if a is not None:
                    actnorm[n] += a.detach().view(B,-1).abs().mean(1).sum().item()
                    curm = a.mean(dim=0, keepdim=True, dtype=a.dtype).detach()
                    if mean_buf[n] is None: mean_buf[n]=curm
                    else: mean_buf[n] = (mean_buf[n]*mean_cnt + curm)/(mean_cnt+1)
        loss.backward()
        with torch.no_grad():
            for n in layer_names:
                a = activations.get(n, None); g = None if a is None else a.grad
                if (a is not None) and (g is not None):
                    taylorfo[n] += (a.detach().view(B,-1).abs()*g.detach().view(B,-1).abs()).sum(1).sum().item()
        mean_cnt += 1
        if options.quick: break

    base_acc = eval_acc(model, val_loader) if options.compute_delta_acc else float('nan')

    # Pass B: per-sample predictions and actuals
    processed=0
    delta_acc_rows=[]
    for xb,yb in tqdm(val_loader, desc="pass B (per-sample)", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        B = xb.size(0)
        for i in range(B):
            if processed >= options.sample_cap: break
            xi = xb[i:i+1].detach()

            activations.clear()
            f_base = model(xi)

            for n in layer_names:
                a = activations.get(n, None)
                if a is None: continue
                a_i = a[0:1].detach().requires_grad_(True)

                # g(a_rep): rerun with this layer's output replaced
                def g(a_rep):
                    return forward_with_replacement(n, xi, lambda out: a_rep.to(out.dtype))

                # pred (zero baseline): v = a_i
                jv = safe_jvp_with_fallback(g, a_point=a_i, v_dir=a_i, use_jvp=True, fd_eps=1e-3)
                pred_zero[n] += float(jv.norm().item())

                # pred (mean baseline)
                if options.do_mean_baseline and (mean_buf[n] is not None):
                    a_bar = mean_buf[n].to(a_i.dtype); v_m = (a_i - a_bar)
                    jv_m = safe_jvp_with_fallback(g, a_point=a_i, v_dir=v_m, use_jvp=True, fd_eps=1e-3)
                    pred_mean[n] += float(jv_m.norm().item())

                # actual via zeroing (graph-preserving): replace with o*0
                f_zero = forward_with_replacement(n, xi, lambda o: o * 0.0)
                # logits L2
                act_logitsL2[n] += float((f_base - f_zero).norm().item())
                # softmax-KL
                act_softKL[n]   += float(softmax_kl(f_base, f_zero).mean().item())

                # store pairs for corr vs samples (use KL as signal)
                pair_zero[n].append((float(jv.norm().item()), float(softmax_kl(f_base, f_zero).mean().item())))
                if options.do_mean_baseline and (mean_buf[n] is not None):
                    pair_mean[n].append((float(jv_m.norm().item()), float(softmax_kl(f_base, f_zero).mean().item())))

            processed += 1
        if processed >= options.sample_cap: break

    # Delta accuracy per layer (task-level)
    if options.compute_delta_acc:
        for n,_ in layer_list:
            h = layer_by[n].register_forward_hook(lambda m,i,o: o * 0.0)
            acc_after = eval_acc(model, val_loader)
            h.remove()
            delta_acc_rows.append({"layer": n, "delta_acc": float(base_acc - acc_after)})
    delta_acc_df = pd.DataFrame(delta_acc_rows).set_index("layer") if options.compute_delta_acc else pd.DataFrame()

    # Aggregate per layer
    N = max(1, processed)
    results={}
    for n in layer_names:
        results[n] = {
            "S_GNDI_zero": pred_zero[n]/N,
            "S_GNDI_mean": pred_mean[n]/N if options.do_mean_baseline else float("nan"),
            "actual_logits_L2": act_logitsL2[n]/N,
            "actual_softmax_KL": act_softKL[n]/N,
            "weight_l1": weight_l1[n],
            "act_norm": actnorm[n]/max(1, mean_cnt),
            "taylor_fo": taylorfo[n]/max(1, mean_cnt),
            "params": params_count[n],
            "S_GNDI_zero_per_param": (pred_zero[n]/N) / max(1, params_count[n]),
            "samples": N
        }

    # Correlations vs actual (use KL as primary; logits L2 kept too)
    names = list(results.keys())
    pred_vec = np.array([results[n]["S_GNDI_zero"] for n in names], float)
    act_kl   = np.array([results[n]["actual_softmax_KL"] for n in names], float)
    act_l2   = np.array([results[n]["actual_logits_L2"] for n in names], float)

    def corr_pair(x,y):
        if len(x) >= 3 and (not np.allclose(x, x[0])) and (not np.allclose(y, y[0])):
            return (pearsonr(x,y)[0], spearmanr(x,y)[0], kendalltau(x,y)[0])
        return (float('nan'), float('nan'), float('nan'))

    pr, sr, tr = corr_pair(pred_vec, act_kl)

    # calibration act_kl ≈ beta * pred
    X = pred_vec.reshape(-1,1)
    try:
        lr = LinearRegression(fit_intercept=False).fit(X, act_kl)
        beta = float(lr.coef_[0]); r2 = float(lr.score(X, act_kl))
    except Exception:
        beta, r2 = float('nan'), float('nan')

    # AUC for critical-layer detection (top 25% by KL)
    k = max(1, int(0.25 * len(names)))
    thresh = np.partition(act_kl, -k)[-k]
    y_true = (act_kl >= thresh).astype(int)
    auc = float('nan')
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        try: auc = float(roc_auc_score(y_true, pred_vec))
        except Exception: pass

    # Correlation vs samples (KL target)
    sizes=[32,64,128,256,512,1024]
    corr_rows=[]
    for Nsub in sizes:
        P,A = [],[]
        for n in names:
            arr = np.array(pair_zero[n], float)
            if arr.shape[0]==0: continue
            arr = arr[:min(Nsub, arr.shape[0])]
            P.append(arr[:,0].mean()); A.append(arr[:,1].mean())
        if len(P)>=3:
            p = pearsonr(P,A)[0]; s = spearmanr(P,A)[0]; t = kendalltau(P,A)[0]
        else:
            p = s = t = float('nan')
        corr_rows.append({"samples":Nsub, "pearson_r":p, "spearman_rho":s, "kendall_tau":t, "variant":"zero"})

    for h in handles: h.remove()

    stats = {
        "pearson_r": float(pr),
        "spearman_rho": float(sr),
        "kendall_tau": float(tr),
        "beta_slope": float(beta),
        "r2": float(r2),
        "auc_critical": float(auc),
        "base_accuracy": float(base_acc) if options.compute_delta_acc else float('nan')
    }
    corr_df = pd.DataFrame(corr_rows)
    return results, stats, corr_df, delta_acc_df

# ---------------------- Ranking / tiny fine-tune ----------------------

def tiny_ft(model: nn.Module, loader: DataLoader, steps: int=10, lr: float=5e-4, subset: int=1024) -> None:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    n=0
    with torch.set_grad_enabled(True):
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb); loss = F.cross_entropy(out,yb)
            loss.backward(); opt.step()
            n += xb.size(0)
            if n >= subset: break
    model.eval()

def ranking_curve(model: nn.Module, val_loader: DataLoader, layer_list: List[Tuple[str, nn.Module]],
                  results: Dict[str, dict], metric_key: str,
                  tiny_finetune_loader: DataLoader=None) -> Tuple[List[float], List[float], float]:
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
                handles.append(m.register_forward_hook(lambda module,i,o: o * 0.0))
        if tiny_finetune_loader is not None:
            tiny_ft(model, tiny_finetune_loader, steps=10, lr=5e-4, subset=1024)
        acc = eval_acc(model, val_loader)
        for h in handles: h.remove()
        fracs.append(k/total); accs.append(acc)
    return fracs, accs, base

def auc_of_curve(fracs: List[float], accs: List[float]) -> float:
    x = np.array(fracs, dtype=float)
    y = np.array(accs, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y=y, x=x))
    return float(np.trapz(y=y, x=x))

# ---------------------- Intervention sweep ----------------------

@torch.no_grad()
def intervention_sweep(model: nn.Module, layer_by: Dict[str, nn.Module],
                       select_layers: List[str], xi: torch.Tensor,
                       alpha_list=(0.0,0.25,0.5,0.75,1.0)) -> Dict[str, List[float]]:
    model.eval()
    out={}; f0 = model(xi)
    for name in select_layers:
        h = layer_by[name].register_forward_hook(lambda m,i,o: alpha_list[0] * o)
        deltas=[]
        for alpha in alpha_list:
            h.remove()
            h = layer_by[name].register_forward_hook(lambda m,i,o,alpha=alpha: (alpha * o))
            f = model(xi)
            deltas.append((f0 - f).norm().item())
        h.remove()
        out[name]=deltas
    return out

# ---------------------- Report (Latin-1 safe) ----------------------

class Report:
    def __init__(self, out_path: str):
        self.pdf = FPDF()
        self.out_path = out_path
    @staticmethod
    def _latin1_safe(text: str) -> str:
        repl = {"β":"beta","ρ":"rho","τ":"tau","Δ":"Delta","→":"->","≤":"<=","≥":">=","·":"*","–":"-","—":"-","…":"...","×":"x"}
        for k,v in repl.items(): text = text.replace(k,v)
        return text.encode("latin-1","replace").decode("latin-1")
    def add_title(self, text: str):
        self.pdf.add_page(); self.pdf.set_font("Arial","B",18)
        self.pdf.cell(0,10,txt=self._latin1_safe(text), ln=1)
    def add_paragraph(self, text: str, size: int=11):
        self.pdf.set_font("Arial","",size)
        for line in textwrap.wrap(self._latin1_safe(text), width=105):
            self.pdf.cell(0,6,txt=line, ln=1)
        self.pdf.ln(2)
    def add_image(self, path: str, w: int=190):
        self.pdf.image(path, x=10, y=None, w=w); self.pdf.ln(5)
    def add_subtitle(self, text: str):
        self.pdf.set_font("Arial","B",14)
        self.pdf.cell(0,8,txt=self._latin1_safe(text), ln=1)
    def add_kv(self, pairs: List[Tuple[str,str]]):
        self.pdf.set_font("Arial","",11)
        for k,v in pairs: self.pdf.cell(0,6,txt=self._latin1_safe(f"{k}: {v}"), ln=1)
        self.pdf.ln(2)
    def output(self): self.pdf.output(self.out_path)

def save_table_image(df: pd.DataFrame, path: str, title: str=None, max_rows: int=30):
    df_show = df.head(max_rows)
    fig, ax = plt.subplots(figsize=(min(12, max(6, 0.4*df_show.shape[1]+4)), min(12, max(2, 0.4*df_show.shape[0]+2))))
    ax.axis('off')
    if title: ax.set_title(title)
    table = ax.table(cellText=df_show.values, colLabels=df_show.columns, loc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.2)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

# ---------------------- Run one experiment ----------------------

def run_once(args, seed: int, tag: str):
    set_seed(seed)
    train_loader, val_loader = get_loaders(args.dataset, args.batch, args.quick)
    model_name = args.model
    model = get_model(model_name, num_classes=10)
    ckpt = args.checkpoint or os.path.join(RESULTS_DIR, f"{model_name}_{args.dataset}_seed{seed}.pt")
    if args.train or not os.path.exists(ckpt):
        train(model, train_loader, val_loader, epochs=args.epochs, ckpt=ckpt)
    else:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE); model.eval()
    base_acc = eval_acc(model, val_loader)

    layer_list = collect_layers(model, take_every=args.take_every)

    opts = MetricOptions(quick=args.quick, sample_cap=args.sample_cap,
                         do_mean_baseline=args.mean_baseline, compute_delta_acc=True)
    results, stats, corr_df, delta_acc_df = compute_all_metrics(model, val_loader, layer_list, opts)

    # SNIP/SynFlow
    snip = snip_score(model, train_loader, layer_list, max_batches=1)
    input_shape=(1,3,32,32)
    synf = synflow_score(model, layer_list, input_shape)
    for n,_ in layer_list:
        results[n]["snip"] = snip.get(n,0.0)
        results[n]["synflow"] = synf.get(n,0.0)

    # Save metrics
    metrics_rows=[]
    for n,_ in layer_list:
        r = results[n].copy(); r["layer"]=n; metrics_rows.append(r)
    metrics_df = pd.DataFrame(metrics_rows).set_index("layer")
    metrics_json = os.path.join(RESULTS_DIR, f"metrics_{tag}_seed{seed}.json")
    with open(metrics_json,"w") as f: json.dump({"stats":stats, "results":results, "base_acc":base_acc}, f, indent=2)
    metrics_csv  = os.path.join(RESULTS_DIR, f"metrics_{tag}_seed{seed}.csv"); metrics_df.to_csv(metrics_csv)
    corr_csv     = os.path.join(RESULTS_DIR, f"correlations_{tag}_seed{seed}.csv"); corr_df.to_csv(corr_csv, index=False)
    if not delta_acc_df.empty:
        delta_acc_csv = os.path.join(RESULTS_DIR, f"delta_acc_{tag}_seed{seed}.csv"); delta_acc_df.to_csv(delta_acc_csv)

    # Top-k curves (no fine-tune)
    rank_metrics = ["S_GNDI_zero"]
    if args.mean_baseline: rank_metrics.append("S_GNDI_mean")
    rank_metrics += ["weight_l1","act_norm","taylor_fo","snip","synflow"]

    plt.figure(figsize=(7,5)); auc_rows=[]
    for mk in rank_metrics:
        fr, ac, base = ranking_curve(model, val_loader, layer_list, results, mk, tiny_finetune_loader=None)
        plt.plot(fr, ac, label=mk)
        auc_rows.append({"metric":mk, "auc": auc_of_curve(fr, ac)})
    plt.axhline(base, linestyle="--", linewidth=1); plt.legend()
    p_topk = os.path.join(RESULTS_DIR, f"topk_{tag}_seed{seed}.png")
    plt.tight_layout(); plt.savefig(p_topk, dpi=160); plt.close()
    auc_df = pd.DataFrame(auc_rows); auc_csv = os.path.join(RESULTS_DIR, f"topk_auc_{tag}_seed{seed}.csv"); auc_df.to_csv(auc_csv, index=False)

    # Top-k with tiny fine-tune (G-NDI vs SNIP)
    plt.figure(figsize=(7,5))
    fr, ac, base = ranking_curve(model, val_loader, layer_list, results, "S_GNDI_zero", tiny_finetune_loader=train_loader)
    plt.plot(fr, ac, label="G-NDI (zero) + tiny_ft")
    fr2, ac2, _ = ranking_curve(model, val_loader, layer_list, results, "snip", tiny_finetune_loader=train_loader)
    plt.plot(fr2, ac2, label="SNIP + tiny_ft")
    plt.axhline(base, linestyle="--", linewidth=1); plt.legend()
    p_topk_ft = os.path.join(RESULTS_DIR, f"topk_tinyft_{tag}_seed{seed}.png")
    plt.tight_layout(); plt.savefig(p_topk_ft, dpi=160); plt.close()

    # Scatter: G-NDI vs softmax-KL
    names=list(metrics_df.index)
    x = metrics_df["S_GNDI_zero"].values; y = metrics_df["actual_softmax_KL"].values
    plt.figure(figsize=(6,5)); plt.scatter(x,y,s=22)
    for i,n in enumerate(names): plt.text(x[i],y[i],n,fontsize=6)
    plt.xlabel("G-NDI (zero)"); plt.ylabel("Actual softmax-KL")
    p_sc0 = os.path.join(RESULTS_DIR, f"scatter_zero_{tag}_seed{seed}.png")
    plt.tight_layout(); plt.savefig(p_sc0, dpi=160); plt.close()

    # Corr vs samples
    cdf = pd.read_csv(corr_csv)
    plt.figure(figsize=(7,5))
    for variant in sorted(cdf["variant"].unique()):
        sub = cdf[cdf["variant"]==variant]
        plt.plot(sub["samples"], sub["spearman_rho"], marker="o", label=variant)
    plt.xlabel("samples"); plt.ylabel("spearman rho"); plt.legend()
    p_cvs = os.path.join(RESULTS_DIR, f"corr_vs_samples_{tag}_seed{seed}.png")
    plt.tight_layout(); plt.savefig(p_cvs, dpi=160); plt.close()

    # Normalized bar
    sel_cols = ["S_GNDI_zero","actual_softmax_KL","actual_logits_L2","weight_l1","act_norm","taylor_fo","snip","synflow"]
    df_norm = (metrics_df[sel_cols] - metrics_df[sel_cols].min()) / (metrics_df[sel_cols].max() - metrics_df[sel_cols].min() + 1e-12)
    df_norm.plot.bar(figsize=(12,4))
    p_bar = os.path.join(RESULTS_DIR, f"metrics_bar_{tag}_seed{seed}.png")
    plt.tight_layout(); plt.savefig(p_bar, dpi=160); plt.close()

    # Tables
    save_table_image(metrics_df.sort_values("S_GNDI_zero", ascending=False).round(4).reset_index(),
                     os.path.join(RESULTS_DIR, f"table_metrics_{tag}_seed{seed}.png"), title="Per-layer metrics (sorted by G-NDI zero)")
    save_table_image(auc_df.round(4), os.path.join(RESULTS_DIR, f"table_auc_{tag}_seed{seed}.png"), title="Top-k AUC per metric")
    save_table_image(cdf.round(4), os.path.join(RESULTS_DIR, f"table_corr_vs_samples_{tag}_seed{seed}.png"), title="Correlation vs samples")
    if not delta_acc_df.empty:
        save_table_image(delta_acc_df.round(4).reset_index(), os.path.join(RESULTS_DIR, f"table_delta_acc_{tag}_seed{seed}.png"), title="Delta accuracy per layer")

    # Intervention sweep
    top = metrics_df["S_GNDI_zero"].sort_values(ascending=False)
    if len(top) >= 3:
        sel_names = [top.index[0], top.index[len(top)//2], top.index[-1]]
        xi = next(iter(val_loader))[0][:1].to(DEVICE)
        layer_by = {n:m for n,m in layer_list}
        sweep = intervention_sweep(model, layer_by, sel_names, xi)
        plt.figure(figsize=(7,5))
        alpha = [0.0,0.25,0.5,0.75,1.0]
        for n in sel_names:
            plt.plot(alpha, sweep[n], marker="o", label=n)
        plt.xlabel("alpha (fraction kept)"); plt.ylabel("||f0 - f_alpha||"); plt.legend()
        p_sweep = os.path.join(RESULTS_DIR, f"sweep_{tag}_seed{seed}.png")
        plt.tight_layout(); plt.savefig(p_sweep, dpi=160); plt.close()
    else:
        p_sweep = None

    # PDF report
    out_pdf = os.path.join(RESULTS_DIR, f"report_full_{tag}_seed{seed}.pdf")
    rep = Report(out_pdf)
    rep.add_title("G-NDI Comprehensive Report")
    rep.add_kv([("Timestamp", now()),
                ("Device", DEVICE),
                ("Platform", f"{platform.system()} {platform.release()}"),
                ("Dataset", args.dataset),
                ("Model", model_name),
                ("Base accuracy", f"{base_acc:.4f}"),
                ("Layers evaluated", str(len(layer_list))),
                ("Sample cap (Pass B)", str(args.sample_cap)),
                ("Take-every", str(args.take_every))])
    rep.add_subtitle("Summary")
    rep.add_paragraph(
        "We evaluate General Neuro-Dynamic Importance (G-NDI) against strong baselines on a deep network. "
        "Predicted impact uses a first-order causal proxy JvP; actual impact uses delta accuracy and softmax-KL "
        "under layer zeroing with graph-preserving hooks. Mean-replacement and pruning curves are included."
    )
    rep.add_subtitle("Key Statistics (vs softmax-KL)")
    rep.add_kv([("Pearson r", f"{stats['pearson_r']:.4f}"),
                ("Spearman rho", f"{stats['spearman_rho']:.4f}"),
                ("Kendall tau", f"{stats['kendall_tau']:.4f}"),
                ("Calibration slope (beta)", f"{stats['beta_slope']:.4f}"),
                ("Calibration R^2", f"{stats['r2']:.4f}"),
                ("AUC (critical-layer detection)", f"{stats['auc_critical']:.4f}")])
    rep.add_subtitle("Top-k pruning curves (no fine-tune)"); rep.add_image(p_topk)
    rep.add_subtitle("Top-k curves with tiny fine-tune (G-NDI vs SNIP)"); rep.add_image(p_topk_ft)
    rep.add_subtitle("Predicted vs Actual (softmax-KL scatter)"); rep.add_image(p_sc0)
    rep.add_subtitle("Correlation vs Samples (Spearman)"); rep.add_image(p_cvs)
    rep.add_subtitle("Normalized Metrics per Layer"); rep.add_image(p_bar)
    rep.add_subtitle("Tables")
    rep.add_image(os.path.join(RESULTS_DIR, f"table_metrics_{tag}_seed{seed}.png"))
    rep.add_image(os.path.join(RESULTS_DIR, f"table_auc_{tag}_seed{seed}.png"))
    rep.add_image(os.path.join(RESULTS_DIR, f"table_corr_vs_samples_{tag}_seed{seed}.png"))
    if not delta_acc_df.empty:
        rep.add_image(os.path.join(RESULTS_DIR, f"table_delta_acc_{tag}_seed{seed}.png"))
    if p_sweep is not None:
        rep.add_subtitle("Intervention sweep (Delta vs alpha)"); rep.add_image(p_sweep)
    rep.output()

    return {"pdf": out_pdf, "metrics_json": metrics_json, "metrics_csv": metrics_csv, "corr_csv": corr_csv}

# ---------------------- Seeds aggregation ----------------------

def aggregate_seeds(arts: List[dict]) -> str:
    rows=[]
    for a in arts:
        jd = json.load(open(a["metrics_json"]))
        s = jd["stats"]; base = jd.get("base_acc", float('nan'))
        rows.append({"pearson_r":s.get("pearson_r",np.nan),
                     "spearman_rho":s.get("spearman_rho",np.nan),
                     "kendall_tau":s.get("kendall_tau",np.nan),
                     "beta_slope":s.get("beta_slope",np.nan),
                     "r2":s.get("r2",np.nan),
                     "auc_critical":s.get("auc_critical",np.nan),
                     "base_acc":base})
    df=pd.DataFrame(rows)
    agg = df.agg(["mean","std"])
    out=os.path.join(RESULTS_DIR, "seed_summary.csv"); agg.to_csv(out)
    return out

# ---------------------- CLI ----------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    ap.add_argument("--model", choices=["resnet18","small_cnn"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--sample-cap", type=int, default=256)
    ap.add_argument("--take-every", type=int, default=1)
    ap.add_argument("--mean-baseline", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    args=ap.parse_args()

    arts=[]
    tag=f"{args.dataset}_{args.model}"
    for s in range(args.seeds):
        arts.append(run_once(args, seed=s, tag=tag))

    if args.seeds > 1:
        summary = aggregate_seeds(arts)
        print("Seed summary CSV:", summary)
    print("Report:", arts[0]["pdf"])
    print("Artifacts in:", RESULTS_DIR)
    print("Device:", DEVICE)

if __name__=="__main__":
    main()
