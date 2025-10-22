#!/usr/bin/env python3
"""
gndi_overnight.py
Fast, diverse G-NDI benchmark across CV and NLP with small models & subsetted datasets.
Outputs:
  - ./overnight_results/all_results.json  (all raw + summary metrics)
  - ./overnight_results/overnight_report.pdf (short textual summary + tables + figures)

Default suite (~overnight on laptop):
  CV: MNIST (SmallCNN), FashionMNIST (SmallCNN), CIFAR-10 (SmallCNN)
  NLP: AG_NEWS (TextCNN)
All tasks use small subsets; epochs are modest; per-sample G-NDI caps are enforced.

You can add/remove tasks in BUILD_TASKS() below.
"""

import os, json, time, random, argparse, textwrap, platform
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from datasets import load_dataset
from collections import Counter
import re

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from fpdf import FPDF

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ------------------------- Util: device, seed, IO -------------------------

def pick_device() -> str:
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

DEVICE = pick_device()
OUTDIR = "overnight_results"
os.makedirs(OUTDIR, exist_ok=True)

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# ------------------------- CV: models & loaders -------------------------

class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(128, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)

def get_cv_loaders(task: str, batch=128, quick=False, train_size=5000, test_size=2000):
    pin=False
    if task == "mnist":
        tr = te = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train = torchvision.datasets.MNIST("../data", True, download=True, transform=tr)
        test  = torchvision.datasets.MNIST("../data", False, download=True, transform=te)
        in_ch=1; nclass=10
    elif task == "fashionmnist":
        tr = te = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
        train = torchvision.datasets.FashionMNIST("../data", True, download=True, transform=tr)
        test  = torchvision.datasets.FashionMNIST("../data", False, download=True, transform=te)
        in_ch=1; nclass=10
    elif task == "cifar10":
        tr = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                        T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        te = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        train = torchvision.datasets.CIFAR10("../data", True, download=True, transform=tr)
        test  = torchvision.datasets.CIFAR10("../data", False, download=True, transform=te)
        in_ch=3; nclass=10
    else:
        raise ValueError(task)

    if quick:
        train = Subset(train, list(range(min(train_size//2, len(train)))))
        test  = Subset(test,  list(range(min(test_size//2,  len(test)))))
    else:
        train = Subset(train, list(range(min(train_size, len(train)))))
        test  = Subset(test,  list(range(min(test_size,  len(test)))))

    return (DataLoader(train, batch, True, num_workers=4, pin_memory=pin),
            DataLoader(test,  batch, False, num_workers=4, pin_memory=pin),
            in_ch, nclass)

# ------------------------- NLP: tiny TextCNN pipeline -------------------------

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=4, kernel_sizes=(3,4,5), num_channels=64, pad_idx=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_channels, k) for k in kernel_sizes])
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)
    def forward(self, ids):
        # ids: [B, T]
        x = self.emb(ids)           # [B, T, E]
        x = x.transpose(1, 2)       # [B, E, T]
        xs = [F.relu(conv(x)) for conv in self.convs]   # list of [B, C, T']
        xs = [F.max_pool1d(t, t.size(2)).squeeze(2) for t in xs]  # [B, C]
        x = torch.cat(xs, dim=1)    # [B, C*len(K)]
        return self.fc(x)

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def simple_tokenize(text: str):
    # Lowercase + keep alphanum and apostrophes; split like a basic_english tokenizer
    return _TOKEN_RE.findall(text.lower())

def build_vocab(texts, max_vocab_size=50000, specials=("<unk>", "<pad>")):
    cnt = Counter()
    for t in texts:
        cnt.update(simple_tokenize(t))
    # Most common terms
    common = [w for w, _ in cnt.most_common(max_vocab_size - len(specials))]
    stoi = {tok: i for i, tok in enumerate(specials + tuple(common))}
    unk_id = stoi[specials[0]]
    pad_id = stoi[specials[1]]
    return stoi, unk_id, pad_id

def encode_text(text: str, stoi: dict, unk_id: int, pad_id: int, max_len: int):
    ids = [stoi.get(tok, unk_id) for tok in simple_tokenize(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [pad_id] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def yield_tokens_agnews(tokenizer, data_iter, max_items=None):
    n=0
    for (label, text) in data_iter:
        yield tokenizer(text)
        n+=1
        if (max_items is not None) and (n>=max_items): break

def build_agnews_loaders(batch=128, quick=False, train_size=8000, test_size=4000, max_len=128):
    """
    AG_NEWS via Hugging Face 'datasets' (no torchtext).
    - Small subsets for speed
    - Simple regex tokenizer + custom vocab
    - Returns PyTorch DataLoaders with fixed-length ID tensors
    """
    ds = load_dataset("ag_news")
    # Extract raw texts/labels
    train_raw = ds["train"]
    test_raw  = ds["test"]

    # Subset sizes
    if quick:
        train_size = max(1, train_size // 2)
        test_size  = max(1, test_size  // 2)

    # Slice
    train_raw = train_raw.select(range(min(train_size, len(train_raw))))
    test_raw  = test_raw.select(range(min(test_size,  len(test_raw))))

    # Build vocab on training texts
    train_texts = [train_raw[i]["text"] for i in range(len(train_raw))]
    stoi, unk_id, pad_id = build_vocab(train_texts, max_vocab_size=50000, specials=("<unk>", "<pad>"))

    # Encode
    Xtr = torch.stack([encode_text(train_raw[i]["text"], stoi, stoi["<unk>"], stoi["<pad>"], max_len) for i in range(len(train_raw))])
    ytr = torch.tensor([int(train_raw[i]["label"]) for i in range(len(train_raw))], dtype=torch.long)
    Xte = torch.stack([encode_text(test_raw[i]["text"],  stoi, stoi["<unk>"], stoi["<pad>"], max_len) for i in range(len(test_raw))])
    yte = torch.tensor([int(test_raw[i]["label"]) for i in range(len(test_raw))], dtype=torch.long)

    tr_ds = torch.utils.data.TensorDataset(Xtr, ytr)
    te_ds = torch.utils.data.TensorDataset(Xte, yte)

    # MPS tends to prefer num_workers=0
    tr = DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=0)
    te = DataLoader(te_ds, batch_size=batch, shuffle=False, num_workers=0)

    vocab_size = len(stoi)
    num_classes = 4
    return tr, te, vocab_size, num_classes, pad_id

# ------------------------- Training & eval -------------------------

def train_simple(model, train_loader, val_loader, epochs=6, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = (DEVICE == "mps")
    for ep in range(epochs):
        model.train()
        for xb,yb in tqdm(train_loader, desc=f"train ep{ep+1}/{epochs}", leave=False):
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    out = model(xb); loss = F.cross_entropy(out,yb)
            else:
                out = model(xb); loss = F.cross_entropy(out,yb)
            loss.backward(); opt.step()
    return model

@torch.no_grad()
def eval_acc(model, loader):
    model.eval(); c=t=0
    for xb,yb in loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb).argmax(1)
        c += (pred==yb).sum().item(); t += yb.size(0)
    return c/max(1,t)

# ------------------------- G-NDI core -------------------------

def safe_jvp_with_fallback(g_func, a_point, v_dir, use_jvp=True, fd_eps=1e-3):
    if use_jvp and hasattr(torch.autograd.functional, "jvp"):
        try:
            _, jv = torch.autograd.functional.jvp(
                func=g_func, inputs=(a_point,), v=(v_dir,), create_graph=False, strict=True
            )
            return jv
        except Exception:
            pass
    denom = (v_dir.norm().detach().item() + 1e-12)
    eps = fd_eps / denom
    y0 = g_func(a_point).detach()
    y1 = g_func(a_point + eps * v_dir).detach()
    return (y1 - y0) / eps

def softmax_kl(p_logits, q_logits):
    p = F.softmax(p_logits, dim=-1)
    logp = F.log_softmax(p_logits, dim=-1)
    logq = F.log_softmax(q_logits, dim=-1)
    return (p * (logp - logq)).sum(dim=-1)

def collect_layers(model, take_every=1):
    layers=[]; k=0
    for n,m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if k % max(1, take_every) == 0:
                layers.append((n,m))
            k += 1
    return layers

def compute_gndi_suite(model, val_loader, layer_list,
                       sample_cap=256, do_mean=True, quick=False):
    model.eval()
    layer_names=[n for n,_ in layer_list]
    layer_by={n:m for n,m in layer_list}

    activations={}
    def hook(name):
        def _h(m,i,o):
            activations[name]=o
            if o.requires_grad:
                try: o.retain_grad()
                except: pass
            return o
        return _h
    hooks=[m.register_forward_hook(hook(n)) for n,m in layer_list]

    # gather mean activations & taylor & actnorm
    actnorm={n:0.0 for n in layer_names}
    taylor ={n:0.0 for n in layer_names}
    meanbuf={n:None for n in layer_names}; mean_cnt=0

    # first pass
    for xb,yb in tqdm(val_loader, desc="pass A (stats)", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        activations.clear()
        model.zero_grad(set_to_none=True)
        logits = model(xb)
        B = xb.size(0)
        with torch.no_grad():
            for n in layer_names:
                a = activations.get(n, None)
                if a is not None:
                    actnorm[n] += a.detach().view(B,-1).abs().mean(1).sum().item()
                    curm = a.mean(dim=0, keepdim=True, dtype=a.dtype).detach()
                    if meanbuf[n] is None: meanbuf[n]=curm
                    else: meanbuf[n] = (meanbuf[n]*mean_cnt + curm)/(mean_cnt+1)
        (F.cross_entropy(logits,yb)).backward()
        with torch.no_grad():
            for n in layer_names:
                a = activations.get(n, None); g = None if a is None else a.grad
                if (a is not None) and (g is not None):
                    taylor[n] += (a.detach().view(B,-1).abs()*g.detach().view(B,-1).abs()).sum(1).sum().item()
        mean_cnt += 1
        if quick: break

    # predictions and actuals
    pred_zero={n:0.0 for n in layer_names}
    pred_mean={n:0.0 for n in layer_names}
    act_kl   ={n:0.0 for n in layer_names}
    act_l2   ={n:0.0 for n in layer_names}
    processed=0

    def forward_with_replacement(name, x1, replace_fn):
        mod = layer_by[name]
        h = mod.register_forward_hook(lambda m,i,o: replace_fn(o))
        try:
            y = model(x1)
        finally:
            h.remove()
        return y

    for xb,yb in tqdm(val_loader, desc="pass B (per-sample)", leave=False):
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        B = xb.size(0)
        for i in range(B):
            if processed >= sample_cap: break
            xi = xb[i:i+1].detach()

            activations.clear()
            f0 = model(xi)

            for n in layer_names:
                a = activations.get(n, None)
                if a is None: continue
                a_i = a[0:1].detach().requires_grad_(True)
                def g(a_rep): return forward_with_replacement(n, xi, lambda out: a_rep.to(out.dtype))

                jv = safe_jvp_with_fallback(g, a_point=a_i, v_dir=a_i, use_jvp=True, fd_eps=1e-3)
                pred_zero[n] += float(jv.norm().item())

                if do_mean and (meanbuf[n] is not None):
                    a_bar = meanbuf[n].to(a_i.dtype); v_m = (a_i - a_bar)
                    jv_m = safe_jvp_with_fallback(g, a_point=a_i, v_dir=v_m, use_jvp=True, fd_eps=1e-3)
                    pred_mean[n] += float(jv_m.norm().item())

                fz = forward_with_replacement(n, xi, lambda o: o * 0.0)
                act_l2[n] += float((f0 - fz).norm().item())
                act_kl[n] += float(softmax_kl(f0, fz).mean().item())

            processed += 1
        if processed >= sample_cap: break

    # delta accuracy per layer
    base_acc = eval_acc(model, val_loader)
    delta_rows=[]
    for n,_ in layer_list:
        h = layer_by[n].register_forward_hook(lambda m,i,o: o * 0.0)
        acc_after = eval_acc(model, val_loader)
        h.remove()
        delta_rows.append({"layer":n, "delta_acc": float(base_acc - acc_after)})
    delta_df = pd.DataFrame(delta_rows).set_index("layer")

    N = max(1, processed)
    results={}
    for n,_ in layer_list:
        results[n] = {
            "S_GNDI_zero": pred_zero[n]/N,
            "S_GNDI_mean": pred_mean[n]/N if do_mean else float("nan"),
            "actual_softmax_KL": act_kl[n]/N,
            "actual_logits_L2": act_l2[n]/N,
            "act_norm": actnorm[n]/max(1, mean_cnt),
            "taylor_fo": taylor[n]/max(1, mean_cnt),
            "samples": N
        }

    # correlations vs KL (primary)
    names = list(results.keys())
    pred = np.array([results[n]["S_GNDI_zero"] for n in names], float)
    actk = np.array([results[n]["actual_softmax_KL"] for n in names], float)

    def corr_pair(x,y):
        if len(x)>=3 and (not np.allclose(x,x[0])) and (not np.allclose(y,y[0])):
            return (pearsonr(x,y)[0], spearmanr(x,y)[0], kendalltau(x,y)[0])
        return (float('nan'),)*3

    pr, sr, tr = corr_pair(pred, actk)

    # calibration
    try:
        X = pred.reshape(-1,1)
        lr = LinearRegression(fit_intercept=False).fit(X, actk)
        beta = float(lr.coef_[0]); r2 = float(lr.score(X, actk))
    except Exception:
        beta=r2=float('nan')

    # AUC for top 25% actual
    k = max(1, int(0.25*len(names)))
    thresh = np.partition(actk, -k)[-k]
    y_true = (actk >= thresh).astype(int)
    auc = float('nan')
    try:
        if (y_true.sum()>0) and (y_true.sum()<len(y_true)):
            auc = float(roc_auc_score(y_true, pred))
    except Exception:
        pass

    # clean hooks
    for h in hooks: h.remove()

    # pack
    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    return {
        "metrics_df": metrics_df,
        "delta_acc_df": delta_df,
        "stats": {
            "base_acc": float(base_acc),
            "pearson_r": float(pr),
            "spearman_rho": float(sr),
            "kendall_tau": float(tr),
            "beta_slope": float(beta),
            "r2": float(r2),
            "auc_critical": float(auc)
        }
    }

# ------------------------- Report helpers -------------------------

def latin1(text: str) -> str:
    rep = {"β":"beta","ρ":"rho","τ":"tau","Δ":"Delta","→":"->","≤":"<=","≥":">=","·":"*","–":"-","—":"-","…":"...","×":"x"}
    for k,v in rep.items(): text = text.replace(k,v)
    return text.encode("latin-1","replace").decode("latin-1")

class Report:
    def __init__(self, out_pdf):
        self.pdf = FPDF()
        self.out_pdf = out_pdf
    def title(self, t):
        self.pdf.add_page(); self.pdf.set_font("Arial","B",18)
        self.pdf.cell(0,10,txt=latin1(t), ln=1)
    def h2(self, t):
        self.pdf.set_font("Arial","B",14)
        self.pdf.cell(0,8,txt=latin1(t), ln=1)
    def p(self, t):
        self.pdf.set_font("Arial","",11)
        for line in textwrap.wrap(latin1(t), width=105):
            self.pdf.cell(0,6,txt=line, ln=1)
        self.pdf.ln(2)
    def img(self, path, w=190):
        self.pdf.image(path, x=10, y=None, w=w); self.pdf.ln(5)
    def kv(self, rows):
        self.pdf.set_font("Arial","",11)
        for k,v in rows:
            self.pdf.cell(0,6,txt=latin1(f"{k}: {v}"), ln=1)
        self.pdf.ln(2)
    def out(self): self.pdf.output(self.out_pdf)

def save_table_image(df: pd.DataFrame, path: str, title: str=None, max_rows: int=30):
    df_show = df.head(max_rows)
    fig, ax = plt.subplots(figsize=(min(12, max(6, 0.4*df_show.shape[1]+4)),
                                    min(12, max(2, 0.4*df_show.shape[0]+2))))
    ax.axis('off')
    if title: ax.set_title(title)
    table = ax.table(cellText=df_show.values,
                     colLabels=df_show.columns,
                     rowLabels=df_show.index,
                     loc='center')
    table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.2)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

# ------------------------- Task registry -------------------------

@dataclass
class TaskCfg:
    kind: str          # "cv" or "nlp"
    name: str          # dataset/model label
    epochs: int
    train_size: int
    test_size: int
    batch: int

def BUILD_TASKS(args) -> List[TaskCfg]:
    # You can edit this to add/remove tasks.
    base = []
    # CV (SmallCNN everywhere for speed & consistency)
    base += [TaskCfg("cv","mnist",         args.cv_epochs,  5000, 2000, args.batch)]
    base += [TaskCfg("cv","fashionmnist",  args.cv_epochs,  5000, 2000, args.batch)]
    base += [TaskCfg("cv","cifar10",       args.cv_epochs,  8000, 3000, args.batch)]
    # NLP (AG_NEWS + TextCNN)
    base += [TaskCfg("nlp","ag_news",      args.nlp_epochs, 8000, 4000, args.batch)]
    if args.quick:
        # Halve sizes implicitly via loader; still run all tasks but faster
        pass
    return base

# ------------------------- Orchestrate per-task -------------------------

def run_task_cv(cfg: TaskCfg, args, seed=0):
    set_seed(seed)
    tr, te, in_ch, nclass = get_cv_loaders(cfg.name, cfg.batch, args.quick, cfg.train_size, cfg.test_size)
    model = SmallCNN(in_ch=in_ch, num_classes=nclass)
    train_simple(model, tr, te, epochs=cfg.epochs, lr=args.lr)
    layers = collect_layers(model, take_every=args.take_every)
    res = compute_gndi_suite(model, te, layers,
                             sample_cap=args.sample_cap,
                             do_mean=args.mean_baseline,
                             quick=args.quick)
    return {
        "task": cfg.name,
        "domain": "cv",
        "model": "SmallCNN",
        "stats": res["stats"],
        "metrics": res["metrics_df"].reset_index().rename(columns={"index":"layer"}).to_dict(orient="records"),
        "delta_acc": res["delta_acc_df"].reset_index().to_dict(orient="records")
    }

def run_task_nlp(cfg: TaskCfg, args, seed=0):
    set_seed(seed)
    tr, te, vocab_size, nclass, pad_id = build_agnews_loaders(cfg.batch, args.quick, cfg.train_size, cfg.test_size, max_len=args.max_len)
    model = TextCNN(vocab_size=vocab_size, embed_dim=args.emb_dim, num_classes=nclass, pad_idx=pad_id)
    train_simple(model, tr, te, epochs=cfg.epochs, lr=args.lr)
    layers = collect_layers(model, take_every=args.take_every)
    res = compute_gndi_suite(model, te, layers,
                             sample_cap=args.sample_cap,
                             do_mean=args.mean_baseline,
                             quick=args.quick)
    return {
        "task": cfg.name,
        "domain": "nlp",
        "model": f"TextCNN(e{args.emb_dim})",
        "stats": res["stats"],
        "metrics": res["metrics_df"].reset_index().rename(columns={"index":"layer"}).to_dict(orient="records"),
        "delta_acc": res["delta_acc_df"].reset_index().to_dict(orient="records")
    }

# ------------------------- Master run + report -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="halve subset sizes & stop stats pass early")
    ap.add_argument("--seeds", type=int, default=1)

    # speed / size knobs
    ap.add_argument("--cv-epochs", type=int, default=6)
    ap.add_argument("--nlp-epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=128)

    # G-NDI knobs
    ap.add_argument("--take-every", type=int, default=1)
    ap.add_argument("--sample-cap", type=int, default=256)
    ap.add_argument("--mean-baseline", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)

    args = ap.parse_args()

    tasks = BUILD_TASKS(args)
    all_runs=[]
    for seed in range(args.seeds):
        for cfg in tasks:
            print(f"\n=== Running [{cfg.kind.upper()}] {cfg.name} (seed {seed}) ===")
            if cfg.kind == "cv":
                runres = run_task_cv(cfg, args, seed=seed)
            else:
                runres = run_task_nlp(cfg, args, seed=seed)
            all_runs.append(runres)

    # Write JSON (full dump)
    json_path = os.path.join(OUTDIR, "all_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": now_str(),
            "device": DEVICE,
            "platform": f"{platform.system()} {platform.release()}",
            "runs": all_runs
        }, f, indent=2)

    # Build summary tables
    rows=[]
    for r in all_runs:
        s = r["stats"]
        rows.append({
            "domain": r["domain"],
            "task": r["task"],
            "model": r["model"],
            "base_acc": s.get("base_acc", np.nan),
            "pearson_r": s.get("pearson_r", np.nan),
            "spearman_rho": s.get("spearman_rho", np.nan),
            "kendall_tau": s.get("kendall_tau", np.nan),
            "beta_slope": s.get("beta_slope", np.nan),
            "r2": s.get("r2", np.nan),
            "auc_critical": s.get("auc_critical", np.nan),
        })
    summ_df = pd.DataFrame(rows)
    summ_csv = os.path.join(OUTDIR, "summary.csv"); summ_df.to_csv(summ_csv, index=False)

    # Per-run correlation plots
    figpaths=[]
    for r in all_runs:
        df = pd.DataFrame(r["metrics"])
        if ("S_GNDI_zero" in df.columns) and ("actual_softmax_KL" in df.columns):
            plt.figure(figsize=(6,5))
            plt.scatter(df["S_GNDI_zero"], df["actual_softmax_KL"], s=25)
            for i,row in df.iterrows():
                plt.text(row["S_GNDI_zero"], row["actual_softmax_KL"], row["layer"], fontsize=6)
            plt.xlabel("G-NDI (zero)"); plt.ylabel("Actual softmax-KL")
            path = os.path.join(OUTDIR, f"scatter_{r['domain']}_{r['task']}.png")
            plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
            figpaths.append((f"{r['domain']} / {r['task']}", path))

    # Convert summary to table image
    tab1 = os.path.join(OUTDIR, "table_summary.png")
    save_table_image(summ_df.round(4), tab1, title="Run-level summary")

    # PDF
    pdf_path = os.path.join(OUTDIR, "overnight_report.pdf")
    rep = Report(pdf_path)
    rep.title("G-NDI Overnight Benchmark — Summary Report")
    rep.kv([("Timestamp", now_str()),
            ("Device", DEVICE),
            ("Platform", f"{platform.system()} {platform.release()}"),
            ("Tasks run", str(len(all_runs))),
            ("Mean-baseline", str(bool(args.mean_baseline))),
            ("Sample cap per run", str(args.sample_cap)),
            ("Take-every", str(args.take_every))])
    rep.h2("What this is")
    rep.p("We evaluate a causal layer-importance proxy (G-NDI) across small CV and NLP models trained on subsetted datasets. "
          "For each model we compute predictions (JvP-based) and actuals (softmax-KL and delta-accuracy under layer zeroing), "
          "then report correlations, calibration, and AUC for critical-layer detection. "
          "The suite is sized for an overnight run on a laptop; it emphasizes breadth over base accuracy.")
    rep.h2("Run-level summary")
    rep.img(tab1, w=190)
    rep.h2("Scatter plots (G-NDI vs Actual KL)")
    for title, pth in figpaths:
        rep.p(title); rep.img(pth, w=170)
    rep.out()

    print("\nArtifacts written:")
    print(" JSON :", json_path)
    print(" CSV  :", summ_csv)
    print(" PDF  :", pdf_path)
    print(" PNGs :", ", ".join([p for _,p in figpaths]))

if __name__ == "__main__":
    main()
