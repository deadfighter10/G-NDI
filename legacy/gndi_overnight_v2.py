"""
G-NDI Overnight Evaluation v2
----------------------------------
Runs small CV & NLP models with multiple G-NDI variants (raw, log, L2, mean)
Generates all_results.json and a table-only overnight_report.pdf
"""

import os, time, json, math, argparse, random, re
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms, models
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr, kendalltau

# ------------------------------------------------------------
# Utility: device setup
# ------------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------------
# Model definitions
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TinyCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(embed_dim, 100, kernel_size=5)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        return self.fc(x)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h[-1])

# ------------------------------------------------------------
# NLP helpers (simple tokenizer/vocab)
# ------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def simple_tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())

def build_vocab(texts, max_vocab_size=50000):
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(simple_tokenize(t))
    specials = ["<unk>", "<pad>"]
    common = [w for w, _ in cnt.most_common(max_vocab_size - len(specials))]
    stoi = {tok: i for i, tok in enumerate(specials + common)}
    return stoi, stoi["<unk>"], stoi["<pad>"]

def encode_text(text, stoi, unk, pad, max_len=128):
    ids = [stoi.get(tok, unk) for tok in simple_tokenize(text)][:max_len]
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# ------------------------------------------------------------
# Dataset builders
# ------------------------------------------------------------
def build_mnist(batch=128, quick=False):
    tr = transforms.Compose([transforms.ToTensor()])
    ds_train = datasets.MNIST("../data", train=True, download=True, transform=tr)
    ds_test = datasets.MNIST("../data", train=False, download=True, transform=tr)
    if quick:
        ds_train, _ = random_split(ds_train, [2000, len(ds_train)-2000])
        ds_test, _ = random_split(ds_test, [500, len(ds_test)-500])
    return DataLoader(ds_train,batch_size=batch,shuffle=True), DataLoader(ds_test,batch_size=batch)

def build_fashionmnist(batch=128, quick=False):
    tr = transforms.Compose([transforms.ToTensor()])
    ds_train = datasets.FashionMNIST("../data", train=True, download=True, transform=tr)
    ds_test = datasets.FashionMNIST("../data", train=False, download=True, transform=tr)
    if quick:
        ds_train, _ = random_split(ds_train, [2000, len(ds_train)-2000])
        ds_test, _ = random_split(ds_test, [500, len(ds_test)-500])
    return DataLoader(ds_train,batch_size=batch,shuffle=True), DataLoader(ds_test,batch_size=batch)

def build_cifar10(batch=128, quick=False):
    tr = transforms.Compose([transforms.ToTensor()])
    ds_train = datasets.CIFAR10("../data", train=True, download=True, transform=tr)
    ds_test = datasets.CIFAR10("../data", train=False, download=True, transform=tr)
    if quick:
        ds_train, _ = random_split(ds_train, [3000, len(ds_train)-3000])
        ds_test, _ = random_split(ds_test, [500, len(ds_test)-500])
    return DataLoader(ds_train,batch_size=batch,shuffle=True), DataLoader(ds_test,batch_size=batch)

def build_agnews(batch=64, quick=False, max_len=128):
    ds = load_dataset("ag_news")
    train_raw = ds["train"]
    test_raw = ds["test"]
    if quick:
        train_raw = train_raw.select(range(2000))
        test_raw = test_raw.select(range(500))
    texts = [train_raw[i]["text"] for i in range(len(train_raw))]
    stoi, unk, pad = build_vocab(texts)
    def proc(split):
        data = split
        X = torch.stack([encode_text(r["text"], stoi, unk, pad, max_len) for r in data])
        y = torch.tensor([int(r["label"]) for r in data],dtype=torch.long)
        return TensorDataset(X,y)
    return DataLoader(proc(train_raw),batch_size=batch,shuffle=True), DataLoader(proc(test_raw),batch_size=batch), len(stoi), 4, pad

def build_yelp(batch=64, quick=False, max_len=128):
    ds = load_dataset("yelp_polarity")
    train_raw = ds["train"]
    test_raw = ds["test"]
    if quick:
        train_raw = train_raw.select(range(2000))
        test_raw = test_raw.select(range(500))
    texts = [train_raw[i]["text"] for i in range(len(train_raw))]
    stoi, unk, pad = build_vocab(texts)
    def proc(split):
        X = torch.stack([encode_text(r["text"], stoi, unk, pad, max_len) for r in split])
        y = torch.tensor([int(r["label"]) for r in split],dtype=torch.long)
        return TensorDataset(X,y)
    return DataLoader(proc(train_raw),batch_size=batch,shuffle=True), DataLoader(proc(test_raw),batch_size=batch), len(stoi), 2, pad

import math
from typing import Dict, Callable, Optional
# ---- Collect target layers ---------------------------------------------------
def _collect_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d))]

# ---- Baselines for b_L -------------------------------------------------------
def _baseline_from(h: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(h)
    elif mode == "batch_mean":
        b = h.mean(dim=0, keepdim=True)
        return b.expand_as(h)
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")

# ---- p-norm per-sample on model outputs -------------------------------------
def _pnorm_per_sample(y: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    flat = y.flatten(start_dim=1) if y.ndim > 1 else y.unsqueeze(0)
    if p == 1:
        return flat.abs().sum(dim=1)
    if p == 2:
        return flat.pow(2).sum(dim=1).sqrt()
    if math.isinf(p) and p > 0:
        return flat.abs().max(dim=1).values
    return flat.abs().pow(p).sum(dim=1).pow(1.0/p)

# ---- Robust correlations (no warnings on constants) --------------------------
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np

def correlate_dicts(pred: Dict[str, float], truth: Dict[str, float]):
    keys = [k for k in pred.keys() if k in truth]
    pv = np.array([pred[k] for k in keys], dtype=float)
    tv = np.array([truth[k] for k in keys], dtype=float)
    if pv.size < 2:
        return {"pearson": float("nan"), "spearman": float("nan"), "kendall": float("nan")}
    def _safe(fn, a, b):
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return float("nan")
        try:
            r = fn(a, b)[0]
            return float(r) if np.isfinite(r) else float("nan")
        except Exception:
            return float("nan")
    return {
        "pearson": _safe(pearsonr, pv, tv),
        "spearman": _safe(spearmanr, pv, tv),
        "kendall": _safe(kendalltau, pv, tv),
    }

# ---- True causal ground truth via actual ablations ---------------------------
@torch.no_grad()
def measure_true_effects(model: nn.Module, data_loader, sample_batches: int = 2) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    DEVICE = next(model.parameters()).device

    # Small, fixed evaluation batch (or two) for stability
    xs_list, ys_list = [], []
    it = iter(data_loader)
    for _ in range(sample_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        xs_list.append(x.to(DEVICE))
        ys_list.append(y.to(DEVICE))
    xs = torch.cat(xs_list, dim=0)
    ys = torch.cat(ys_list, dim=0)

    base = F.cross_entropy(model(xs), ys).item()

    effects = {}
    layers = _collect_layers(model)
    for i, layer in enumerate(layers):
        def _override(m, inp, out):
            return torch.zeros_like(out)
        h = layer.register_forward_hook(_override)
        try:
            loss = F.cross_entropy(model(xs), ys).item()
        finally:
            h.remove()
        effects[f"layer_{i}"] = max(0.0, loss - base)
    if was_training:
        model.train()
    return effects

# ---- Correct first-order G-NDI using JVP (your causal proxy) -----------------
def gndi_score(
    model: nn.Module,
    data_loader,
    *,
    baseline: str = "zero",
    p: float = 2.0,
    max_batches: int = 4,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    DEVICE = next(model.parameters()).device

    layers = _collect_layers(model)
    sums = torch.zeros(len(layers), dtype=torch.float64)
    counts = torch.zeros(len(layers), dtype=torch.float64)

    def _capture_acts(xs):
        acts = {}
        def make_cap(idx):
            def _cap(m, inp, out): acts[idx] = out.detach()
            return _cap
        hs = [layers[i].register_forward_hook(make_cap(i)) for i in range(len(layers))]
        _ = model(xs)
        for h in hs: h.remove()
        return acts

    def _make_f_of_z(layer_idx, xs):
        L = layers[layer_idx]
        def f_of_z(z):
            def _override(m, inp, out): return z
            h = L.register_forward_hook(_override)
            try:
                y = model(xs)
            finally:
                h.remove()
            return y
        return f_of_z

    it = iter(data_loader)
    for _ in range(max_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        xs = x.to(DEVICE)

        acts = _capture_acts(xs)
        for li, hL in acts.items():
            bL = _baseline_from(hL, baseline)
            v = (bL - hL).detach()
            z0 = hL.detach().requires_grad_(True)
            fL = _make_f_of_z(li, xs)
            _, jvp = torch.autograd.functional.jvp(fL, (z0,), (v,), create_graph=False, strict=True)
            per_sample = _pnorm_per_sample(jvp, p=p)
            sums[li] += per_sample.mean().double()
            counts[li] += 1

    out = {f"layer_{i}": float((sums[i] / counts[i]).item()) if counts[i] > 0 else float("nan")
           for i in range(len(layers))}
    if was_training:
        model.train()
    return out

# ---- Baselines ---------------------------------------------------------------
def weight_magnitude_scores(model: nn.Module) -> Dict[str, float]:
    s, idx = {}, 0
    for m in _collect_layers(model):
        s[f"layer_{idx}"] = float(m.weight.detach().norm().item())
        idx += 1
    return s


def synflow_scores(model: nn.Module, input_shape) -> Dict[str, float]:
    """
    Minimal SynFlow-style pass:
      - Make params positive (abs)
      - Forward 1s tensor, backprop sum(outputs)
      - Score per layer: sum(|w * grad_w|)  (or just sum(|grad_w|) if needed)
    """
    was_training = model.training
    model.eval()

    # Build ones input roughly matching the model’s expected shape.
    # For CV we generally have NCHW; for text, use an integer token tensor of ones.
    # Here we try a generic float-ones tensor and rely on first layer shape.
    # If you have dataset-specific shapes, pass them in more explicitly.
    x = torch.ones(*input_shape).to(DEVICE)

    # Save original params and use absolute-valued copy
    params = [p for p in model.parameters() if p.requires_grad]
    originals = [p.data.clone() for p in params]
    for p in params:
        p.data = p.data.abs()

    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    # Single forward/backward
    out = model(x)
    if out.ndim == 2:
        s = out.sum()
    else:
        s = out.view(-1).sum()
    s.backward()

    scores = {}
    idx = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            w = m.weight
            g = m.weight.grad if m.weight.grad is not None else torch.zeros_like(w)
            scores[f"layer_{idx}"] = float((w.abs() * g.abs()).sum().item())
            idx += 1

    # Restore originals
    for p, o in zip(params, originals):
        p.data = o

    if was_training:
        model.train()
    return scores


# ------------------------------------------------------------
# Metric evaluation
# ------------------------------------------------------------
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np

def correlate_dicts(pred: Dict[str, float], truth: Dict[str, float]):
    keys = [k for k in pred.keys() if k in truth]
    pv = np.array([pred[k] for k in keys], dtype=float)
    tv = np.array([truth[k] for k in keys], dtype=float)

    if pv.size < 2:
        return {"pearson": float("nan"), "spearman": float("nan"), "kendall": float("nan")}

    # Handle constant vectors cleanly
    def _safe_corr(fn, a, b):
        try:
            if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                return float("nan")
            r = fn(a, b)[0]
            return float(r) if np.isfinite(r) else float("nan")
        except Exception:
            return float("nan")

    return {
        "pearson": _safe_corr(pearsonr, pv, tv),
        "spearman": _safe_corr(spearmanr, pv, tv),
        "kendall": _safe_corr(kendalltau, pv, tv),
    }

# ------------------------------------------------------------
# Runner for all experiments
# ------------------------------------------------------------
def run_all(args):
    start_total = time.time()
    results = []

    cfgs = [
        ("MNIST",          "MLP",       build_mnist,        lambda _: MLP().to(DEVICE),             ("cv", (args.batch, 1, 28, 28))),
        ("FashionMNIST",   "TinyCNN",   build_fashionmnist, lambda _: TinyCNN(in_channels=1, num_classes=10).to(DEVICE), ("cv", (args.batch, 1, 28, 28))),
        ("CIFAR10",        "ResNet18",  build_cifar10,      lambda _: torchvision.models.resnet18(num_classes=10).to(DEVICE), ("cv", (args.batch, 3, 32, 32))),
        ("AGNews",         "TextCNN",   build_agnews,       lambda v: TextCNN(vocab_size=v[2][0], embed_dim=128, num_classes=v[3], pad_idx=v[4]).to(DEVICE), ("nlp", (args.batch, 128))),
        ("YelpPolarity",   "GRUClass",  build_yelp,         lambda v: GRUClassifier(vocab_size=v[2][0], embed_dim=128, hidden_dim=128, num_classes=v[3], pad_idx=v[4]).to(DEVICE), ("nlp", (args.batch, 128))),
    ]

    variants = ["raw", "log", "l2norm", "meannorm"]

    for (dsname, modelname, loader_fn, model_fn, (domain, input_shape)) in cfgs:
        print(f"\n=== Running {dsname} / {modelname} ===")
        if domain == "nlp":
            tr, te, vocab, nc, pad = loader_fn(args.batch, args.quick)
            model = model_fn((None, None, vocab, nc, pad))
        else:
            tr, te = loader_fn(args.batch, args.quick)
            model = model_fn(None)

        # Ground-truth causal effects (per-layer ablation deltas)
        truth = measure_true_effects(model, tr)

        # Predictions: G-NDI variants
        for var in variants:
            pred = gndi_predict_scores(model, tr, var)
            corr = correlate_dicts(pred, truth)
            results.append({
                "dataset": dsname,
                "model": modelname,
                "method": f"GNDI-{var}",
                **corr,
                "layers": len(pred),
                "time": round(time.time() - start_total, 2),
            })

        # Baselines
        wm = weight_magnitude_scores(model)
        corr_wm = correlate_dicts(wm, truth)
        results.append({
            "dataset": dsname,
            "model": modelname,
            "method": "WeightMagnitude",
            **corr_wm,
            "layers": len(wm),
            "time": round(time.time() - start_total, 2),
        })

        # SynFlow (lightweight)
        try:
            syn = synflow_scores(model, input_shape=input_shape)
            corr_syn = correlate_dicts(syn, truth)
            results.append({
                "dataset": dsname,
                "model": modelname,
                "method": "SynFlow",
                **corr_syn,
                "layers": len(syn),
                "time": round(time.time() - start_total, 2),
            })
        except Exception as e:
            print(f"[SynFlow skipped for {dsname}/{modelname}]: {e}")

    json.dump(results, open("all_results.json", "w"), indent=2)
    print("\n[✓] Finished all tasks. JSON saved.")
    return results

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--batch",type=int,default=64)
    args = parser.parse_args()

    from reporter import generate_pdf
    start_time = time.time()
    results = run_all(args)
    generate_pdf(results, "overnight_report.pdf")
    elapsed = round((time.time() - start_time) / 60, 2)
    print(f"\nTotal runtime: {elapsed} min")
    print("[✓] All done — results in all_results.json and overnight_report.pdf")


if __name__=="__main__":
    main()
