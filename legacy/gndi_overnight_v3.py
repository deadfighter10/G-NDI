"""
G-NDI Overnight v5 — Real-data, trained models (≈5 epochs), pruning damage comparison

What this script does (end-to-end):
  • Loads real CV datasets (MNIST, FashionMNIST, CIFAR-10) via torchvision
  • Loads real NLP datasets (AG News, Yelp Polarity) with a true BERT encoder (Hugging Face)
  • Trains each model for --epochs (default 5)
  • Computes layer-importance by:
      - GNDI (first-order causal intervention via JVP)
      - Weight Magnitude (baseline)
      - SynFlow (baseline; CV only)
  • Prunes the bottom-K% layers per method (no retraining) by zeroing layer outputs via hooks
  • Measures the accuracy drop ("damage") after pruning for each method & prune ratio
  • Saves a machine-readable JSON plus a friendly console table summary

Notes & Requirements:
  • This script uses: torch, torchvision, transformers, datasets, tqdm, numpy, scipy.
  • Internet is typically required on first run to download models/datasets.
  • Pruning here is *structural-by-effect*: we mask layer outputs at inference time to simulate removal
    (fast, interpretable, retraining-free). For residual architectures (e.g., BERT, ResNet), this approximates
    removing a block's contribution.

Run examples:
  python gndi_overnight_v5.py                       # full pipeline, 5 epochs each
  python gndi_overnight_v5.py --epochs 3            # quicker pass
  python gndi_overnight_v5.py --prune 0.1 0.3 0.5   # test multiple prune ratios
  python gndi_overnight_v5.py --no-cv               # NLP only (AG News, Yelp)
  python gndi_overnight_v5.py --no-nlp              # CV only (MNIST, FMNIST, CIFAR10)

Output files:
  pruning_results.json  — all metrics and per-scenario results

"""
from __future__ import annotations
import argparse, math, json, time, random, os, traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from scipy.stats import pearsonr, spearmanr, kendalltau

# CV
import torchvision
from torchvision import transforms

# NLP
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337

# -----------------------------------------------------------------------------
# Repro
# -----------------------------------------------------------------------------

def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Data: CV
# -----------------------------------------------------------------------------

def build_mnist(batch=128):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return DataLoader(train, batch_size=batch, shuffle=True, num_workers=2), DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

def build_fashionmnist(batch=128):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
    test  = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)
    return DataLoader(train, batch_size=batch, shuffle=True, num_workers=2), DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

def build_cifar10(batch=128):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_test = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)
    return DataLoader(train, batch_size=batch, shuffle=True, num_workers=2), DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

# -----------------------------------------------------------------------------
# Models: CV
# -----------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """A simple but slightly larger CNN for MNIST/FashionMNIST."""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.c2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(128*10*10, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool(x)
        x = F.relu(self.c3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_resnet18(num_classes=10):
    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    return m

# -----------------------------------------------------------------------------
# Data + Models: NLP (AG News, Yelp Polarity) with BERT
# -----------------------------------------------------------------------------

@dataclass
class NLPBundle:
    tokenizer: Any
    model: nn.Module
    collator: Any
    train_loader: DataLoader
    test_loader: DataLoader


def build_nlp(task: str, model_name: str = "bert-base-uncased", batch: int = 16) -> NLPBundle:
    assert task in {"ag_news", "yelp_polarity"}
    ds = load_dataset(task)
    num_labels = 4 if task == "ag_news" else 2
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    train_loader = DataLoader(ds["train"], batch_size=batch, shuffle=True, collate_fn=collator, num_workers=2)
    test_loader  = DataLoader(ds["test"],  batch_size=batch, shuffle=False, collate_fn=collator, num_workers=2)

    return NLPBundle(tokenizer, model, collator, train_loader, test_loader)

# -----------------------------------------------------------------------------
# Training & Eval
# -----------------------------------------------------------------------------

def train_classifier(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, *, epochs: int = 5, lr: float = 3e-4, is_nlp: bool=False) -> Dict[str, Any]:
    model.to(DEVICE)
    if is_nlp:
        # standard BERT fine-tuning setup
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        num_training_steps = epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(0, num_training_steps//10), num_training_steps=num_training_steps)
        loss_fn = None  # handled by model
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        total = 0.0; n = 0
        pbar = tqdm(train_loader, desc=f"train ep{ep}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            if is_nlp:
                batch = {k: v.to(DEVICE) for k,v in batch.items()}
                out = model(**batch)
                loss = out.loss
            else:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * (batch["input_ids"].size(0) if is_nlp else x.size(0))
            n += (batch["input_ids"].size(0) if is_nlp else x.size(0))
            pbar.set_postfix({"loss": f"{total/n:.4f}"})
        acc = evaluate_accuracy(model, test_loader, is_nlp=is_nlp)
        print(f"  eval acc: {acc:.4f}")
    return {"acc": evaluate_accuracy(model, test_loader, is_nlp=is_nlp)}


def evaluate_accuracy(model: nn.Module, data_loader: DataLoader, *, is_nlp: bool=False) -> float:
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for batch in data_loader:
            if is_nlp:
                batch = {k: v.to(DEVICE) for k,v in batch.items()}
                out = model(**batch)
                logits = out.logits
                y = batch["labels"]
            else:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(1,total)

# -----------------------------------------------------------------------------
# Layer utilities (collect, override, pruning masks)
# -----------------------------------------------------------------------------

def collect_layers(model: nn.Module, is_nlp: bool) -> List[nn.Module]:
    layers: List[nn.Module] = []
    if is_nlp:
        # target BERT encoder layers (transformer blocks) + classifier
        for m in model.modules():
            if m.__class__.__name__ == "BertLayer":
                layers.append(m)
        # also include final classifier Linear
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features == model.config.num_labels:
                layers.append(m)
    else:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                layers.append(m)
    return layers

# Hook-based pruning: zero the output of selected layers
class LayerZeroMask:
    def __init__(self, model: nn.Module, target_layers: List[int], layers: List[nn.Module]):
        self.handles = []
        tlset = set(target_layers)
        for idx, layer in enumerate(layers):
            if idx in tlset:
                self.handles.append(layer.register_forward_hook(lambda m, inp, out: torch.zeros_like(out)))
    def remove(self):
        for h in self.handles:
            h.remove()

# -----------------------------------------------------------------------------
# GNDI (JVP) and baselines
# -----------------------------------------------------------------------------

def _pnorm_per_sample(y: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    flat = y.flatten(start_dim=1) if y.ndim > 1 else y.reshape(1, -1)
    if p == 1:
        return flat.abs().sum(dim=1)
    if p == 2:
        return flat.pow(2).sum(dim=1).sqrt()
    if math.isinf(p) and p > 0:
        return flat.abs().max(dim=1).values
    return flat.abs().pow(p).sum(dim=1).pow(1.0/p)


def _baseline_from(h: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(h)
    elif mode == "batch_mean":
        b = h.mean(dim=0, keepdim=True)
        return b.expand_as(h)
    else:
        raise ValueError("Unknown baseline")


def gndi_scores(model: nn.Module, data_loader: DataLoader, *, is_nlp: bool, baseline: str = "zero", p: float=2.0, max_batches: int = 4) -> Dict[str, float]:
    model.eval()
    layers = collect_layers(model, is_nlp)
    sums = torch.zeros(len(layers), dtype=torch.float64)
    counts = torch.zeros(len(layers), dtype=torch.float64)

    def capture_acts(xs, attn=None, token_type_ids=None):
        acts: Dict[int, torch.Tensor] = {}
        def make_cap(i):
            def _cap(m, inp, out):
                acts[i] = out.detach()
            return _cap
        hs = [layers[i].register_forward_hook(make_cap(i)) for i in range(len(layers))]
        if is_nlp:
            out = model(input_ids=xs, attention_mask=attn, token_type_ids=token_type_ids)
        else:
            out = model(xs)
        for h in hs: h.remove()
        return acts

    def make_f_of_z(li, xs, attn=None, token_type_ids=None):
        L = layers[li]
        def f_of_z(z):
            def _override(m, inp, out): return z
            h = L.register_forward_hook(_override)
            try:
                if is_nlp:
                    y = model(input_ids=xs, attention_mask=attn, token_type_ids=token_type_ids).logits
                else:
                    y = model(xs)
            finally:
                h.remove()
            return y
        return f_of_z

    it = iter(data_loader)
    for _ in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        if is_nlp:
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            xs = batch["input_ids"]; attn = batch.get("attention_mask"); tok = batch.get("token_type_ids")
        else:
            xs, _ = batch
            xs = xs.to(DEVICE)
            attn = tok = None
        acts = capture_acts(xs, attn, tok)
        for li, hL in acts.items():
            bL = _baseline_from(hL, baseline)
            v = (bL - hL).detach()
            z0 = hL.detach().requires_grad_(True)
            fL = make_f_of_z(li, xs, attn, tok)
            _, jvp = torch.autograd.functional.jvp(fL, (z0,), (v,), create_graph=False, strict=True)
            if is_nlp:
                per = _pnorm_per_sample(jvp, p=p)
            else:
                per = _pnorm_per_sample(jvp, p=p)
            sums[li] += per.mean().double()
            counts[li] += 1

    return {f"layer_{i}": float((sums[i]/counts[i]).item()) if counts[i] > 0 else float("nan") for i in range(len(layers))}


def weight_magnitude_scores(model: nn.Module, is_nlp: bool) -> Dict[str, float]:
    layers = collect_layers(model, is_nlp)
    s = {}
    for i, m in enumerate(layers):
        if hasattr(m, 'weight') and isinstance(m.weight, torch.Tensor):
            s[f"layer_{i}"] = float(m.weight.detach().abs().sum().item())
        else:
            # For BertLayer (no single weight tensor), aggregate its submodule weights
            tot = 0.0
            for p in m.parameters(): tot += p.detach().abs().sum().item()
            s[f"layer_{i}"] = float(tot)
    return s


def synflow_scores(model: nn.Module, input_shape: Tuple[int,...]) -> Dict[str, float]:
    # CV only
    was_training = model.training
    model.eval()
    x = torch.ones(*input_shape, device=DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    originals = [p.data.clone() for p in params]
    for p in params:
        p.data = p.data.abs()
        if p.grad is not None: p.grad.zero_()
    try:
        out = model(x)
        s = out.sum() if out.ndim > 1 else out.view(-1).sum()
        s.backward()
        layers = collect_layers(model, is_nlp=False)
        scores = {}
        for i, m in enumerate(layers):
            if hasattr(m, 'weight') and m.weight.grad is not None:
                scores[f"layer_{i}"] = float((m.weight.data.abs()*m.weight.grad.abs()).sum().item())
            else:
                tot = 0.0
                for p in m.parameters():
                    if p.grad is not None:
                        tot += (p.data.abs()*p.grad.abs()).sum().item()
                scores[f"layer_{i}"] = float(tot)
    finally:
        for p,o in zip(params, originals): p.data = o
        if was_training: model.train()
    return scores

# -----------------------------------------------------------------------------
# Experiment: score, prune, evaluate damage
# -----------------------------------------------------------------------------

def rank_layers(score_dict: Dict[str, float]) -> List[int]:
    items = [(int(k.split("_")[1]), v) for k,v in score_dict.items() if math.isfinite(v)]
    items.sort(key=lambda kv: kv[1])  # ascending: small = less important
    return [idx for idx,_ in items]


def prune_and_eval(model: nn.Module, layers: List[nn.Module], layer_order: List[int], prune_ratio: float, test_loader: DataLoader, *, is_nlp: bool) -> float:
    k = max(1, int(len(layers) * prune_ratio))
    to_prune = layer_order[:k]
    masker = LayerZeroMask(model, to_prune, layers)
    try:
        acc = evaluate_accuracy(model, test_loader, is_nlp=is_nlp)
    finally:
        masker.remove()
    return acc

# -----------------------------------------------------------------------------
# Driver per dataset
# -----------------------------------------------------------------------------

def run_cv_dataset(name: str, train_loader: DataLoader, test_loader: DataLoader, model_builder: Callable[[], nn.Module], *, epochs: int, prune_list: List[float]) -> Dict[str, Any]:
    print(f"\n=== CV: {name} ===")
    model = model_builder().to(DEVICE)
    train_classifier(model, train_loader, test_loader, epochs=epochs, is_nlp=False)
    base_acc = evaluate_accuracy(model, test_loader, is_nlp=False)
    print(f"Base accuracy: {base_acc:.4f}")

    layers = collect_layers(model, is_nlp=False)

    scores_gndi = gndi_scores(model, train_loader, is_nlp=False, baseline="zero", p=2.0, max_batches=4)
    scores_wmag = weight_magnitude_scores(model, is_nlp=False)
    # input_shape for synflow
    sample = next(iter(test_loader))[0].to(DEVICE)
    input_shape = (sample.shape[0],) + tuple(sample.shape[1:])
    scores_synf = synflow_scores(model, input_shape)

    methods = {
        "GNDI": scores_gndi,
        "WeightMag": scores_wmag,
        "SynFlow": scores_synf,
    }

    results = {"dataset": name, "base_acc": base_acc, "prune": {}}

    for mname, sdict in methods.items():
        order = rank_layers(sdict)
        results["prune"][mname] = {}
        for r in prune_list:
            acc_after = prune_and_eval(model, layers, order, r, test_loader, is_nlp=False)
            damage = base_acc - acc_after
            results["prune"][mname][str(r)] = {"acc": acc_after, "damage": damage}
            print(f"  {mname:10s} prune {r:>4.0%} -> acc {acc_after:.4f} (damage {damage:.4f})")

    return results


def run_nlp_dataset(task: str, bundle: NLPBundle, *, epochs: int, prune_list: List[float]) -> Dict[str, Any]:
    print(f"\n=== NLP: {task} (BERT) ===")
    model = bundle.model.to(DEVICE)
    train_classifier(model, bundle.train_loader, bundle.test_loader, epochs=epochs, is_nlp=True)
    base_acc = evaluate_accuracy(model, bundle.test_loader, is_nlp=True)
    print(f"Base accuracy: {base_acc:.4f}")

    layers = collect_layers(model, is_nlp=True)

    scores_gndi = gndi_scores(model, bundle.train_loader, is_nlp=True, baseline="zero", p=2.0, max_batches=4)
    scores_wmag = weight_magnitude_scores(model, is_nlp=True)
    methods = {
        "GNDI": scores_gndi,
        "WeightMag": scores_wmag,
        # SynFlow not applied to NLP
    }

    results = {"dataset": task, "base_acc": base_acc, "prune": {}}

    for mname, sdict in methods.items():
        order = rank_layers(sdict)
        results["prune"][mname] = {}
        for r in prune_list:
            acc_after = prune_and_eval(model, layers, order, r, bundle.test_loader, is_nlp=True)
            damage = base_acc - acc_after
            results["prune"][mname][str(r)] = {"acc": acc_after, "damage": damage}
            print(f"  {mname:10s} prune {r:>4.0%} -> acc {acc_after:.4f} (damage {damage:.4f})")

    return results

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="G-NDI v5: real-data training + pruning damage comparison")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--nlp-batch", type=int, default=16)
    ap.add_argument("--prune", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--no-nlp", action="store_true")
    ap.add_argument("--bert", type=str, default="bert-base-uncased")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    all_results: Dict[str, Any] = {"cv": [], "nlp": []}

    if not args.no_cv:
        # MNIST
        tr, te = build_mnist(args.batch)
        all_results["cv"].append(run_cv_dataset("MNIST", tr, te, lambda: SmallCNN(1, 10), epochs=args.epochs, prune_list=args.prune))
        # FashionMNIST
        tr, te = build_fashionmnist(args.batch)
        all_results["cv"].append(run_cv_dataset("FashionMNIST", tr, te, lambda: SmallCNN(1, 10), epochs=args.epochs, prune_list=args.prune))
        # CIFAR10 (bigger model: ResNet18)
        tr, te = build_cifar10(args.batch)
        all_results["cv"].append(run_cv_dataset("CIFAR10", tr, te, lambda: build_resnet18(10), epochs=args.epochs, prune_list=args.prune))

    if not args.no_nlp:
        # AG News (4-class)
        ag = build_nlp("ag_news", model_name=args.bert, batch=args.nlp_batch)
        all_results["nlp"].append(run_nlp_dataset("AGNews", ag, epochs=args.epochs, prune_list=args.prune))
        # Yelp Polarity (2-class)
        yp = build_nlp("yelp_polarity", model_name=args.bert, batch=args.nlp_batch)
        all_results["nlp"].append(run_nlp_dataset("YelpPolarity", yp, epochs=args.epochs, prune_list=args.prune))

    with open("pruning_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[✓] Saved pruning_results.json")

    # Quick friendly summary
    def best_method(entry):
        # choose method with least damage at highest prune ratio
        rmax = max(map(float, entry["prune"][next(iter(entry["prune"]))].keys()))
        best = None; best_dmg = 1e9
        for m, tbl in entry["prune"].items():
            dmg = tbl[str(rmax)]["damage"]
            if dmg < best_dmg:
                best_dmg = dmg; best = m
        return best, rmax, best_dmg

    print("\n=== SUMMARY (least damage at highest prune ratio) ===")
    for entry in all_results["cv"]:
        b, r, d = best_method(entry)
        print(f"CV/{entry['dataset']}: base_acc={entry['base_acc']:.4f} | best={b} at {r:.0%} prune (damage={d:.4f})")
    for entry in all_results["nlp"]:
        b, r, d = best_method(entry)
        print(f"NLP/{entry['dataset']}: base_acc={entry['base_acc']:.4f} | best={b} at {r:.0%} prune (damage={d:.4f})")

if __name__ == "__main__":
    main()
