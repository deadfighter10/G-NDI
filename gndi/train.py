# gndi/train.py
# -*- coding: utf-8 -*-
"""
Unified training/eval loops for CV+NLP classification.
Exposes:
- train_and_eval(cfg) -> (model, loaders)
- evaluate(model, loader, amp=True) -> dict(acc@1=...)
"""
from __future__ import annotations
from typing import Tuple, Dict, Any
import argparse, os, yaml, json, time

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data import build_dataloaders
from .models_cv import build_cv_model
from .models_nlp import build_nlp_model
from .eval.metrics import eval_classification

def _is_nlp(cfg) -> bool:
    name = str(cfg.get("dataset", {}).get("name", "")).lower()
    return name in ["ag_news", "agnews", "sst2", "sst-2", "yelp_polarity", "yelp", "mnli"]

def _make_optimizer(params, cfg):
    opt = cfg.get("train", {}).get("optimizer", "sgd").lower()
    lr = float(cfg.get("train", {}).get("lr", 0.1))
    wd = float(cfg.get("train", {}).get("weight_decay", 0.0))
    if opt == "sgd":
        return optim.SGD(params, lr=lr, momentum=cfg.get("train", {}).get("momentum", 0.9), nesterov=cfg.get("train", {}).get("nesterov", True), weight_decay=wd)
    return optim.AdamW(params, lr=lr, weight_decay=wd)

def _make_scheduler(optimizer, cfg):
    sched = cfg.get("train", {}).get("lr_schedule", "cosine")
    epochs = int(cfg.get("train", {}).get("epochs", 10))
    if sched == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return None

def _build_model(cfg):
    return build_nlp_model(cfg) if _is_nlp(cfg) else build_cv_model(cfg)

def train_and_eval(cfg) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, float]]:
    amp = bool(cfg.get("amp", True))
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    loaders = build_dataloaders(cfg)
    model = _build_model(cfg).to(device)

    epochs = int(cfg.get("train", {}).get("epochs", 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model.parameters(), cfg)
    scheduler = _make_scheduler(optimizer, cfg)

    model.train()
    for ep in range(epochs):
        print(f"\n[Epoch {ep + 1}/{epochs}]")
        for batch in tqdm(loaders["train"], desc="Training", ncols=80):
            # Handle NLP dictionary-based batches
            if isinstance(batch, dict):
                y = batch.pop('label').to(device, non_blocking=True)
                x = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Handle CV tuple/list-based batches
            elif isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    x, y = batch
                elif len(batch) == 3:
                    x, y, _ = batch  # ignore metadata/index
                else:
                    x, y = batch[0], batch[1]
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                # NLP models need dict unpacking, CV models expect tensors
                if isinstance(x, dict):
                    outputs = model(**x)
                    # Extract logits from HuggingFace model output
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                else:
                    logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            if cfg.get("train", {}).get("grad_clip", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

    # Final eval
    metrics = eval_classification(model, loaders["val"], amp=amp)
    out_dir = cfg.get("paths", {}).get("out_dir", "./runs/tmp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "dense_final.pt"))
    return model, loaders, metrics

def evaluate(model, loader, amp=True):
    return eval_classification(model, loader, amp=amp)

# --- add near the imports if missing ---
import argparse, os, yaml, json, time, inspect

def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _apply_overrides(cfg, args):
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = float(args.lr)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
        cfg["train"].setdefault("eval_batch_size", int(args.batch_size))
    if args.subset is not None:
        cfg.setdefault("dataset", {})["subset_fraction"] = float(args.subset)
    if args.mps_fast:
        ds = cfg.setdefault("dataset", {})
        ds["num_workers"] = 0
        ds["pin_memory"] = False
    return cfg

def _pretty_kv(d):
    return " ".join(f"{k}={v}" for k, v in d.items())

def cli_main():
    ap = argparse.ArgumentParser(description="Train a dense model (standalone).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--subset", type=float, default=None, help="Fraction of dataset to use (e.g., 1.0, 0.5, 0.1)")
    ap.add_argument("--mps-fast", action="store_true", help="Mac-friendly DataLoader defaults")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    cfg = _apply_overrides(cfg, args)

    # Echo settings
    tcfg = cfg.get("train", {})
    dcfg = cfg.get("dataset", {})
    print(f"[train] {_pretty_kv({'epochs': tcfg.get('epochs'), 'lr': tcfg.get('lr'), 'batch': tcfg.get('batch_size')})}")
    print(f"[data ] {_pretty_kv({'subset_fraction': dcfg.get('subset_fraction', 1.0), 'num_workers': dcfg.get('num_workers', 0), 'pin_memory': dcfg.get('pin_memory', False)})}")

    # Import your core function
    from .train import train_and_eval as _core_train_and_eval  # this is YOUR existing function

    # Build model/loaders only if the core expects them
    needs_model_loaders = False
    sig = inspect.signature(_core_train_and_eval)
    npos = sum(1 for p in sig.parameters.values()
               if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty)
    # Call strategy:
    # - if it takes 1 required arg -> assume (cfg)
    # - if it takes 3 required args -> assume (model, loaders, cfg)
    # - otherwise, fall back to kwargs by name
    call_mode = "cfg_only"
    if npos >= 3:
        call_mode = "model_loaders_cfg"
        needs_model_loaders = True

    if needs_model_loaders:
        from .data import build_dataloaders
        from .models_cv import build_cv_model
        from .models_nlp import build_nlp_model

        loaders = build_dataloaders(cfg)
        name = str(cfg.get("dataset", {}).get("name", "")).lower()
        if name in ["cifar10", "cifar100", "imagenet_subset", "imagenet-subset", "imagenet_subset_200"]:
            model = build_cv_model(cfg)
        else:
            model = build_nlp_model(cfg)

    # Run training
    start = time.time()
    if call_mode == "cfg_only":
        model, loader, results = _core_train_and_eval(cfg)
    elif call_mode == "model_loaders_cfg":
        model, loader, results = _core_train_and_eval(model, loaders, cfg)
    else:
        # be generous: try kwargs
        try:
            model, loader, results = _core_train_and_eval(cfg=cfg)
        except TypeError:
            model, loader, results = _core_train_and_eval(model=model, loaders=loaders, cfg=cfg)
    dur = time.time() - start

    # Normalize results to a dict
    if isinstance(results, tuple):
        # allow (metrics_dict, ...) shape
        metrics = results[0] if isinstance(results[0], dict) else {}
    elif isinstance(results, dict):
        metrics = results
    else:
        metrics = {}

    acc = metrics.get("acc@1", None)
    if acc is None:
        # optionally compute quick eval here if your core didn't
        print("[warn ] 'acc@1' not found in results. Ensure your core returns a metrics dict.")
    else:
        print(f"[done ] acc@1={acc:.3f}  time={dur/60:.1f} min")

    out_dir = cfg.get("paths", {}).get("out_dir", "./runs/tmp_train")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dense_final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    cli_main()
