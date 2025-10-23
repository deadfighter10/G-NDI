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

def train_and_eval(cfg) -> Tuple[torch.nn.Module, Dict[str, Any]]:
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
        for x, y in tqdm(loaders["train"], desc="Training", ncols=80):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
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
    return model, loaders

def evaluate(model, loader, amp=True):
    return eval_classification(model, loader, amp=amp)
