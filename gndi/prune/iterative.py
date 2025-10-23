# gndi/prune/iterative.py
# -*- coding: utf-8 -*-
"""
Iterative pruning schedules: prune -> (optional) finetune -> repeat.

Exposes:
- run_iterative_pruning(
      model, train_loader, val_loader, units, score_fns, selector_cfg, finetune_cfg,
      step_budgets, runtime_masking=True, structural=False, eval_fn=None, device=None
  )

Inputs:
- score_fns: dict name->callable(model, loader, units, **kwargs)->{unit_id: score}
  (e.g., {"gndi": compute_gndi_units, "magnitude": compute_magnitude_units, ...})
- selector_cfg: {
      "method": "gndi" (scoring method name),
      "tie_break": "random",
      "budget_type": "params"|"flops",
      "per_layer_cap": 0.5 (optional),
      "score_kwargs": {...}
  }
- finetune_cfg: {
      "enabled": True, "epochs": 3, "optimizer": "sgd"/"adamw", "lr":..., "weight_decay":...,
      "schedule": "cosine"/"none"
  }
- step_budgets: list of fractions or absolute amounts to prune per step (e.g., [0.1, 0.1, 0.1])

- eval_fn: optional callable(model, val_loader)->metrics dict to log after each step

Returns:
- history: list of dicts with {"step": i, "pruned_ids": [...], "metrics": {...}, "cum_pruned": int}
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from .selectors import rank_units, select_by_budget
from .maskers import RuntimeMasker, structurally_prune_units

__all__ = ["run_iterative_pruning"]

def _make_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    name = (cfg.get("optimizer") or "sgd").lower()
    lr = cfg.get("lr", 0.03)
    wd = cfg.get("weight_decay", 0.0)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=cfg.get("momentum", 0.9), nesterov=cfg.get("nesterov", True), weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)

def _finetune(model: nn.Module, train_loader, val_loader, cfg: Dict[str, Any], device: torch.device):
    if not cfg.get("enabled", True):
        return {}
    epochs = int(cfg.get("epochs", 3))
    amp = bool(cfg.get("amp", True))
    criterion = nn.CrossEntropyLoss()
    opt = _make_optimizer(model, cfg)
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 0.0) or float("inf"))
            opt.step()
    return {}

def run_iterative_pruning(
    model: nn.Module,
    train_loader,
    val_loader,
    units: List[Dict[str, Any]],
    score_fns: Dict[str, Callable],
    selector_cfg: Dict[str, Any],
    finetune_cfg: Dict[str, Any],
    step_budgets: List[float],
    *,
    runtime_masking: bool = True,
    structural: bool = False,
    eval_fn: Optional[Callable[[nn.Module, Any], Dict[str, Any]]] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:

    device = device or next(model.parameters()).device
    history: List[Dict[str, Any]] = []
    remaining_units = {u["id"]: u for u in units}  # mutable pool
    method = selector_cfg.get("method", "gndi")
    tie_break = selector_cfg.get("tie_break", "random")
    budget_type = selector_cfg.get("budget_type", "params")
    per_layer_cap = selector_cfg.get("per_layer_cap", None)
    score_kwargs = selector_cfg.get("score_kwargs", {})
    amp = bool(selector_cfg.get("amp", True))

    masker = RuntimeMasker() if runtime_masking else None

    cum_pruned: List[str] = []

    for step, b in enumerate(step_budgets, start=1):
        # (1) score current remaining units
        scorer = score_fns[method]
        # Build a thin dataloader iterator for scoring max_batches as per kwargs.
        # We pass train_loader (or val_loader) depending on method needs; typically train is fine.
        units_list = list(remaining_units.values())
        scores = scorer(model, train_loader, units_list, **score_kwargs)

        # (2) rank and select to meet this step's budget
        order = rank_units(scores, tie_break=tie_break)
        selected_ids = select_by_budget(order, units_list, budget_type=budget_type, budget_value=b, per_layer_cap=per_layer_cap)

        # (3) apply prune
        selected_units = [remaining_units[uid] for uid in selected_ids]
        if structural:
            structurally_prune_units(model, selected_units)
        else:
            if masker:
                masker.add(selected_units)
                masker.activate()

        # Remove from remaining pool
        for uid in selected_ids:
            remaining_units.pop(uid, None)
        cum_pruned.extend(selected_ids)

        # (4) optional finetune
        _ = _finetune(model, train_loader, val_loader, finetune_cfg, device)

        # (5) evaluate & log
        metrics = {}
        if eval_fn is not None:
            model.eval()
            with torch.no_grad():
                metrics = eval_fn(model, val_loader)

        history.append({
            "step": step,
            "pruned_ids": selected_ids,
            "metrics": metrics,
            "cum_pruned": len(cum_pruned),
            "remaining": len(remaining_units),
        })

    # Deactivate masker at the end if used
    if masker:
        masker.deactivate()
    return history
