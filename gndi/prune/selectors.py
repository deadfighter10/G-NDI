# gndi/prune/selectors.py
# -*- coding: utf-8 -*-
"""
Selectors: global ranking & budgeted selection.

Exposes:
- rank_units(scores, tie_break="random")
- estimate_unit_params(unit)
- estimate_unit_flops(unit, input_shape=None, seq_len=None)
- select_by_budget(order, units, budget_type, budget_value)

Assumptions about `unit`:
{
  "id": "layer3.conv2:17",
  "module": nn.Module,
  "type": "conv_channel" | "fc_neuron" | "ffn_neuron" | "attn_head",
  "index": int,
  # Optional hints to improve FLOPs estimates:
  "meta": {
     "in_channels": int,
     "out_channels": int,
     "kernel_size": (kH, kW),
     "stride": (sH, sW),
     "feature_shape": (B, C, H, W) OR (B, T, C) for Transformers,
     "head_dim": int,           # for attn heads
     "seq_len": int,            # for attn/ffn FLOPs
  }
}
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import math
import random

import torch
import torch.nn as nn

__all__ = [
    "rank_units",
    "estimate_unit_params",
    "estimate_unit_flops",
    "select_by_budget",
]

# ------------------------------ Ranking ------------------------------

def rank_units(scores: Dict[str, float], tie_break: str = "random") -> List[str]:
    """
    Returns a list of unit_ids sorted from LOWEST score to HIGHEST (to prune bottom first).
    tie_break: "random" | "stable" (stable = python sort stable order)
    """
    items = list(scores.items())
    if tie_break == "random":
        random.shuffle(items)
    # bottom-first
    items.sort(key=lambda kv: kv[1])
    return [k for k, v in items]

# -------------------------- Param accounting -------------------------

def _conv2d_params_per_out_channel(mod: nn.Conv2d) -> int:
    kH, kW = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
    per = mod.in_channels * kH * kW
    if mod.bias is not None:
        per += 1
    return per

def _linear_params_per_out_feature(mod: nn.Linear) -> int:
    per = mod.in_features
    if mod.bias is not None:
        per += 1
    return per

def estimate_unit_params(unit: Dict[str, Any]) -> int:
    """
    Rough parameter count associated to the unit.
    """
    mod = unit["module"]; utype = unit["type"]
    if isinstance(mod, nn.Conv2d) and utype == "conv_channel":
        return _conv2d_params_per_out_channel(mod)
    if isinstance(mod, nn.Linear) and utype in ("fc_neuron", "ffn_neuron"):
        return _linear_params_per_out_feature(mod)
    if utype == "attn_head":
        # Assume head contributes head_dim * embed_dim in output projection
        head_dim = unit.get("meta", {}).get("head_dim")
        embed_dim = getattr(mod, "embed_dim", None) or getattr(mod, "hidden_size", None) or None
        if head_dim and embed_dim:
            return head_dim * embed_dim
        # Fallback constant
        return 0
    # Fallback
    params = 0
    for p in mod.parameters(recurse=False):
        params += p.numel()
    return max(params // 32, 0)  # very rough

# ---------------------------- FLOPs accounting -----------------------

def _conv2d_flops_per_out_channel(mod: nn.Conv2d, H: int, W: int) -> int:
    """
    Multiply-accumulate (MACs) per output channel; FLOPs ~ 2*MACs (mul+add).
    """
    kH, kW = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
    outH = math.floor((H + 2*mod.padding[0] - kH) / mod.stride[0] + 1)
    outW = math.floor((W + 2*mod.padding[1] - kW) / mod.stride[1] + 1)
    macs = mod.in_channels * kH * kW * outH * outW
    return 2 * macs

def _linear_flops_per_out_feature(mod: nn.Linear) -> int:
    # FLOPs per output feature for one example is 2*in_features; aggregate externally by batch/time if desired
    return 2 * mod.in_features

def estimate_unit_flops(
    unit: Dict[str, Any],
    input_shape: Optional[Tuple[int, ...]] = None,
    seq_len: Optional[int] = None,
) -> int:
    """
    Rough FLOPs cost for the unit for ONE input example.
    For batches and time (seq), multiply externally if needed.
    """
    mod = unit["module"]; utype = unit["type"]
    meta = unit.get("meta", {})
    if isinstance(mod, nn.Conv2d) and utype == "conv_channel":
        H = W = None
        if input_shape and len(input_shape) == 4:
            H, W = input_shape[2], input_shape[3]
        elif "feature_shape" in meta and len(meta["feature_shape"]) == 4:
            H, W = meta["feature_shape"][2], meta["feature_shape"][3]
        if H is None or W is None:
            # fallback assume CIFAR
            H, W = 32, 32
        return _conv2d_flops_per_out_channel(mod, H, W)
    if isinstance(mod, nn.Linear) and utype in ("fc_neuron", "ffn_neuron"):
        fl = _linear_flops_per_out_feature(mod)
        # account for sequence length if provided (Transformers)
        L = seq_len or meta.get("seq_len")
        return fl * (L if L is not None else 1)
    if utype == "attn_head":
        # rough attention FLOPs: QK^T + softmax + AV ~ O(L^2*head_dim + L*head_dim*d_model)
        head_dim = meta.get("head_dim", 64)
        L = seq_len or meta.get("seq_len", 128)
        d_model = getattr(mod, "embed_dim", None) or getattr(mod, "hidden_size", 768)
        return 2 * (L * L * head_dim + L * head_dim * d_model)
    return 0

# ----------------------------- Selection -----------------------------

def _accumulate_budget(
    order: List[str],
    units: List[Dict[str, Any]],
    budget_type: str,
    budget_value: float,
) -> List[str]:
    """
    Select a prefix of 'order' such that the accumulated budget (params or flops)
    does not exceed budget_value *TOTAL* if budget_value in (0,1], else treat as absolute number.
    """
    id2unit = {u["id"]: u for u in units}

    # Precompute per-unit costs and total
    per = {}
    total = 0
    if budget_type == "params":
        for u in units:
            c = estimate_unit_params(u)
            per[u["id"]] = c
            total += c
    elif budget_type == "flops":
        for u in units:
            c = estimate_unit_flops(u)
            per[u["id"]] = c
            total += c
    else:
        raise ValueError("budget_type must be 'params' or 'flops'")

    if 0 < budget_value <= 1.0:
        limit = budget_value * total
    else:
        limit = float(budget_value)

    chosen = []
    acc = 0.0
    for uid in order:
        c = per.get(uid, 0)
        if acc + c > limit:
            break
        chosen.append(uid)
        acc += c
    return chosen

def select_by_budget(
    order: List[str],
    units: List[Dict[str, Any]],
    *,
    budget_type: str = "params",
    budget_value: float = 0.2,
    per_layer_cap: Optional[float] = None,
) -> List[str]:
    """
    Returns the list of unit_ids to prune given an ordering (bottom-first).
    - budget_type: "params" or "flops"
    - budget_value: fraction (0..1] of total or absolute count if >1
    - per_layer_cap: optional cap (fraction of units per layer/module)
    """
    if per_layer_cap is not None:
        # enforce cap by filtering order: keep at most cap per module
        counts = {}
        filtered = []
        for uid in order:
            mod = next(u["module"] for u in units if u["id"] == uid)
            counts.setdefault(mod, 0)
            # count units per module
            total_in_module = sum(1 for u in units if u["module"] is mod)
            cap_abs = max(int(per_layer_cap * total_in_module), 1)
            if counts[mod] < cap_abs:
                filtered.append(uid)
                counts[mod] += 1
        order = filtered

    selected = _accumulate_budget(order, units, budget_type, budget_value)
    return selected
