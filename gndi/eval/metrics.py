# gndi/eval/metrics.py
# -*- coding: utf-8 -*-
"""
Common evaluation metrics & utilities.

Exports:
- compute_correlations(pred: dict, true: dict) -> {'pearson': ..., 'spearman': ..., 'kendall': ...}
- bootstrap_spearman(pred, true, rounds=1000, ci=0.95, seed=123)
- eval_classification(model, loader, amp=True) -> {'acc@1': float}
- model_complexity(model, sample_input=None) -> {'flops': int|None, 'params': int}
- auc_accuracy_sparsity(points)  # list of (sparsity_frac, acc) tuples
- accuracy_at_flops(points, targets=[0.3, 0.5])
- aggregate_mean_std(records)    # list[float] -> {'mean':..., 'std':...}
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, List, Optional
import math
import random


import numpy as np
import torch
import torch.nn as nn

# Optional stats deps
try:
    from scipy.stats import spearmanr, pearsonr, kendalltau
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _align_xy(pred: Dict[str, float], true: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    # Intersect keys and keep consistent order
    keys = sorted(set(pred.keys()) & set(true.keys()))
    x = np.asarray([pred[k] for k in keys], dtype=np.float64)
    y = np.asarray([true[k] for k in keys], dtype=np.float64)
    return x, y

def _nan_guard(v: float) -> float:
    return float(v) if (v == v and math.isfinite(v)) else 0.0

def compute_correlations(pred: Dict[str, float], true: Dict[str, float]) -> Dict[str, float]:
    """
    Defensive correlation computation that returns 0.0 on degeneracy
    (e.g., constant vectors).
    """
    x, y = _align_xy(pred, true)
    out = {"pearson": 0.0, "spearman": 0.0, "kendall": 0.0}
    if len(x) < 2:
        return out

    if _HAS_SCIPY:
        try:
            pr = pearsonr(x, y)[0]
            sr = spearmanr(x, y)[0]
            kt = kendalltau(x, y)[0]
        except Exception:
            pr = sr = kt = 0.0
    else:
        # Fallback Pearson
        try:
            xm, ym = x - x.mean(), y - y.mean()
            pr = float((xm @ ym) / (np.linalg.norm(xm) * np.linalg.norm(ym) + 1e-12))
        except Exception:
            pr = 0.0
        # Fallback Spearman via rank corr
        try:
            rx = np.argsort(np.argsort(x))
            ry = np.argsort(np.argsort(y))
            rxm, rym = rx - rx.mean(), ry - ry.mean()
            sr = float((rxm @ rym) / (np.linalg.norm(rxm) * np.linalg.norm(rym) + 1e-12))
        except Exception:
            sr = 0.0
        kt = 0.0

    return {
        "pearson": _nan_guard(pr),
        "spearman": _nan_guard(sr),
        "kendall": _nan_guard(kt),
    }

def bootstrap_spearman(
    pred: Dict[str, float],
    true: Dict[str, float],
    *,
    rounds: int = 1000,
    ci: float = 0.95,
    seed: int = 123,
) -> Dict[str, float]:
    """
    Non-parametric bootstrap for Spearman rho CI.
    Returns dict: {'rho': float, 'low': float, 'high': float}
    """
    rng = np.random.default_rng(seed)
    x, y = _align_xy(pred, true)
    n = len(x)
    if n < 2:
        return {"rho": 0.0, "low": 0.0, "high": 0.0}

    def _spearman(a, b):
        if _HAS_SCIPY:
            try:
                return float(spearmanr(a, b)[0])
            except Exception:
                return 0.0
        # fallback
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        am, bm = ra - ra.mean(), rb - rb.mean()
        den = (np.linalg.norm(am) * np.linalg.norm(bm) + 1e-12)
        return float((am @ bm) / den)

    base = _spearman(x, y)
    boots = []
    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        boots.append(_spearman(x[idx], y[idx]))
    boots = np.sort(np.asarray(boots))
    lo = (1 - ci) / 2
    hi = 1 - lo
    return {"rho": _nan_guard(base), "low": float(boots[int(lo * (rounds - 1))]), "high": float(boots[int(hi * (rounds - 1))])}


@torch.no_grad()
def eval_classification(model: nn.Module, loader, *, amp: bool = True) -> Dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.amp.autocast(device_type=device.type, enabled=amp):
        for batch in loader:
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

            # Get logits based on input type
            if isinstance(x, dict):
                outputs = model(**x)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            else:
                logits = model(x)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    acc = 100.0 * correct / max(total, 1)
    return {"acc@1": acc}


def model_complexity(model: nn.Module, sample_input: Optional[torch.Tensor] = None) -> Dict[str, Optional[int]]:
    """
    Returns {'flops': int or None, 'params': int}
    Tries fvcore/ptflops if available; otherwise flops=None.
    """
    params = sum(p.numel() for p in model.parameters())
    flops = None

    # Try fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        if sample_input is not None:
            flops = int(FlopCountAnalysis(model, sample_input).total())
    except Exception:
        pass

    # Try ptflops
    if flops is None:
        try:
            from ptflops import get_model_complexity_info
            if sample_input is not None:
                # sample_input: [B, C, H, W] -> use single example
                shape = tuple(sample_input.shape[1:])
                macs, params_pt = get_model_complexity_info(model, shape, as_strings=False, print_per_layer_stat=False)
                flops = int(macs * 2)
        except Exception:
            pass

    return {"flops": flops, "params": params}

def auc_accuracy_sparsity(points: List[Tuple[float, float]]) -> float:
    """
    Compute AUC over sparsity in [0, 0.5] (or whatever range provided), sorted by sparsity.
    points: list of (sparsity_frac, accuracy_percent)
    """
    if not points:
        return 0.0
    pts = sorted(points, key=lambda t: t[0])
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        auc += 0.5 * (y0 + y1) * (x1 - x0)
    return float(auc)

def accuracy_at_flops(points: List[Tuple[float, float]], targets: List[float] = [0.3, 0.5]) -> Dict[float, float]:
    """
    Given (flops_reduction_frac, accuracy) points, interpolate accuracy at given targets.
    """
    if not points:
        return {t: 0.0 for t in targets}
    pts = sorted(points, key=lambda t: t[0])
    xs, ys = zip(*pts)
    out = {}
    for t in targets:
        if t <= xs[0]: out[t] = ys[0]; continue
        if t >= xs[-1]: out[t] = ys[-1]; continue
        # linear interp
        for i in range(len(xs) - 1):
            if xs[i] <= t <= xs[i+1]:
                w = (t - xs[i]) / max(xs[i+1] - xs[i], 1e-12)
                out[t] = (1 - w) * ys[i] + w * ys[i+1]
                break
    return {k: float(v) for k, v in out.items()}

def aggregate_mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0))}
