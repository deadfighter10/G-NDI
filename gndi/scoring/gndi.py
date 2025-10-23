# gndi/scoring/gndi.py
# -*- coding: utf-8 -*-
"""
G-NDI: Causal first-order effect via directional intervention and suffix re-evaluation.

We approximate the JVP term J_{>L}(x) (b_L - h_L(x)) with a small directional
forward difference: f(h + α v) - f(h), where v is (b - h)_restricted_to_unit.

This keeps the implementation robust without requiring an explicit forward-mode JVP
decomposition of the model into prefix/suffix. Your model wrappers should expose
units and ensure that the module's forward output contains the unit along a known axis:

- conv_channel: module output shaped [B, C, H, W]
- fc_neuron:    module output shaped [B, C] or [B, T, C] (neuron along last dim)
- ffn_neuron:   same as fc_neuron (Transformer MLP output)
- attn_head:    module output shaped one of: [B, Heads, T, D] or [B, T, Heads, D]
                (we auto-detect the head axis by matching 'index')

Config knobs exposed via kwargs:
- p_norm: 1|2|float("inf")  (default 2)
- baseline: "zero"|"batch_mean"|"identity" (identity: residual bypass = no change)
- alpha: small step size for finite difference (default 1e-2)
- max_batches: limit scoring batches for efficiency
- amp: bool mixed precision guard
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import contextlib

import torch
import torch.nn as nn

__all__ = ["compute_gndi_units"]

from tqdm import tqdm


def _amp_guard(enabled: bool, device: torch.device):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)

def _unit_restrict_like(h: torch.Tensor, unit: Dict[str, Any], vec: Optional[torch.Tensor] = None, fill: float = 0.0) -> torch.Tensor:
    """
    Create a tensor same-shape as h with zeros everywhere except the slice for `unit`.
    If `vec` is provided, copy the matching slice from `vec` into that slice;
    otherwise copy the slice from `h` (or fill with a constant if desired).
    """
    t = torch.zeros_like(h)
    utype = unit["type"]
    idx = int(unit["index"])

    src_base = h if vec is None else vec  # use vec's slice when given

    if utype == "conv_channel":
        # h: [B, C, H, W]
        t[:, idx:idx+1, ...] = src_base[:, idx:idx+1, ...]
        return t

    if utype in ("fc_neuron", "ffn_neuron"):
        # h: [B, C] or [B, T, C] (neuron along last dim)
        if h.dim() == 2:
            t[:, idx:idx+1] = src_base[:, idx:idx+1]
        elif h.dim() == 3:
            t[..., idx:idx+1] = src_base[..., idx:idx+1]
        else:
            raise ValueError(f"Unexpected tensor rank for {utype}: {h.shape}")
        return t

    if utype == "attn_head":
        # Supported formats: [B, Heads, T, D] or [B, T, Heads, D]
        if h.dim() != 4:
            raise ValueError(f"Unexpected tensor rank for attn_head: {h.shape}")
        B, A, C, D = h.shape
        # Prefer axis=1 as heads; fallback to axis=2 if it fits idx
        if A > idx:
            slc = (slice(None), slice(idx, idx+1), slice(None), slice(None))
        elif C > idx:
            slc = (slice(None), slice(None), slice(idx, idx+1), slice(None))
        else:
            raise ValueError(f"Cannot locate head axis for shape {h.shape} and head idx={idx}")
        t[slc] = src_base[slc]
        return t

    raise ValueError(f"Unknown unit type: {utype}")


def _make_baseline(h: torch.Tensor, unit: Dict[str, Any], kind: str) -> torch.Tensor:
    """
    Construct baseline b with same shape as h.
    - 'zero': zero-out the unit slice (default, safe everywhere)
    - 'batch_mean': replace with batch mean (lower-variance shot)
    - 'identity': residual bypass -> equivalent to removing the residual branch,
                  so baseline for the residual's output should be ZERO.
    """
    if kind == "zero":
        return torch.zeros_like(h)

    elif kind == "batch_mean":
        # per-tensor batch mean (we only copy the unit's slice later)
        mean = h.mean(dim=0, keepdim=True).expand_as(h)
        return mean

    elif kind == "identity":
        # IMPORTANT: for residual branches, 'identity' means "skip connection only"
        # => residual branch contribution = 0
        return torch.zeros_like(h)

    else:
        raise ValueError(f"Unknown baseline kind: {kind}")

@contextlib.contextmanager
def _replace_output_hook(module: nn.Module, new_output: torch.Tensor):
    """
    Temporarily replace the module's output by returning new_output from its forward hook.
    """
    handle = None
    def hook(_mod, _inp, _out):
        return new_output
    handle = module.register_forward_hook(lambda m, i, o: hook(m, i, o))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()

def _model_forward(model: nn.Module, batch: Any, amp: bool, device: torch.device) -> torch.Tensor:
    x, _ = batch
    with _amp_guard(amp, device):
        return model(x)


def _p_norm(v: torch.Tensor, p: float) -> torch.Tensor:
    if p == float("inf"):
        return v.abs().amax(dim=tuple(range(1, v.dim())), keepdim=False)
    return v.flatten(1).norm(p=p, dim=1)

def compute_gndi_units(
    model: nn.Module,
    dataloader,
    units: List[Dict[str, Any]],
    *,
    p_norm: float = 2.0,
    baseline: str = "zero",
    alpha: float = 1e-2,
    max_batches: int = 8,
    amp: bool = True,
    reduction: str = "mean",
    ** kwargs,
) -> Dict[str, float]:
    """
    Returns {unit_id: score}, where score is E_x || f(h + α v_u) - f(h) ||_p.

    Notes:
    - Requires the *unit.module* to be exactly the module whose output contains the unit.
    - Uses a forward hook to override that module's output with h or h+αv.
    - alpha is automatically scaled by the unit slice magnitude for stability.
    """

    device = next(model.parameters()).device
    model.eval()
    scores = {u["id"]: 0.0 for u in units}
    counts = {u["id"]: 0 for u in units}

    # Cache one clean forward per batch; then per unit, re-run with replacement
    num_batches = 0
    for batch in tqdm(dataloader, total=max_batches, desc=f"GNDI scoring ({len(units)} units)", ncols=90):
        if num_batches >= max_batches:
            break
        num_batches += 1
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True) if torch.is_tensor(y) else y

        # First pass: get original outputs and capture h per-module
        # We capture module outputs with simple hooks.
        module_to_h = {}

        def make_capture(mod):
            return mod.register_forward_hook(lambda m, i, o: module_to_h.__setitem__(m, o))

        handles = [make_capture(u["module"]) for u in units]
        device = next(model.parameters()).device
        with _amp_guard(amp, device):
            base_out = model(x)
        for h in handles: h.remove()

        for u in tqdm(units, desc="Units", leave=False, ncols=80):
            uid = u["id"]
            module = u["module"]
            h = module_to_h.get(module, None)
            if h is None:
                # Module didn't run (e.g., conditional path); skip safely
                continue

            # Build baseline and directional vector v = (b - h)_restricted_to_unit
            b = _make_baseline(h, u, baseline)
            diff = (b - h).detach()
            v_u = _unit_restrict_like(h, u, vec=diff)

            # Scale α by unit-slice magnitude to keep step comparable
            denom = (_unit_restrict_like(h, u).abs().mean() + 1e-8).item()
            step = alpha if denom == 0 else alpha * denom

            # Evaluate outputs with h (sanity) and h+αv
            # 1) f(h): just reuse base_out
            with torch.no_grad():
                y0 = base_out

            # 2) f(h + α v)
            h_pert = (h + step * v_u).detach()
            with _replace_output_hook(module, h_pert), torch.no_grad(), _amp_guard(amp, device):
                y1 = model(x)

            # Per-sample Δ and p-norm
            delta = y1 - y0
            s = _p_norm(delta, p=p_norm)
            s_val = s.mean().item() if reduction == "mean" else s.sum().item()

            scores[uid] += s_val
            counts[uid] += 1

    # Normalize by number of batches processed
    for uid in scores:
        if counts[uid] > 0:
            scores[uid] /= counts[uid]
        else:
            scores[uid] = 0.0
    return scores
