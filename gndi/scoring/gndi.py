# gndi/scoring/gndi.py
# -*- coding: utf-8 -*-
"""
G-NDI (JVP version): Causal first-order effect
Δ̂_L(x) = J_{>L}(x) (b_L - h_L(x))
Score per sample: CIS_L(x) = || Δ̂_L(x) ||_p
Score per unit:  E_x[CIS_L(x)] using the unit-restricted vector (b - h)_u.

This implements the *true* JVP using torch.autograd.functional.jvp by
treating the module output as an explicit variable z and defining a closure:

    F(z) := model(x) with that module's output replaced by (h.detach() + z)

Then   J_{>L}(x) v  = jvp(F, (0,), (v,)).jvp   where v is unit-restricted (b-h).
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import contextlib

import torch
import torch.nn as nn
from tqdm import tqdm

__all__ = ["compute_gndi_units"]


# ------------------------------- utilities -------------------------------

def _amp_guard(enabled: bool, device: torch.device):
    # Device-agnostic autocast (works on MPS/CPU/CUDA)
    return torch.amp.autocast(device_type=device.type, enabled=enabled)

@contextlib.contextmanager
def _replace_output_hook(module: nn.Module, new_output: torch.Tensor):
    """Temporarily replace a module's forward output with `new_output`."""
    handle = module.register_forward_hook(lambda m, i, o: new_output)
    try:
        yield
    finally:
        handle.remove()

def _unit_restrict_like(h: torch.Tensor, unit: Dict[str, Any],
                        vec: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Return a tensor like h with zeros everywhere except the slice for `unit`.
    If vec is given, use the unit-slice from vec; else copy from h.
    """
    t = torch.zeros_like(h)
    idx = int(unit["index"])
    src = h if vec is None else vec
    utype = unit["type"]

    if utype == "conv_channel":          # [B, C, H, W]
        t[:, idx:idx+1, ...] = src[:, idx:idx+1, ...]
    elif utype in ("fc_neuron", "ffn_neuron"):
        if h.dim() == 2:                 # [B, C]
            t[:, idx:idx+1] = src[:, idx:idx+1]
        elif h.dim() == 3:               # [B, T, C]
            t[..., idx:idx+1] = src[..., idx:idx+1]
        else:
            raise ValueError(f"Unexpected rank for {utype}: {h.shape}")
    elif utype == "attn_head":
        # Accept [B, Heads, T, D] or [B, T, Heads, D]
        if h.dim() != 4:
            raise ValueError(f"Unexpected rank for attn_head: {h.shape}")
        _, A, C, _ = h.shape
        if A > idx:
            t[:, idx:idx+1, :, :] = src[:, idx:idx+1, :, :]
        elif C > idx:
            t[:, :, idx:idx+1, :] = src[:, :, idx:idx+1, :]
        else:
            raise ValueError(f"Cannot locate head axis for shape {h.shape} / idx={idx}")
    else:
        raise ValueError(f"Unknown unit type: {utype}")
    return t

def _p_norm(v: torch.Tensor, p: float) -> torch.Tensor:
    if p == float("inf"):
        return v.abs().amax(dim=tuple(range(1, v.dim())), keepdim=False)
    return v.flatten(1).norm(p=p, dim=1)

def _make_baseline(h: torch.Tensor, kind: str) -> torch.Tensor:
    """
    Construct baseline b with same shape as h.
    - 'zero'       : zeros
    - 'batch_mean' : repeat batch mean across batch
    - 'identity'   : for residual branches, identity means "skip-only",
                     i.e. residual contribution = 0  (=> b = 0)
    """
    if kind == "zero":
        return torch.zeros_like(h)
    if kind == "batch_mean":
        return h.mean(dim=0, keepdim=True).expand_as(h)
    if kind == "identity":
        # IMPORTANT: residual branch contribution set to 0
        return torch.zeros_like(h)
    raise ValueError(f"Unknown baseline kind: {kind}")


# ------------------------------- main API --------------------------------

def compute_gndi_units(
    model: nn.Module,
    dataloader,
    units: List[Dict[str, Any]],
    *,
    p_norm: float = 2.0,
    baseline: str = "zero",
    max_batches: int = 8,
    amp: bool = True,
    reduction: str = "mean",
    **kwargs,
) -> Dict[str, float]:
    """
    Compute G-NDI scores for the provided `units` using the analytical JVP:
        Δ̂_L(x) = J_{>L}(x) (b_L - h_L(x))
    Returns {unit_id: E_x ||Δ̂_L(x)||_p }.

    Notes
    -----
    - Assumes each unit dict has: {"id", "module", "type", "index"}.
    - Works with conv channels, FC/FFN neurons, and (best-effort) attention heads.
    - No parameter grads are needed; this uses functional JVP over an activation variable.
    """

    device = next(model.parameters()).device
    model.eval()

    # Prepare accumulators
    scores = {u["id"]: 0.0 for u in units}
    counts = {u["id"]: 0 for u in units}

    # Iterate over a limited number of batches
    num_batches = 0
    for batch in tqdm(dataloader, total=max_batches, desc=f"GNDI scoring ({len(units)} units)", ncols=92):
        if num_batches >= max_batches:
            break
        num_batches += 1

        x, y = batch
        x = x.to(device, non_blocking=True)
        if torch.is_tensor(y):  # y unused here, but move if tensor
            y = y.to(device, non_blocking=True)

        # 1) Forward once to capture the per-module outputs h_L(x)
        module_to_h = {}
        def _capture_hook(m, i, o):
            module_to_h[m] = o

        hooks = [u["module"].register_forward_hook(_capture_hook) for u in units]
        with _amp_guard(amp, device):
            _ = model(x)
        for h in hooks:
            h.remove()

        # 2) For each unit, compute JVP at z=0 with v = (b - h)_restricted_to_unit
        for u in tqdm(units, desc="Units", leave=False, ncols=80):
            uid = u["id"]
            mod = u["module"]
            h = module_to_h.get(mod, None)
            if h is None:
                continue  # module not executed (conditional path)

            # Build baseline and perturbation vector
            b = _make_baseline(h, baseline)         # same shape as h
            v_full = (b - h).detach()               # do NOT require grad on v
            v_u = _unit_restrict_like(h, u, vec=v_full)

            # Define F(z): run model with module output replaced by (h_detached + z)
            h_det = h.detach()

            def F(z: torch.Tensor) -> torch.Tensor:
                with _replace_output_hook(mod, h_det + z), _amp_guard(amp, device):
                    return model(x)

            # Evaluate jvp at z0 = 0 (same shape as h)
            z0 = torch.zeros_like(h_det, requires_grad=False)
            # jvp returns (F(z0), J@v); we only need the jvp term
            y0, jv = torch.autograd.functional.jvp(F, (z0,), (v_u,), create_graph=False, strict=False)

            # score for this batch
            s = _p_norm(jv, p=p_norm)
            s_val = s.mean().item() if reduction == "mean" else s.sum().item()
            scores[uid] += s_val
            counts[uid] += 1

        # Early sanity after first batch
        if num_batches == 1:
            nz = sum(v > 0 for v in scores.values())
            print(f"[GNDI/JVP] after 1 batch: nonzero={nz}/{len(scores)}")

    # Average over batches
    for k in scores:
        scores[k] = (scores[k] / counts[k]) if counts[k] > 0 else 0.0

    return scores
