# gndi/eval/causal_gt.py
# -*- coding: utf-8 -*-
"""
Measure ground-truth intervention effects per unit by *masking* that unit and
observing the change at the model output or loss.

Exports:
- measure_true_effect_units(model, units, batch_iter, metric='output_delta'|'loss_delta',
                            p=2, batches=4, amp=True, criterion=None)

Returns:
  dict {unit_id: effect_value}

Notes
-----
- Uses lightweight forward hooks to zero a unit's contribution at its emitting module.
- 'output_delta': E_x || f(x) - f(-i)(x) ||_p  (default p=2)
- 'loss_delta'  : E_x max(0, L(-i) - L), i.e., positive loss increase

This is *read-only* (eval mode) and does not update params. Batches are pulled from
'batch_iter', which can be any iterable of (x, y). We iterate at most 'batches' items.
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, Optional

import contextlib
import torch
import torch.nn as nn
from tqdm import tqdm


def _amp(enabled: bool, device: torch.device):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)

@contextlib.contextmanager
def _mask_unit_hook(module: nn.Module, unit: Dict[str, Any]):
    """
    Register a forward hook that zeros the specific unit slice in the module output.
    """
    idx = int(unit["index"])
    utype = unit["type"]

    def hook(_m, _inp, out):
        if not isinstance(out, torch.Tensor):
            return out  # unsupported multi-output module
        out = out.clone()
        if utype == "conv_channel":
            out[:, idx:idx+1, ...] = 0.0
        elif utype in ("fc_neuron", "ffn_neuron"):
            if out.dim() == 2:
                out[:, idx:idx+1] = 0.0
            elif out.dim() == 3:
                out[..., idx:idx+1] = 0.0
        elif utype == "attn_head":
            # Try [B, H, T, D] first, then [B, T, H, D]
            if out.dim() == 4:
                if out.size(1) > idx:
                    out[:, idx:idx+1, :, :] = 0.0
                elif out.size(2) > idx:
                    out[:, :, idx:idx+1, :] = 0.0
        return out

    h = module.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()

def _p_norm(v: torch.Tensor, p: float) -> torch.Tensor:
    if p == float("inf"):
        return v.abs().amax(dim=tuple(range(1, v.dim())), keepdim=False)
    return v.flatten(1).norm(p=p, dim=1)

@torch.no_grad()
def measure_true_effect_units(
    model: nn.Module,
    units: Iterable[Dict[str, Any]],
    batch_iter: Iterable,
    *,
    metric: str = "output_delta",
    p: float = 2.0,
    batches: int = 4,
    amp: bool = True,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    See module docstring.
    """
    device = next(model.parameters()).device
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Initialize accumulators
    units = list(units)
    sums = {u["id"]: 0.0 for u in units}
    counts = {u["id"]: 0 for u in units}

    # Pull up to `batches` mini-batches
    it = iter(batch_iter)
    for bidx in range(batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True) if torch.is_tensor(y) else y

        device = next(model.parameters()).device
        with _amp(amp, device):
            base_out = model(x)
            if metric == "loss_delta":
                base_loss = criterion(base_out, y)

        for u in tqdm(units, desc="Measuring true effects", ncols=90):
            with _mask_unit_hook(u["module"], u):
                with _amp(amp, device):
                    alt_out = model(x)
                    if metric == "output_delta":
                        s = _p_norm(alt_out - base_out, p=p).mean().item()
                    elif metric == "loss_delta":
                        alt_loss = criterion(alt_out, y)
                        # positive damage only (max(0, L(-i)-L))
                        s = torch.clamp(alt_loss - base_loss, min=0.0).mean().item()
                    else:
                        raise ValueError("metric must be 'output_delta' or 'loss_delta'")

            sums[u["id"]] += s
            counts[u["id"]] += 1

    # Average across processed batches
    for k in sums:
        if counts[k] > 0:
            sums[k] /= counts[k]
        else:
            sums[k] = 0.0
    return sums
