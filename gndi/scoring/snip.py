# gndi/scoring/snip.py
# -*- coding: utf-8 -*-
"""
SNIP (connection sensitivity at init/warmup):
Saliency for parameter θ is |∂L/∂θ ⊙ θ| evaluated on a (small) batch.
Unit score aggregates parameter saliencies corresponding to that unit.

Assumptions:
- For conv_channel: aggregate saliency over conv.weight[out_ch, :, :, :]
- For (fc|ffn)_neuron: aggregate over linear.weight[out_idx, :]
- For attn_head: try to aggregate over out-projection chunk corresponding to the head (if detectable).
  Otherwise, sum over all parameters in the module as fallback.

Args:
- max_batches: small integer (1–2)
- criterion: loss function (default CrossEntropyLoss)
- at_init: if True, assumes model is at (near) init; else works after warmup as well
- amp: mixed precision

Returns {unit_id: score}.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

__all__ = ["compute_snip_units"]

def _accumulate_grads(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

def _ensure_list(t):
    return t if isinstance(t, list) else [t]

def compute_snip_units(
    model: nn.Module,
    dataloader,
    units: List[Dict[str, Any]],
    *,
    max_batches: int = 2,
    criterion: Optional[nn.Module] = None,
    amp: bool = True,
    head_dim: Optional[int] = None,
) -> Dict[str, float]:

    device = next(model.parameters()).device
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Collect saliency map: |grad * weight|
    saliency = {}
    for u in units:
        saliency[u["id"]] = 0.0

    num = 0
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # forward AMP only; grads normal
    for batch in dataloader:
        if num >= max_batches: break
        num += 1
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        _accumulate_grads(model)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()

        # Aggregate per unit
        for u in units:
            uid = u["id"]; mod = u["module"]; idx = int(u["index"])
            if isinstance(mod, nn.Conv2d) and u["type"] == "conv_channel":
                w = mod.weight
                g = w.grad
                s = (g[idx] * w[idx]).abs().sum().item()
            elif isinstance(mod, nn.Linear) and u["type"] in ("fc_neuron", "ffn_neuron"):
                w = mod.weight
                g = w.grad
                s = (g[idx] * w[idx]).abs().sum().item()
            elif u["type"] == "attn_head":
                # Try out-proj chunk by head_dim if weight present
                w = getattr(mod, "out_proj", None)
                w = getattr(w, "weight", None) if w is not None else None
                if w is not None and head_dim is not None and w.grad is not None:
                    in_dim = w.shape[1]
                    if in_dim % head_dim == 0:
                        n_heads = in_dim // head_dim
                        if idx < n_heads:
                            sl = slice(idx * head_dim, (idx + 1) * head_dim)
                            s = (w.grad[:, sl] * w[:, sl]).abs().sum().item()
                        else:
                            s = 0.0
                    else:
                        s = (w.grad * w).abs().sum().item()
                else:
                    # Fallback: sum over first param in module
                    par = [p for p in mod.parameters() if p.grad is not None]
                    s = sum([(p.grad * p).abs().sum().item() for p in par]) if par else 0.0
            else:
                par = [p for p in mod.parameters() if p.grad is not None]
                s = sum([(p.grad * p).abs().sum().item() for p in par]) if par else 0.0

            saliency[uid] += s

        _accumulate_grads(model)

    # Normalize by number of batches
    if num > 0:
        for k in saliency:
            saliency[k] /= num
    return saliency
