# gndi/scoring/synflow.py
# -*- coding: utf-8 -*-
"""
SynFlow (data-free): propagate ones to measure flow through weights.

Implementation outline:
1) Switch to "positive" weights by taking absolute values with sign preservation trick.
2) Use an input of ones (matching input shape) and sum model output S = sum(f(1)).
3) Backprop to get ∂S/∂θ; saliency is |θ ⊙ ∂S/∂θ|.
4) Aggregate to unit-level.

Args:
- input_shape: tuple like (B, C, H, W) for CV or (B, T) / (B, C, T) for NLP encoders if applicable
  (For Transformers with tokenizers, SynFlow is less meaningful; keep for CV.)
- batch_size: synthetic batch B
- amp: mixed precision

Returns {unit_id: score}.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["compute_synflow_units"]

def _positive_params(model: nn.Module):
    signs = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            s = torch.sign(p)
            p.data = p.data.abs()
            signs[n] = s
    return signs

def _restore_params(model: nn.Module, signs: Dict[str, torch.Tensor]):
    for n, p in model.named_parameters():
        if p.requires_grad and n in signs:
            p.data = p.data * signs[n]

def compute_synflow_units(
    model: nn.Module,
    dataloader,  # unused
    units: List[Dict[str, Any]],
    *,
    input_shape: Optional[Tuple[int, ...]] = None,
    batch_size: int = 16,
    amp: bool = True
) -> Dict[str, float]:

    device = next(model.parameters()).device
    model.eval()

    # Infer an input shape if not provided by peeking a real batch (CV path)
    if input_shape is None:
        # Try to draw one batch from a dataloader if it exists
        x_try = None
        for batch in dataloader or []:
            x_try = batch[0]
            break
        if x_try is None:
            raise ValueError("SynFlow requires input_shape or a dataloader to infer it.")
        input_shape = tuple(x_try.shape)

    # Create synthetic ones input
    ones = torch.ones((batch_size,) + tuple(input_shape[1:]), device=device)

    # Make weights positive
    signs = _positive_params(model)

    # Forward sum and backprop
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    dev = next(model.parameters()).device
    with torch.amp.autocast(device_type=dev.type, enabled=amp):
        out = model(ones)
        s = out.sum()

    s.backward()

    # Unit saliency: |θ ⊙ ∂S/∂θ|
    per_unit = {u["id"]: 0.0 for u in units}
    for u in units:
        uid = u["id"]; mod = u["module"]; idx = int(u["index"]); utype = u["type"]
        if isinstance(mod, nn.Conv2d) and utype == "conv_channel":
            w = mod.weight; g = w.grad if w.grad is not None else torch.zeros_like(w)
            per_unit[uid] = (w[idx].abs() * g[idx].abs()).sum().item()
        elif isinstance(mod, nn.Linear) and utype in ("fc_neuron", "ffn_neuron"):
            w = mod.weight; g = w.grad if w.grad is not None else torch.zeros_like(w)
            per_unit[uid] = (w[idx].abs() * g[idx].abs()).sum().item()
        else:
            # Fallback: sum over module params
            s_val = 0.0
            for p in mod.parameters():
                if p.grad is not None:
                    s_val += (p.abs() * p.grad.abs()).sum().item()
            per_unit[uid] = s_val

    # Restore signs
    _restore_params(model, signs)

    return per_unit
