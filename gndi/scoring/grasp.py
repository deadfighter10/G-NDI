# gndi/scoring/grasp.py
# -*- coding: utf-8 -*-
"""
GraSP (Gradient Signal Preservation) approximation.

Goal: select weights that preserve gradient flow (maximize alignment between
saliency and the gradient of the loss).

Practical proxy here:
- Compute gradient g = ∂L/∂θ on a batch.
- Compute an approximation to H g via a second backward of (g · θ) (HVP trick).
- Saliency per parameter is -θ * (H g), aggregated to the unit.

This is a lightweight approximation aligned with the GraSP spirit (without full HVP infra).

Args:
- max_batches: usually 1–2
- criterion: loss (CrossEntropy by default)
- amp: mixed precision

Returns {unit_id: score}.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

__all__ = ["compute_grasp_units"]

def _zero_grads(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

def compute_grasp_units(
    model: nn.Module,
    dataloader,
    units: List[Dict[str, Any]],
    *,
    max_batches: int = 1,
    criterion: Optional[nn.Module] = None,
    amp: bool = True,
) -> Dict[str, float]:

    device = next(model.parameters()).device
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Collect grads g
    _zero_grads(model)
    num = 0
    for batch in dataloader:
        if num >= max_batches: break
        num += 1
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        dev = next(model.parameters()).device
        with torch.amp.autocast(device_type=dev.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y) / max_batches  # average across mini-batches

        loss.backward()

    # Snapshot g
    params = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    # Compute H g via grad of dot(g, grad(params)) wrt params
    _zero_grads(model)
    dot = 0.0
    for p, g in zip(params, grads):
        if p.grad is not None:
            p.grad.zero_()
    # Create graph for HVP
    hvp_inputs = []
    for p, g in zip(params, grads):
        dot = dot + (p * g).sum()
        hvp_inputs.append(p)

    # Backward to get gradients equivalent to H g
    torch.autograd.backward(hvp_inputs, grad_tensors=grads, retain_graph=False)

    # Saliency per param: -θ * (H g)
    # After the HVP step, p.grad contains H g
    per_unit = {u["id"]: 0.0 for u in units}
    for u in units:
        uid = u["id"]; mod = u["module"]; idx = int(u["index"]); utype = u["type"]
        if isinstance(mod, nn.Conv2d) and utype == "conv_channel":
            w = mod.weight; g = w.grad if w.grad is not None else torch.zeros_like(w)
            s = -(w[idx] * g[idx]).sum().item()
        elif isinstance(mod, nn.Linear) and utype in ("fc_neuron", "ffn_neuron"):
            w = mod.weight; g = w.grad if w.grad is not None else torch.zeros_like(w)
            s = -(w[idx] * g[idx]).sum().item()
        else:
            # Fallback: sum over all params in module
            s = 0.0
            for p in mod.parameters():
                if p.grad is not None:
                    s -= (p * p.grad).sum().item()
        per_unit[uid] = s

    return per_unit
