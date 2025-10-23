# gndi/scoring/magnitude.py
# -*- coding: utf-8 -*-
"""
Magnitude baseline: unit weight norms.

For a module that *emits* the unit on its output, we approximate unit importance by:
- conv_channel: ||W[out_ch, :, :, :]||_{p}  (default L1)
- fc_neuron / ffn_neuron: ||W[:, out_idx]||_{p} or ||W[out_idx, :]|| depending on definition.
  We assume the emitting layer is Linear producing features along the last dim; i.e., W: [out_dim, in_dim].
- attn_head: sum of norms of the projection parameters for that head (e.g., W_O rows for that head).
  Because implementations vary, we provide a fallback based on output-projection weight chunking if present,
  else we fall back to activation proxy using a small loader (optional).

Args:
- norm: "l1"|"l2"
- head_dim: optional int to slice per-head in W_O if available.
- attn_proj_attr: name of output projection param on the module (e.g., "out_proj.weight" for fairseq-style,
                  "wo.weight" or "W_O" for some wrappers). We'll try a few names automatically.

Returns {unit_id: score}.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

__all__ = ["compute_magnitude_units"]

def _lpnorm(t: torch.Tensor, p: str) -> float:
    if p == "l1":
        return t.abs().sum().item()
    elif p == "l2":
        return t.pow(2).sum().sqrt().item()
    else:
        raise ValueError(f"Unknown norm {p}")

def _find_param(module: nn.Module, names: List[str]) -> Optional[torch.Tensor]:
    for n in names:
        obj = module
        ok = True
        for part in n.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, torch.Tensor):
            return obj
    return None

def compute_magnitude_units(
    model: nn.Module,
    dataloader,  # unused; kept for signature harmony
    units: List[Dict[str, Any]],
    *,
    norm: str = "l1",
    head_dim: Optional[int] = None,
    attn_proj_attr: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    scores = {}

    for u in units:
        uid = u["id"]
        mod = u["module"]
        idx = int(u["index"])
        utype = u["type"]

        if isinstance(mod, nn.Conv2d) and utype == "conv_channel":
            # Conv2d weight: [out_ch, in_ch, kH, kW]; unit is an output channel
            w = mod.weight[idx, :, :, :]
            scores[uid] = _lpnorm(w, norm)
        elif isinstance(mod, nn.Linear) and utype in ("fc_neuron", "ffn_neuron"):
            # Linear weight: [out_features, in_features]; neuron is an output feature
            w = mod.weight[idx, :]
            scores[uid] = _lpnorm(w, norm)
        elif utype == "attn_head":
            # Try to find an output projection weight of shape [embed_dim, embed_dim]
            # and slice per head if head_dim is provided.
            # Attempt several common attribute names
            names = [attn_proj_attr] if attn_proj_attr else [
                "out_proj.weight", "wo.weight", "W_O", "W_O.weight", "o_proj.weight", "proj.weight"
            ]
            w = _find_param(mod, [n for n in names if n])
            if w is not None and head_dim is not None:
                # Interpret columns as heads chunks along the input dimension
                in_dim = w.shape[1]
                if in_dim % head_dim == 0:
                    n_heads = in_dim // head_dim
                    if idx < n_heads:
                        sl = slice(idx * head_dim, (idx + 1) * head_dim)
                        scores[uid] = _lpnorm(w[:, sl], norm)
                        continue
            # Fallback: whole weight norm (weak but consistent)
            scores[uid] = _lpnorm(w if w is not None else torch.zeros(1, device=next(model.parameters()).device), norm)
        else:
            # Fallback: parameter magnitude on the first parameter of the module (rough)
            params = [p for p in mod.parameters(recurse=False)]
            if params:
                scores[uid] = _lpnorm(params[0], norm)
            else:
                scores[uid] = 0.0

    return scores
