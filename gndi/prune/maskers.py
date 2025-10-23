# gndi/prune/maskers.py
# -*- coding: utf-8 -*-
"""
Maskers: apply pruning either via runtime masking (forward hooks) or via structural edits.

Exposes:
- RuntimeMasker (hook-based zeroing of unit outputs)
  - add(unit_ids), remove(unit_ids), clear(), context()
- Structural pruners:
  - structurally_prune_units(model, units_to_prune)

Runtime masking is safe and reversible; structural pruning is permanent and must
update module parameters (e.g., delete conv out-channels / linear out-features).

The units share the same schema used elsewhere.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Iterable
import contextlib

import torch
import torch.nn as nn

__all__ = [
    "RuntimeMasker",
    "structurally_prune_units",
]

# --------------------------- Runtime Masking --------------------------

def _apply_unit_mask(output: torch.Tensor, unit: Dict[str, Any]) -> torch.Tensor:
    idx = int(unit["index"]); utype = unit["type"]

    if utype == "conv_channel":
        output[:, idx:idx+1, ...] = 0.0
    elif utype in ("fc_neuron", "ffn_neuron"):
        if output.dim() == 2:
            output[:, idx:idx+1] = 0.0
        elif output.dim() == 3:
            output[..., idx:idx+1] = 0.0
        else:
            raise ValueError(f"Unexpected rank for {utype}: {output.shape}")
    elif utype == "attn_head":
        if output.dim() != 4:
            # If head not explicitly separated, best-effort zero along likely head axis
            # Fall back to last-known: treat second dim as heads
            pass
        B, A, C, D = output.shape
        if A > idx:
            output[:, idx:idx+1, :, :] = 0.0
        elif C > idx:
            output[:, :, idx:idx+1, :] = 0.0
        else:
            # nothing to do safely
            return output
    else:
        raise ValueError(f"Unknown unit type: {utype}")
    return output

class RuntimeMasker:
    """
    Forward-hook based unit masking. You register units and the hook zeros
    their slices in the module outputs during forward().
    """
    def __init__(self):
        self._module_to_units: Dict[nn.Module, List[Dict[str, Any]]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._active = False

    def add(self, units: Iterable[Dict[str, Any]]):
        for u in units:
            self._module_to_units.setdefault(u["module"], []).append(u)

    def remove(self, unit_ids: Iterable[str]):
        unit_ids = set(unit_ids)
        for m in list(self._module_to_units.keys()):
            self._module_to_units[m] = [u for u in self._module_to_units[m] if u["id"] not in unit_ids]
            if not self._module_to_units[m]:
                del self._module_to_units[m]

    def clear(self):
        self.deactivate()
        self._module_to_units.clear()

    def activate(self):
        if self._active:
            return
        self._handles = []
        for m, units in self._module_to_units.items():
            def make_hook(bound_units):
                def _hook(_m, _inp, out):
                    # out can be Tensor or tuple; we support Tensor outputs
                    if isinstance(out, torch.Tensor):
                        out = out.clone()  # avoid in-place on shared tensor
                        for u in bound_units:
                            out = _apply_unit_mask(out, u)
                        return out
                    return out
                return _hook
            self._handles.append(m.register_forward_hook(make_hook(list(units))))
        self._active = True

    def deactivate(self):
        if not self._active:
            return
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        self._active = False

    @contextlib.contextmanager
    def context(self):
        self.activate()
        try:
            yield self
        finally:
            self.deactivate()

# -------------------------- Structural Pruning ------------------------

def _prune_conv2d_out_channel(mod: nn.Conv2d, idx: int):
    keep = [i for i in range(mod.out_channels) if i != idx]
    mod.out_channels = len(keep)
    mod.weight = nn.Parameter(mod.weight.data[keep, :, :, :].contiguous())
    if mod.bias is not None:
        mod.bias = nn.Parameter(mod.bias.data[keep].contiguous())

def _prune_linear_out_feature(mod: nn.Linear, idx: int):
    keep = [i for i in range(mod.out_features) if i != idx]
    mod.out_features = len(keep)
    mod.weight = nn.Parameter(mod.weight.data[keep, :].contiguous())
    if mod.bias is not None:
        mod.bias = nn.Parameter(mod.bias.data[keep].contiguous())

def _try_prune_attention_head(module: nn.Module, idx: int, head_dim: Optional[int] = None):
    """
    Best-effort structural prune for attention head by editing the output projection W_O:
    remove the columns corresponding to the head's subspace.
    """
    # Common attribute names
    candidates = []
    if hasattr(module, "out_proj") and hasattr(module.out_proj, "weight"):
        candidates.append(module.out_proj.weight)
    for attr in ("wo", "W_O", "o_proj", "proj"):
        if hasattr(module, attr):
            obj = getattr(module, attr)
            if isinstance(obj, nn.Linear):
                candidates.append(obj.weight)

    if not candidates:
        return False

    W = candidates[0]
    in_dim = W.shape[1]
    if head_dim is None:
        # Heuristic: infer number of heads from module if present
        n_heads = getattr(module, "num_heads", None) or getattr(module, "n_heads", None)
        d_model = getattr(module, "embed_dim", None) or getattr(module, "hidden_size", None)
        if n_heads and d_model and in_dim % n_heads == 0:
            head_dim = in_dim // n_heads
        else:
            return False

    if in_dim % head_dim != 0:
        return False
    n_heads = in_dim // head_dim
    if idx >= n_heads:
        return False

    col_slice = slice(idx * head_dim, (idx + 1) * head_dim)
    keep_cols = torch.cat([
        torch.arange(0, col_slice.start),
        torch.arange(col_slice.stop, in_dim)
    ], dim=0).long().to(W.device)

    # Update weight (and bias unaffected)
    newW = W.data[:, keep_cols].contiguous()
    # Try to assign back
    if hasattr(module, "out_proj") and hasattr(module.out_proj, "weight") and module.out_proj.weight is W:
        module.out_proj.in_features = newW.shape[1]
        module.out_proj.weight = nn.Parameter(newW)
        return True
    for attr in ("wo", "W_O", "o_proj", "proj"):
        if hasattr(module, attr):
            obj = getattr(module, attr)
            if isinstance(obj, nn.Linear) and obj.weight is W:
                obj.in_features = newW.shape[1]
                obj.weight = nn.Parameter(newW)
                return True
    return False

def _try_prune_ffn_neuron(module: nn.Module, idx: int):
    """
    Structural prune for FFN neuron by editing W1 (in->hidden) column and W2 (hidden->out) row.
    This assumes the module exposes two Linear layers named (commonly) 'fc1' and 'fc2'
    or 'linear1' and 'linear2'.
    """
    cands = []
    for n1, n2 in (("fc1", "fc2"), ("linear1", "linear2")):
        if hasattr(module, n1) and hasattr(module, n2):
            l1 = getattr(module, n1); l2 = getattr(module, n2)
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
                cands.append((l1, l2))
    if not cands:
        return False
    l1, l2 = cands[0]
    if idx >= l1.out_features or idx >= l2.in_features:
        return False

    keep_h = [i for i in range(l1.out_features) if i != idx]

    # Prune W1: columns of size in_features x out_features^T → PyTorch keeps [out, in]
    l1.out_features = len(keep_h)
    l1.weight = nn.Parameter(l1.weight.data[keep_h, :].contiguous())
    if l1.bias is not None:
        l1.bias = nn.Parameter(l1.bias.data[keep_h].contiguous())

    # Prune W2: remove corresponding row in W2 (out x in)
    l2.in_features = len(keep_h)
    l2.weight = nn.Parameter(l2.weight.data[:, keep_h].contiguous())
    return True

def structurally_prune_units(model: nn.Module, units_to_prune: List[Dict[str, Any]]) -> None:
    """
    Permanently remove units from the model by editing module parameters.
    This function assumes residual/skip compatibility is handled by upstream wrappers.
    """
    # Process units per module, descending indices to keep reindexing stable
    by_mod: Dict[nn.Module, List[Dict[str, Any]]] = {}
    for u in units_to_prune:
        by_mod.setdefault(u["module"], []).append(u)
    for mod, us in by_mod.items():
        # prune from highest index to lowest
        us = sorted(us, key=lambda x: x["index"], reverse=True)
        for u in us:
            idx = int(u["index"]); utype = u["type"]; meta = u.get("meta", {})
            if isinstance(mod, nn.Conv2d) and utype == "conv_channel":
                _prune_conv2d_out_channel(mod, idx)
            elif isinstance(mod, nn.Linear) and utype in ("fc_neuron", "ffn_neuron"):
                _prune_linear_out_feature(mod, idx)
            elif utype == "attn_head":
                _try_prune_attention_head(mod, idx, head_dim=meta.get("head_dim"))
            elif utype == "ffn_neuron":
                _try_prune_ffn_neuron(mod, idx)
            else:
                # Fallback: zeroing as structural removal is undefined
                pass
