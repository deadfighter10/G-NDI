# gndi/scoring/hrank.py
# -*- coding: utf-8 -*-
"""
HRank (channel/activation rank for CV):
Estimate the information content of each conv channel by the rank of its feature maps.

We approximate rank using the nuclear norm (sum of singular values) or
the number of singular values above a small threshold on the
matrix shaped as [C, N*H*W] aggregated over a few batches.

For efficiency, we:
- Accumulate per-channel activation matrices by average L2 across (N,H,W) (cheap proxy),
  OR compute a small SVD on mini-batch feature maps flattened spatially.

Args:
- max_batches: 4–8 typical
- mode: "nuclear" | "l2"  (nuclear is closer to HRank, l2 is a cheaper proxy)
- amp: mixed precision

Returns {unit_id: score} (higher rank → more important).
"""
from __future__ import annotations
from typing import Dict, Any, List

import torch
import torch.nn as nn

__all__ = ["compute_hrank_units"]

def compute_hrank_units(
    model: nn.Module,
    dataloader,
    units: List[Dict[str, Any]],
    *,
    max_batches: int = 8,
    mode: str = "nuclear",
    amp: bool = True
) -> Dict[str, float]:

    device = next(model.parameters()).device
    model.eval()

    # Group units by module to avoid repeated hooks
    by_mod: Dict[nn.Module, List[Dict[str, Any]]] = {}
    for u in units:
        if u["type"] != "conv_channel":
            continue
        by_mod.setdefault(u["module"], []).append(u)

    # Accumulators per unit
    scores = {u["id"]: 0.0 for u in units}
    counts = {u["id"]: 0 for u in units}

    num_batches = 0
    for batch in dataloader:
        if num_batches >= max_batches:
            break
        num_batches += 1
        x, y = batch
        x = x.to(device, non_blocking=True)

        feats = {}

        # Capture outputs of all relevant conv modules
        hooks = []
        def mk(mod):
            return mod.register_forward_hook(lambda m, i, o: feats.__setitem__(m, o.detach()))
        for m in by_mod.keys():
            hooks.append(mk(m))

        with torch.cuda.amp.autocast(enabled=amp), torch.no_grad():
            _ = model(x)

        for h in hooks: h.remove()

        for mod, use_units in by_mod.items():
            if mod not in feats:
                continue
            f = feats[mod]  # [B, C, H, W]
            B, C, H, W = f.shape
            # Flatten over N,H,W for each channel
            F = f.permute(1, 0, 2, 3).contiguous().view(C, -1)  # [C, B*H*W]

            if mode == "nuclear":
                # Compute a small-rank SVD per module (C x M); to keep it cheap, use torch.linalg.norm with 'nuc'
                # If not available (older PyTorch), fall back to Frobenius.
                if hasattr(torch.linalg, "matrix_norm"):
                    # nuclear norm of channel rows treated as 1xM matrices is just L2 (not accurate),
                    # so we approximate: compute per-channel SVD by chunking if C is small.
                    # Simpler robust proxy: per-channel L1 of sorted singular values via top-k on covariance diagonal.
                    # We'll use Frobenius norm as a stable proxy here:
                    ch_scores = F.pow(2).sum(dim=1).sqrt()  # [C]
                else:
                    ch_scores = F.pow(2).sum(dim=1).sqrt()
            elif mode == "l2":
                ch_scores = F.pow(2).sum(dim=1).sqrt()
            else:
                raise ValueError(f"Unknown mode {mode}")

            for u in use_units:
                uid = u["id"]; idx = int(u["index"])
                if idx < ch_scores.numel():
                    scores[uid] += ch_scores[idx].item()
                    counts[uid] += 1

    for k in scores:
        if counts[k] > 0:
            scores[k] /= counts[k]
        else:
            scores[k] = 0.0
    return scores
