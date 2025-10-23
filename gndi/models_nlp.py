# gndi/models_nlp.py
# -*- coding: utf-8 -*-
"""
NLP models (HF) and unit enumeration for FFN neurons (robust) and
best-effort attention heads (via output projection columns if head_dim inferred).

- build_nlp_model(cfg) -> nn.Module
- enumerate_units_nlp(model, cfg) -> List[unit dicts]

Assumes AutoModelForSequenceClassification. FFN neurons are exposed from
Transformer blocks' intermediate (linear1/fc1) layers. Heads are approximated
by slicing the attention output projection input space (columns grouped per head).
"""
from __future__ import annotations
from typing import List, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

Unit = Dict[str, Any]

def build_nlp_model(cfg) -> nn.Module:
    arch = cfg.get("model", {}).get("arch", "bert-base-uncased")
    num_classes = cfg.get("dataset", {}).get("num_classes", None)
    if num_classes is None:
        # fall back to typical datasets (AG=4, SST=2, Yelp=2, MNLI=3)
        name = str(cfg.get("dataset", {}).get("name", "")).lower()
        num_classes = 4 if "ag" in name else (2 if "sst" in name or "yelp" in name else 3)
    mcfg = AutoConfig.from_pretrained(arch, num_labels=num_classes)
    model = AutoModelForSequenceClassification.from_pretrained(arch, config=mcfg)
    return model

def _iter_transformer_blocks(model: nn.Module):
    # Works for BERT/DistilBERT/RoBERTa style (encoders.layers/encoder.layer)
    for name, mod in model.named_modules():
        if any(name.endswith(s) for s in [".intermediate", ".ffn.lin1", ".ffn.intermediate", ".linear1", ".fc1"]):
            yield name, mod

def enumerate_units_nlp(model: nn.Module, cfg) -> List[Unit]:
    want = cfg.get("model", {}).get("unit_kinds", ["ffn"])
    want = [w.lower() for w in want]
    units: List[Unit] = []

    # FFN neurons (Robust)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(k in name.lower() for k in ["intermediate.dense", "ffn.lin1", "linear1", "fc1"]):
            for j in range(mod.out_features):
                units.append({
                    "id": f"{name}:{j}",
                    "module": mod,
                    "type": "ffn_neuron",
                    "index": j,
                })

    # Attention heads (best effort via out_proj columns)
    if any(k in want for k in ["heads", "both", "attn", "attn_heads"]):
        # Identify attention blocks and their output projections
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and any(s in name.lower() for s in ["attention.output.dense", "attn.out_proj", "o_proj", "wo"]):
                # Guess number of heads from nearby config
                # Hidden size
                hidden = getattr(mod, "in_features", None)
                # Try to fetch num_heads from enclosing module/config
                n_heads = getattr(model.config, "num_attention_heads", None)
                if hidden and n_heads and hidden % n_heads == 0:
                    head_dim = hidden // n_heads
                    for h in range(n_heads):
                        units.append({
                            "id": f"{name}:head{h}",
                            "module": mod,          # attention out-proj
                            "type": "attn_head",
                            "index": h,
                            "meta": {"head_dim": head_dim, "seq_len": cfg.get("dataset", {}).get("max_length", 128)}
                        })
                # else: skip, can’t infer reliable head_dim
    return units
