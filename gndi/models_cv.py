# gndi/models_cv.py
# -*- coding: utf-8 -*-
"""
CV models and unit enumeration.
- build_cv_model(cfg) -> nn.Module
- enumerate_units_cv(model, cfg) -> List[unit dicts]
"""
from __future__ import annotations
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torchvision.models as tvm

Unit = Dict[str, Any]

def build_cv_model(cfg) -> nn.Module:
    arch = cfg.get("model", {}).get("arch", "resnet18").lower()
    pretrained = bool(cfg.get("model", {}).get("pretrained", False))
    num_classes = int(cfg.get("dataset", {}).get("num_classes", 10))

    if arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "vgg16":
        m = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(f"Unsupported CV arch: {arch}")

def _is_residual_conv(module_path: str) -> bool:
    # In torchvision ResNet, residual branch convs are inside layer{1..4}.{block}.conv1/conv2
    # We exclude downsample layers.
    return (".conv1" in module_path) or (".conv2" in module_path)

def enumerate_units_cv(model: nn.Module, cfg) -> List[Unit]:
    expose = cfg.get("model", {}).get("expose_units", "conv_channels")
    prune_residual_only = bool(cfg.get("model", {}).get("prune_residual_branch_only", True))

    units: List[Unit] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and expose == "conv_channels":
            if prune_residual_only:
                if ("layer" not in name) or ("downsample" in name) or (not _is_residual_conv(name)):
                    continue
            for c in range(mod.out_channels):
                units.append({
                    "id": f"{name}:{c}",
                    "module": mod,
                    "type": "conv_channel",
                    "index": c,
                    "meta": {
                        "in_channels": mod.in_channels,
                        "out_channels": mod.out_channels,
                        "kernel_size": mod.kernel_size,
                        "stride": mod.stride,
                    }
                })
        if isinstance(mod, nn.Linear) and expose == "fc_neurons":
            for j in range(mod.out_features):
                units.append({
                    "id": f"{name}:{j}",
                    "module": mod,
                    "type": "fc_neuron",
                    "index": j,
                })
    return units
