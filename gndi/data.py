# gndi/data.py
# -*- coding: utf-8 -*-
"""
Unified data builders for CV (torchvision) and NLP (HF datasets).
Config-driven and reproducible. Exposes:
  - build_dataloaders(cfg): returns dict(train=..., val=...)
  - build_cv_dataloaders(cfg)
  - build_nlp_dataloaders(cfg)

Expected config structure (see configs/*.yaml):
  dataset:
    name: cifar10 | cifar100 | imagenet_subset | ag_news | sst2 | yelp_polarity | mnli
    # CV
    image_size: 32|224
    normalize: true
    augment: { random_crop: true, crop_padding: 4, random_horizontal_flip: true, randaugment: true, cutout: 0 }
    num_workers: 8
    pin_memory: true
    train_split: train
    val_split: test|val|validation
    # NLP
    tokenizer_name: bert-base-uncased
    max_length: 128
    pad_to_max_length: true

  train:
    batch_size: int
  eval_batch_size: int  (under train or top-level; both supported)

Paths:
  paths:
    data_root: ./data
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Callable

import torch
from torch.utils.data import DataLoader, Subset

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --- CV imports
try:
    import torchvision
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

# --- NLP imports
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding
    _HAS_HF = True
except Exception:
    _HAS_HF = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Helper to read nested dicts by dotted path, e.g., "dataset.name".
    """
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_eval_batch_size(cfg: Dict[str, Any]) -> int:
    bs_eval = _get(cfg, "train.eval_batch_size")
    if bs_eval is None:
        bs_eval = _get(cfg, "eval_batch_size")
    if bs_eval is None:
        # default: 2x train batch size (often safe for eval)
        bs_eval = int(_get(cfg, "train.batch_size", 128) * 2)
    return bs_eval


# ---------------------------------------------------------------------------
# CV pipelines
# ---------------------------------------------------------------------------

def _cv_mean_std(dataset_name: str):
    # Standard normalization for CIFAR and ImageNet
    if dataset_name.lower() in ["cifar10", "cifar100"]:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        # ImageNet stats
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    return mean, std


class Cutout(object):
    """
    Simple Cutout augmentation. Cut out a square of length 'length'.
    """
    def __init__(self, length: int):
        self.length = length

    def __call__(self, img):
        if self.length <= 0:
            return img
        # img: PIL Image -> convert to tensor space for mask application
        t = transforms.ToTensor()(img)
        h, w = t.shape[1], t.shape[2]
        y = random.randrange(h)
        x = random.randrange(w)
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        t[:, y1:y2, x1:x2] = 0.0
        # Back to PIL
        return transforms.ToPILImage()(t)

def _maybe_subset(dataset, fraction):
    if fraction is None or fraction >= 1.0:
        return dataset
    import torch
    subset_size = int(len(dataset) * fraction)
    indices = torch.randperm(len(dataset))[:subset_size]
    from torch.utils.data import Subset
    return Subset(dataset, indices)

def _build_cv_transforms(cfg: Dict[str, Any], split: str):
    ds_cfg = cfg.get("dataset", {})
    name = str(ds_cfg.get("name", "")).lower()
    img_size = int(ds_cfg.get("image_size", 32))
    augment_cfg = ds_cfg.get("augment", {}) or {}
    normalize = bool(ds_cfg.get("normalize", True))
    mean, std = _cv_mean_std(name)

    aug = []
    # Common
    if split == "train":
        if augment_cfg.get("random_resized_crop", False):
            aug.append(transforms.RandomResizedCrop(img_size))
        elif augment_cfg.get("random_crop", False):
            crop_padding = int(augment_cfg.get("crop_padding", 4))
            aug.append(transforms.RandomCrop(img_size, padding=crop_padding))
        else:
            aug.append(transforms.Resize(img_size))

        if augment_cfg.get("random_horizontal_flip", True):
            aug.append(transforms.RandomHorizontalFlip())

        if augment_cfg.get("color_jitter", 0.0) and name.startswith("imagenet"):
            aug.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))

        if augment_cfg.get("randaugment", False):
            # torchvision >= 0.13 has RandAugment
            if hasattr(transforms, "RandAugment"):
                aug.append(transforms.RandAugment())
        if int(augment_cfg.get("cutout", 0)) > 0:
            aug.append(Cutout(int(augment_cfg["cutout"])))
    else:
        aug.append(transforms.Resize(img_size))
        if name.startswith("imagenet"):
            # For ImageNet-style eval, center crop is common
            aug.append(transforms.CenterCrop(img_size))

    aug.append(transforms.ToTensor())
    if normalize:
        aug.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(aug)


def _build_cifar(dataset_name: str, root: str, split: str, transform):
    assert dataset_name in ["cifar10", "cifar100"]
    if dataset_name == "cifar10":
        ds_cls = CIFAR10
    else:
        ds_cls = CIFAR100
    train = split.lower() in ["train", "training"]
    return ds_cls(root=root, train=train, download=True, transform=transform)


def _build_imagenet_subset(root: str, split: str, transform):
    """
    Assumes ImageNet-like folder structure under `root`:
      root/train/<class_name>/*.JPEG
      root/val/<class_name>/*.JPEG
    """
    split = "train" if split.lower() in ["train", "training"] else "val"
    path = os.path.join(root, split)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"ImageNet subset path not found: {path}")
    return ImageFolder(path, transform=transform)


def build_cv_datasets(cfg: Dict[str, Any]):
    if not _HAS_TORCHVISION:
        raise ImportError("torchvision is required for CV datasets. Please install torchvision.")

    ds_cfg = cfg.get("dataset", {})
    name = str(ds_cfg.get("name", "")).lower()
    root = _get(cfg, "paths.data_root", "./data")
    train_split = ds_cfg.get("train_split", "train")
    val_split = ds_cfg.get("val_split", "test")

    tfm_train = _build_cv_transforms(cfg, split="train")
    tfm_val = _build_cv_transforms(cfg, split="val")

    if name in ["cifar10", "cifar100"]:
        train_ds = _build_cifar(name, root, train_split, tfm_train)
        val_ds = _build_cifar(name, root, val_split, tfm_val)
        num_classes = 10 if name == "cifar10" else 100
    elif name in ["imagenet_subset", "imagenet-subset", "imagenet_subset_200"]:
        root = _get(cfg, "paths.data_root", "./data/imagenet_subset")
        train_ds = _build_imagenet_subset(root, train_split, tfm_train)
        val_ds = _build_imagenet_subset(root, val_split, tfm_val)
        num_classes = len(train_ds.classes)
    else:
        raise ValueError(f"Unsupported CV dataset name: {name}")

    return train_ds, val_ds, num_classes


def build_cv_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)

    ds = cfg.get("dataset", {})
    import torch
    IS_MPS = torch.backends.mps.is_available()

    # when building dataloaders
    num_workers = 0 if IS_MPS else int(ds.get("num_workers", 8))
    pin_memory = False if IS_MPS else bool(ds.get("pin_memory", True))
    train_bs = int(_get(cfg, "train.batch_size", 128))
    eval_bs = int(_infer_eval_batch_size(cfg))

    train_ds, val_ds, _ = build_cv_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader}


# ---------------------------------------------------------------------------
# NLP pipelines
# ---------------------------------------------------------------------------

@dataclass
class NLPBuild:
    tokenizer: Any
    train_dataset: Any
    val_dataset: Any
    num_classes: int
    collate_fn: Optional[Callable] = None


def _load_hf_dataset(name: str, split: str, cache_dir: Optional[str] = None):
    """
    Loads HF datasets with consistent split naming.
    """
    name = name.lower()
    if name == "ag_news" or name == "agnews":
        ds_name = "ag_news"
        # splits: train, test
    elif name in ["sst2", "sst-2"]:
        ds_name = "glue"
        # splits: train, validation
    elif name in ["yelp_polarity", "yelp"]:
        ds_name = "yelp_polarity"
        # splits: train, test
    elif name == "mnli":
        ds_name = "glue"
        # splits: train, validation_matched, validation_mismatched, test_*
    else:
        raise ValueError(f"Unsupported NLP dataset: {name}")

    if ds_name == "glue":
        if name in ["sst2", "sst-2"]:
            subset = "sst2"
        elif name == "mnli":
            subset = "mnli"
        else:
            raise ValueError(f"Unsupported GLUE subset mapping for: {name}")

        return load_dataset(ds_name, subset, split=split, cache_dir=cache_dir)
    else:
        return load_dataset(ds_name, split=split, cache_dir=cache_dir)


def _nlp_label_info(name: str) -> Tuple[str, int]:
    """
    Returns (label_column, num_classes)
    """
    name = name.lower()
    if name in ["ag_news", "agnews"]:
        return "label", 4
    if name in ["sst2", "sst-2"]:
        return "label", 2
    if name in ["yelp_polarity", "yelp"]:
        return "label", 2
    if name == "mnli":
        return "label", 3
    raise ValueError(f"No label info for dataset: {name}")


def _nlp_text_columns(name: str) -> Tuple[str, Optional[str]]:
    """
    Returns (primary_text, secondary_text?) column names for tokenization.
    """
    name = name.lower()
    if name in ["ag_news", "agnews"]:
        return "text", None
    if name in ["sst2", "sst-2"]:
        return "sentence", None
    if name in ["yelp_polarity", "yelp"]:
        return "text", None
    if name == "mnli":
        return "premise", "hypothesis"
    raise ValueError(f"No text column mapping for dataset: {name}")


def _tokenize_function_builder(tokenizer, max_length: int, pad_to_max_length: bool, text_col: str, text_pair_col: Optional[str]):
    def _fn(batch):
        if text_pair_col is None:
            return tokenizer(
                batch[text_col],
                truncation=True,
                max_length=max_length,
                padding="max_length" if pad_to_max_length else False,
            )
        else:
            return tokenizer(
                batch[text_col],
                batch[text_pair_col],
                truncation=True,
                max_length=max_length,
                padding="max_length" if pad_to_max_length else False,
            )
    return _fn


def build_nlp_datasets(cfg: Dict[str, Any]) -> NLPBuild:
    if not _HAS_HF:
        raise ImportError("Hugging Face datasets/transformers are required for NLP. Please install datasets & transformers.")

    ds_cfg = cfg.get("dataset", {})
    name = str(ds_cfg.get("name", "")).lower()
    train_split = ds_cfg.get("train_split", "train")
    val_split = ds_cfg.get("val_split", "validation" if name in ["sst2", "sst-2", "mnli"] else "test")
    tokenizer_name = ds_cfg.get("tokenizer_name", "bert-base-uncased")
    max_length = int(ds_cfg.get("max_length", 128))
    pad_to_max_length = bool(ds_cfg.get("pad_to_max_length", True))
    cache_dir = _get(cfg, "paths.data_root", None)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Map splits for GLUE/MNLI consistency
    if name == "mnli":
        # val: validation_matched by default
        if val_split.lower() in ["val", "validation"]:
            val_split = "validation_matched"

    train_ds = _load_hf_dataset(name, train_split, cache_dir=cache_dir)
    val_ds = _load_hf_dataset(name, val_split, cache_dir=cache_dir)

    text_col, text_pair_col = _nlp_text_columns(name)
    token_fn = _tokenize_function_builder(tokenizer, max_length, pad_to_max_length, text_col, text_pair_col)

    # Vectorized tokenization
    train_ds = train_ds.map(token_fn, batched=True, remove_columns=[c for c in train_ds.column_names if c not in [text_col, text_pair_col, "label"]])
    val_ds = val_ds.map(token_fn, batched=True, remove_columns=[c for c in val_ds.column_names if c not in [text_col, text_pair_col, "label"]])

    # Set format to torch (HF datasets)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in train_ds.features else []) + ["label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in val_ds.features else []) + ["label"])

    _, num_classes = _nlp_label_info(name)

    # Collator: dynamic padding if not fixed-length
    collate = None
    if not pad_to_max_length:
        collate = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

    return NLPBuild(
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_classes=num_classes,
        collate_fn=collate
    )


def build_nlp_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)

    ds = cfg.get("dataset", {})
    import torch
    IS_MPS = torch.backends.mps.is_available()

    # when building dataloaders
    num_workers = 0 if IS_MPS else int(ds.get("num_workers", 8))
    pin_memory = False if IS_MPS else bool(ds.get("pin_memory", True))

    train_bs = int(_get(cfg, "train.batch_size", 32))
    eval_bs = int(_infer_eval_batch_size(cfg))

    nlp_build = build_nlp_datasets(cfg)

    train_loader = DataLoader(
        nlp_build.train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=nlp_build.collate_fn,
    )

    val_loader = DataLoader(
        nlp_build.val_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=nlp_build.collate_fn,
    )

    return {"train": train_loader, "val": val_loader}


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------

def _subset_dataset(ds, fraction: float, seed: int):
    if ds is None or fraction is None or fraction >= 1.0:
        return ds
    n = len(ds)
    k = max(1, int(n * float(fraction)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[:k]
    return Subset(ds, idx.tolist())

def _rebuild_loader(base: DataLoader, new_ds) -> DataLoader:
    # Recreate a DataLoader with the same “shape” as the original one
    return DataLoader(
        new_ds,
        batch_size=base.batch_size,
        shuffle=getattr(base, "shuffle", False),    # DataLoader doesn’t expose this, so we default False
        num_workers=base.num_workers,
        pin_memory=base.pin_memory,
        collate_fn=base.collate_fn,
        drop_last=base.drop_last,
        persistent_workers=getattr(base, "persistent_workers", False),
    )

def build_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Main entrypoint used by train/eval scripts.
    Detects dataset type, builds loaders, and (optionally) subsets them.
    """
    name = str(_get(cfg, "dataset.name", "")).lower()

    # 1) Build base loaders by domain
    if name in ["cifar10", "cifar100", "imagenet_subset", "imagenet-subset", "imagenet_subset_200"]:
        loaders = build_cv_dataloaders(cfg)
    elif name in ["ag_news", "agnews", "sst2", "sst-2", "yelp_polarity", "yelp", "mnli"]:
        loaders = build_nlp_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # 2) Optional: take only a fraction of each split’s dataset
    frac = _get(cfg, "dataset.subset_fraction", None)
    if frac is not None and float(frac) < 1.0:
        seed = int(_get(cfg, "seed", 42))
        new_loaders: Dict[str, DataLoader] = {}
        for split, loader in loaders.items():
            ds_sub = _subset_dataset(loader.dataset, float(frac), seed=seed + hash(split) % 997)
            new_loaders[split] = _rebuild_loader(loader, ds_sub)
        loaders = new_loaders

    return loaders

# ---------------------------------------------------------------------------
# Introspection helpers (optional, nice to have)
# ---------------------------------------------------------------------------

def infer_num_classes(cfg: Dict[str, Any]) -> int:
    """
    Builds minimal dataset objects to infer num_classes without full dataloaders.
    """
    name = str(_get(cfg, "dataset.name", "")).lower()
    if name in ["cifar10", "cifar100", "imagenet_subset", "imagenet-subset", "imagenet_subset_200"]:
        _, _, ncls = build_cv_datasets(cfg)
        return ncls
    elif name in ["ag_news", "agnews", "sst2", "sst-2", "yelp_polarity", "yelp", "mnli"]:
        _, ncls = _nlp_label_info(name)
        return ncls
    else:
        raise ValueError(f"Unknown dataset name for class inference: {name}")


def sanity_sample(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a single batch sample dict for quick shape/debug checks.
    """
    loaders = build_dataloaders(cfg)
    batch = next(iter(loaders["train"]))
    if isinstance(batch, (list, tuple)):
        return {"tuple_len": len(batch), "types": [type(x) for x in batch]}
    if isinstance(batch, dict):
        return {k: (v.shape if torch.is_tensor(v) else type(v)) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return {"tensor": batch.shape}
    return {"info": f"unknown batch type: {type(batch)}"}
