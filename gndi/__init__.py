# gndi/__init__.py
# -*- coding: utf-8 -*-
"""
G-NDI — Causal Pruning Framework
--------------------------------

This package implements the full experimental suite described in the G-NDI spec:
causal importance scoring, causal validity benchmarking, and practical pruning
across CV (ResNet/VGG) and NLP (BERT/DistilBERT).

Public imports are organized so users can access all key components directly
from the `gndi` namespace.
"""

from .data import build_dataloaders, build_cv_dataloaders, build_nlp_dataloaders
from .models_cv import build_cv_model, enumerate_units_cv
from .models_nlp import build_nlp_model, enumerate_units_nlp
from .train import train_and_eval, evaluate

# Scoring methods
from .scoring.gndi import compute_gndi_units
from .scoring.magnitude import compute_magnitude_units
from .scoring.snip import compute_snip_units
from .scoring.grasp import compute_grasp_units
from .scoring.synflow import compute_synflow_units
from .scoring.hrank import compute_hrank_units

# Pruning utilities
from .prune.selectors import rank_units, select_by_budget
from .prune.maskers import RuntimeMasker, structurally_prune_units
from .prune.iterative import run_iterative_pruning

# Evaluation utilities
from .eval.casual_gt import measure_true_effect_units
from .eval.metrics import (
    compute_correlations,
    bootstrap_spearman,
    eval_classification,
    model_complexity,
    auc_accuracy_sparsity,
    accuracy_at_flops,
    aggregate_mean_std,
)
from .eval.reporter import Reporter

# Convenience registry (method name → scoring function)
SCORERS = {
    "gndi": compute_gndi_units,
    "magnitude": compute_magnitude_units,
    "snip": compute_snip_units,
    "grasp": compute_grasp_units,
    "synflow": compute_synflow_units,
    "hrank": compute_hrank_units,
}

__all__ = [
    # data / models
    "build_dataloaders", "build_cv_dataloaders", "build_nlp_dataloaders",
    "build_cv_model", "build_nlp_model",
    "enumerate_units_cv", "enumerate_units_nlp",

    # training
    "train_and_eval", "evaluate",

    # scoring
    "compute_gndi_units", "compute_magnitude_units", "compute_snip_units",
    "compute_grasp_units", "compute_synflow_units", "compute_hrank_units", "SCORERS",

    # pruning
    "rank_units", "select_by_budget", "RuntimeMasker",
    "structurally_prune_units", "run_iterative_pruning",

    # evaluation
    "measure_true_effect_units", "compute_correlations", "bootstrap_spearman",
    "eval_classification", "model_complexity", "auc_accuracy_sparsity",
    "accuracy_at_flops", "aggregate_mean_std", "Reporter",
]
