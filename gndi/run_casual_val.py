# gndi/run_causal_val.py
# -*- coding: utf-8 -*-
"""
Run causal validity: compute method scores vs ground truth ablation effects,
then report correlations and CI.
"""
from __future__ import annotations
import argparse, os, json, random
import yaml
import torch
from tqdm import tqdm

from .train import _build_model
from .data import build_dataloaders
from .models_cv import enumerate_units_cv
from .models_nlp import enumerate_units_nlp
from .scoring import REGISTRY as SCORERS
from .eval import measure_true_effect_units, compute_correlations, bootstrap_spearman, Reporter

def _is_nlp(cfg):
    name = str(cfg.get("dataset", {}).get("name", "")).lower()
    return name in ["ag_news","agnews","sst2","sst-2","yelp_polarity","yelp","mnli"]

def _enumerate_units(model, cfg):
    return enumerate_units_nlp(model, cfg) if _is_nlp(cfg) else enumerate_units_cv(model, cfg)

# add near the imports
import inspect

def _call_scorer(scorer, model, loader, units, score_kwargs):
    sig = inspect.signature(scorer)
    allowed = {k: v for k, v in (score_kwargs or {}).items() if k in sig.parameters}
    return scorer(model, loader, units, **allowed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--units", type=int, default=300)
    ap.add_argument("--method", type=str, default="gndi")
    ap.add_argument("--baseline", type=str, default="zero")
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    out_dir = cfg.get("paths", {}).get("out_dir", "./runs/tmp")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(cfg)
    model = _build_model(cfg).to(device)

    # Quick warmup if requested
    warmup = int(cfg.get("score", {}).get("warmup_epochs", 0))
    if warmup > 0:
        from .train import _make_optimizer
        opt = _make_optimizer(model.parameters(), {"train":{"optimizer":"sgd","lr":0.01,"weight_decay":0.0}})
        crit = torch.nn.CrossEntropyLoss()
        model.train()
        for i in range(warmup):
            print(f"\n[Epoch {i + 1}/{warmup}]")
            for xb, yb in tqdm(loaders["train"], desc="Warmup", ncols=80):
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=next(model.parameters()).device.type, enabled=True):
                    logits = model(xb)
                    loss = crit(logits, yb)
                loss.backward(); opt.step()

    # Enumerate and subsample units
    units_all = _enumerate_units(model, cfg)
    random.seed(cfg.get("seed", 42))
    random.shuffle(units_all)
    units_eval = units_all[:args.units]

    # Score
    scorer = SCORERS[args.method]
    score_kwargs = cfg.get("score", {}).get(args.method, {})
    scores = _call_scorer(scorer, model, loaders["train"], units_eval, score_kwargs)

    # Ground truth
    gt_cfg = cfg.get("causal_ground_truth", {})
    gt = measure_true_effect_units(
        model, units_eval, loaders["val"],
        metric=gt_cfg.get("metric", "output_delta"),
        p=gt_cfg.get("p_norm", 2),
        batches=gt_cfg.get("batches", 4),
        amp=bool(cfg.get("amp", True))
    )

    # Correlations & CI
    corr = compute_correlations(scores, gt)
    ci = bootstrap_spearman(scores, gt, rounds=gt_cfg.get("bootstrap_rounds", 1000), ci=gt_cfg.get("ci", 0.95))
    stats = {"pearson":corr["pearson"], "spearman":corr["spearman"], "kendall":corr["kendall"], "ci_low":ci["low"], "ci_high":ci["high"]}

    json.dump({"scores":scores, "ground_truth":gt, "stats":stats}, open(os.path.join(out_dir, "causal_validity.json"), "w"), indent=2)

    rep = Reporter(out_dir)
    ds_name = cfg["dataset"]["name"]; mdl = cfg["model"]["arch"]
    rep.log_causal_corr(ds_name, mdl, args.method, stats)
    rep.save_tables(); rep.build_html(title="Causal Validity Report")

if __name__ == "__main__":
    main()
