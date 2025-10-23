# gndi/run_prune.py
# -*- coding: utf-8 -*-
"""
Run pruning curves: for each method, prune to budgets and eval accuracy/FLOPs.
"""
from __future__ import annotations
import argparse, os, json, random
import yaml, torch

from .train import train_and_eval, evaluate
from .models_cv import enumerate_units_cv
from .models_nlp import enumerate_units_nlp
from .scoring import REGISTRY as SCORERS
from .prune import rank_units, select_by_budget, RuntimeMasker, structurally_prune_units
from .eval import Reporter

def _is_nlp(cfg):
    name = str(cfg.get("dataset", {}).get("name", "")).lower()
    return name in ["ag_news","agnews","sst2","sst-2","yelp_polarity","yelp","mnli"]

def _enumerate_units(model, cfg):
    return enumerate_units_nlp(model, cfg) if _is_nlp(cfg) else enumerate_units_cv(model, cfg)

import inspect
def _call_scorer(scorer, model, loader, units, score_kwargs):
    sig = inspect.signature(scorer)
    allowed = {k: v for k, v in (score_kwargs or {}).items() if k in sig.parameters}
    return scorer(model, loader, units, **allowed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="+", default=["gndi","magnitude"])
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.01,0.02,0.05,0.10,0.20,0.30,0.50])
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    out_dir = cfg.get("paths", {}).get("out_dir", "./runs/tmp")
    os.makedirs(out_dir, exist_ok=True)

    # Train a dense model first (or you can load an existing checkpoint)
    model, loaders = train_and_eval(cfg)
    device = next(model.parameters()).device

    units_all = _enumerate_units(model, cfg)

    rep = Reporter(out_dir)
    ds_name = cfg["dataset"]["name"]; mdl = cfg["model"]["arch"]

    for method in args.methods:
        scorer = SCORERS[method]
        score_kwargs = cfg.get("score", {}).get(method, {})
        scores = _call_scorer(scorer, model, loaders["train"], units_all, score_kwargs)
        order = rank_units(scores, tie_break="random")

        # Evaluate across budgets
        acc_sparsity, acc_flops = [], []
        for b in args.budgets:
            sel = select_by_budget(order, units_all,
                                   budget_type=cfg.get("prune", {}).get("budget_type", "params"),
                                   budget_value=b,
                                   per_layer_cap=cfg.get("prune", {}).get("per_layer_cap", None))
            # Runtime masking for evaluation; then quick finetune if enabled
            masker = RuntimeMasker()
            masker.add([u for u in units_all if u["id"] in sel])
            with masker.context():
                # Optional short finetune
                if cfg.get("finetune", {}).get("enabled", True):
                    from .train import _make_optimizer
                    opt = _make_optimizer(model.parameters(), {"train":{"optimizer":"sgd","lr":cfg.get("finetune",{}).get("lr",0.03),"weight_decay":cfg.get("finetune",{}).get("weight_decay",0.0005),"momentum":0.9}})
                    criterion = torch.nn.CrossEntropyLoss()
                    model.train()
                    for _ in range(int(cfg.get("finetune", {}).get("epochs", 2))):
                        for xb, yb in loaders["train"]:
                            xb, yb = xb.to(device), yb.to(device)
                            opt.zero_grad(set_to_none=True)
                            with torch.cuda.amp.autocast(enabled=bool(cfg.get("amp", True))):
                                logits = model(xb); loss = criterion(logits, yb)
                            loss.backward(); opt.step()

                # Eval
                metrics = evaluate(model, loaders["val"], amp=bool(cfg.get("amp", True)))
            sparsity = b if b <= 1 else b  # already fraction if <=1
            acc_sparsity.append((sparsity, metrics["acc@1"]))
            # FLOPs reduction not recomputed structurally here; approximate via budget fraction
            flops_red = sparsity if cfg.get("prune", {}).get("budget_type", "params") == "flops" else sparsity
            acc_flops.append((flops_red, metrics["acc@1"]))

        rep.log_curve(ds_name, mdl, method, "acc_vs_sparsity", acc_sparsity)
        rep.log_curve(ds_name, mdl, method, "acc_vs_flops", acc_flops)

    rep.save_tables(); rep.build_html(title="Pruning Curves Report")

if __name__ == "__main__":
    main()
