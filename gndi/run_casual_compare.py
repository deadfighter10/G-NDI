# gndi/run_causal_compare_cv.py
from __future__ import annotations
import argparse, os, sys, json, random
from typing import Dict, Any, List, Tuple
from collections import OrderedDict

# --- allow running as script or module ---
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

# Optional SciPy; fall back gracefully if missing
try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
except Exception:
    pearsonr = spearmanr = kendalltau = None

# --- repo imports ---
from gndi.data import build_dataloaders
from gndi.models_cv import build_cv_model
from gndi.eval.casual_gt import measure_true_effect_units  # <- causal, not casual
from gndi.scoring.gndi import compute_gndi_units
from gndi.scoring.magnitude import compute_magnitude_units
from gndi.scoring.snip import compute_snip_units
from gndi.scoring.grasp import compute_grasp_units
from gndi.scoring.synflow import compute_synflow_units
from gndi.scoring.hrank import compute_hrank_units

# ---------------- utils ----------------
def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _qname_map(model: nn.Module) -> dict[nn.Module, str]:
    return {m: q for q, m in model.named_modules()}

def _module_by_qname(model: nn.Module) -> dict[str, nn.Module]:
    return {q: m for q, m in model.named_modules()}

def remap_units_to_model(units, src_model, dst_model):
    src_names = _qname_map(src_model)
    dst_by_name = _module_by_qname(dst_model)
    remapped = []
    lost = 0
    for u in units:
        qn = src_names.get(u["module"])
        if qn is None or qn not in dst_by_name:
            lost += 1
            continue
        u2 = dict(u)
        u2["module"] = dst_by_name[qn]
        remapped.append(u2)
    if lost:
        print(f"[warn] remap_units_to_model: lost {lost}/{len(units)} units during mapping.")
    return remapped

def _seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _is_const(x: np.ndarray) -> bool:
    return (x.size < 2) or np.allclose(x, x[0])

def _corrs(pred: np.ndarray, truth: np.ndarray) -> Dict[str,float]:
    """Assumes both arrays are finite and non-constant."""
    if pearsonr: pr = float(pearsonr(pred, truth)[0])
    else:        pr = float(np.corrcoef(pred, truth)[0,1])

    if spearmanr: sr = float(spearmanr(pred, truth)[0])
    else:
        rp = np.argsort(np.argsort(pred)); rt = np.argsort(np.argsort(truth))
        sr = float(np.corrcoef(rp, rt)[0,1])

    if kendalltau: kt = float(kendalltau(pred, truth)[0])
    else:          kt = float("nan")
    return {"pearson": pr, "spearman": sr, "kendall": kt}

def _bootstrap_ci_spearman(pred: np.ndarray, truth: np.ndarray, rounds=1000, ci=0.95) -> Tuple[float,float,float]:
    n = pred.shape[0]
    if n == 0 or _is_const(pred) or _is_const(truth):
        return float("nan"), float("nan"), float("nan")
    # For very small n, CI is unreliable; return point estimate only.
    if n < 10:
        sr = _corrs(pred, truth)["spearman"]
        return float(sr), float("nan"), float("nan")
    vals = []
    for _ in range(rounds):
        idx = np.random.randint(0, n, size=n)
        p = pred[idx]; t = truth[idx]
        if spearmanr:
            vals.append(float(spearmanr(p, t)[0]))
        else:
            rp = np.argsort(np.argsort(p)); rt = np.argsort(np.argsort(t))
            vals.append(float(np.corrcoef(rp, rt)[0,1]))
    vals = np.array(vals, dtype=float)
    m = float(np.nanmean(vals))
    lo = float(np.nanpercentile(vals, (1-ci)/2*100))
    hi = float(np.nanpercentile(vals, (1+ci)/2*100))
    return m, lo, hi

def replace_maxpool_with_avgpool(model):
    """Replace MaxPool2d with AvgPool2d (optional; only if --avgpool-jvp is set)."""
    for name, module in model.named_children():
        if isinstance(module, nn.MaxPool2d):
            setattr(model, name, nn.AvgPool2d(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding
            ))
        else:
            replace_maxpool_with_avgpool(module)

# ------------- CV unit enumeration -------------
def enumerate_cv_units(model: nn.Module,
                       include_fc: bool = False,
                       residual_branch_only: bool = True) -> List[Dict[str, Any]]:
    """
    Build unit dicts for Conv2d output channels (and optionally final FC neurons).
    For ResNet-like models, exclude downsample/shortcut convs (skip path) when residual_branch_only=True.
    """
    units: List[Dict[str, Any]] = []
    name_map: Dict[nn.Module, str] = {}
    for qn, m in model.named_modules(): name_map[m] = qn

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            qn = name_map.get(m, "")
            if residual_branch_only and ("downsample" in qn or "shortcut" in qn):
                continue
            C = m.out_channels
            for idx in range(C):
                units.append({"id": f"{qn}.ch{idx}", "module": m, "type": "conv_channel", "index": idx})

    if include_fc:
        last_fc = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_fc = m
        if last_fc is not None:
            C = last_fc.out_features
            qn = name_map[last_fc]
            for idx in range(C):
                units.append({"id": f"{qn}.n{idx}", "module": last_fc, "type": "fc_neuron", "index": idx})
    return units

def _method_map():
    return OrderedDict([
        ("gndi",      compute_gndi_units),
        ("magnitude", compute_magnitude_units),
        ("snip",      compute_snip_units),
        ("grasp",     compute_grasp_units),
        ("synflow",   compute_synflow_units),
        ("hrank",     compute_hrank_units),
    ])

def main():
    ap = argparse.ArgumentParser("Compare causal validity (CV only)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["gndi","magnitude","snip","grasp","synflow","hrank"])
    ap.add_argument("--units", type=int, default=300, help="sample this many units")
    ap.add_argument("--metric", choices=["output_delta","loss_delta"], default="output_delta",
                    help="Definition of GT causal effect (ensure scorers align conceptually).")
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--include-fc", action="store_true")
    ap.add_argument("--include-skip", action="store_true", help="include skip-path convs (downsample) as units")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--avgpool_jvp", action="store_true",
                    help="If set, replace MaxPool with AvgPool for GNDI JVP stability (changes semantics slightly).")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    out_dir = args.out or os.path.join(_get(cfg,"paths.out_dir","./runs"), "causal_compare_cv")
    os.makedirs(out_dir, exist_ok=True)

    _seed_everything(args.seed)

    # Build data + model (CV only)
    loaders = build_dataloaders(cfg)
    model = build_cv_model(cfg)
    dev = _device(); model.to(dev); model.eval()

    # Enumerate units
    all_units = enumerate_cv_units(
        model,
        include_fc=args.include_fc,
        residual_branch_only=not args.include_skip
    )
    if len(all_units) == 0:
        raise RuntimeError("No units found. Check model and unit enumeration.")
    # Sample units for efficiency (stable by seed)
    if args.units < len(all_units):
        idx = np.random.RandomState(args.seed).choice(len(all_units), size=args.units, replace=False)
        units_eval = [all_units[i] for i in idx]
    else:
        units_eval = all_units

    # Ground-truth ablation (on val/test)
    gt_loader = loaders.get("val") or loaders.get("test") or loaders["train"]
    print(f"[GT] Measuring true ablation for {len(units_eval)} units (metric={args.metric}) …")
    gt = measure_true_effect_units(
        model, units_eval, batch_iter=gt_loader, metric=args.metric, p=args.p
    )  # dict[unit_id] -> float

    # First pass: compute predictions for each method on the *same* unit set
    METHOD_KWARGS = {
        "gndi":      {"amp","max_batches","p_norm","baseline","alpha"},
        "magnitude": {"amp","max_batches","norm"},
        "snip":      {"amp","max_batches"},
        "grasp":     {"amp","max_batches"},
        "synflow":   {"amp","at_init"},
        "hrank":     {"amp","max_batches"},
    }
    score_cfg = _get(cfg, "score", {})
    amp = bool(_get(cfg, "amp", True))

    pred_by_method: Dict[str, Dict[str, float]] = {}
    units_for_method: Dict[str, List[str]] = {}
    for name, fn in _method_map().items():
        if name not in args.methods:
            continue
        kwargs = dict(score_cfg.get(name, {}))
        if args.max_batches is not None:
            kwargs["max_batches"] = args.max_batches
        kwargs.setdefault("amp", amp)
        allowed = METHOD_KWARGS.get(name, set())
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        print(f"[{name}] scoring kwargs={kwargs}")
        if name == "gndi":
            import copy
            model_g = copy.deepcopy(model)
            if args.avgpool_jvp:
                replace_maxpool_with_avgpool(model_g)
            units_g = remap_units_to_model(units_eval, model, model_g)
            preds_dict = fn(model_g, loaders["train"], units_g, **kwargs)
        else:
            preds_dict = fn(model, loaders["train"], units_eval, **kwargs)

        pred_by_method[name] = preds_dict
        units_for_method[name] = list(preds_dict.keys())

    # Build a *common* set of unit ids where all methods AND gt are finite
    all_methods = list(pred_by_method.keys())
    id_sets = [set(units_for_method[m]) for m in all_methods]
    common_ids = set(gt.keys())
    for s in id_sets:
        common_ids &= s
    if not common_ids:
        raise RuntimeError("No common unit IDs across methods and GT.")

    # Filter for finiteness across ALL methods and GT
    finite_ids = []
    for uid in common_ids:
        ok = np.isfinite(gt[uid])
        if not ok:
            continue
        for m in all_methods:
            v = pred_by_method[m].get(uid, np.nan)
            if not np.isfinite(v):
                ok = False
                break
        if ok:
            finite_ids.append(uid)

    n_common = len(finite_ids)
    if n_common == 0:
        raise RuntimeError("After finiteness filtering, no units remain for comparison.")
    if n_common < 30:
        print(f"[warn] Only n={n_common} common finite units across all methods; correlations may be unstable.")

    # Now compute correlations on the EXACT same (finite) unit set for each method
    results = {}
    rows = []
    rounds = int(_get(cfg, "eval.bootstrap.spearman.rounds", 1000))
    ci = float(_get(cfg, "eval.bootstrap.spearman.ci", 0.95))

    truth_vec = np.array([gt[uid] for uid in finite_ids], dtype=np.float64)

    for name in all_methods:
        pred_vec = np.array([pred_by_method[name][uid] for uid in finite_ids], dtype=np.float64)

        if _is_const(pred_vec) or _is_const(truth_vec):
            print(f"[warn] {name}: constant vectors on common set (n={n_common}).")
            corr = {"pearson": float("nan"), "spearman": float("nan"), "kendall": float("nan")}
            mean_rho = lo = hi = float("nan")
        else:
            corr = _corrs(pred_vec, truth_vec)
            mean_rho, lo, hi = _bootstrap_ci_spearman(pred_vec, truth_vec, rounds=rounds, ci=ci)

        corr["spearman_ci"] = [lo, hi]
        corr["n_common"] = int(n_common)
        results[name] = corr
        rows.append([name, corr["pearson"], corr["spearman"], lo, hi, corr["kendall"], corr["n_common"]])

    # Write JSON/CSV
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "causal_compare_cv.json"), "w") as f:
        json.dump({"config": args.config, "metric": args.metric, "results": results}, f, indent=2)

    import csv
    with open(os.path.join(out_dir, "causal_compare_cv.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","pearson","spearman","spearman_ci_lo","spearman_ci_hi","kendall","n_common"])
        for r in rows: w.writerow(r)

    # Plot bar (Spearman + 95% CI) with robust limits and NaN handling
    try:
        import matplotlib.pyplot as plt
        methods = [r[0] for r in rows]
        spearmans = np.array([r[2] for r in rows], dtype=float)
        lo = np.array([r[3] for r in rows], dtype=float)
        hi = np.array([r[4] for r in rows], dtype=float)

        # yerr for finite values only
        yerr = np.vstack([
            np.where(np.isfinite(spearmans) & np.isfinite(lo), spearmans - lo, np.nan),
            np.where(np.isfinite(spearmans) & np.isfinite(hi), hi - spearmans, np.nan),
        ])

        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(9,4.6))

        # Bars: finite in solid, NaN as hatched
        bars = []
        for i, rho in enumerate(spearmans):
            if np.isfinite(rho):
                bars.append(ax.bar([x[i]], [rho])[0])
            else:
                b = ax.bar([x[i]], [0.0], hatch="//", alpha=0.3)[0]
                bars.append(b)

        ax.errorbar(x, np.where(np.isfinite(spearmans), spearmans, 0.0),
                    yerr=yerr, fmt="none", ecolor="k", capsize=4, linewidth=1)

        ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in methods])

        # Robust y-limits
        ymin = np.nanmin(np.minimum(spearmans, lo))
        ymax = np.nanmax(np.maximum(spearmans, hi))
        if not np.isfinite(ymin): ymin = -1.0
        if not np.isfinite(ymax): ymax = 1.0
        pad = 0.05
        ax.set_ylim(min(-1.0, ymin - pad), max(1.0, ymax + pad))

        ds = f"{_get(cfg,'dataset.name','?')} / {_get(cfg,'model.arch','?')}"
        ax.set_title(f"Causal Validity on CV (Spearman ρ with 95% CI)\n{ds}")
        ax.set_ylabel("Spearman ρ")

        # Legend entry for N/A
        if np.any(~np.isfinite(spearmans)):
            from matplotlib.patches import Patch
            ax.legend([Patch(hatch='//', alpha=0.3)], ["N/A (undefined ρ)"], loc="lower left", frameon=False)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "causal_compare_cv_bar.png"), dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] Could not plot bar chart: {e}")

    print(f"[done] Results saved under: {out_dir}")

if __name__ == "__main__":
    main()
