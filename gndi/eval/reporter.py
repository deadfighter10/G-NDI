# gndi/eval/reporter.py
# -*- coding: utf-8 -*-
"""
Reporting utilities: JSON/CSV dumps, plots, and a compact HTML (and optional PDF) report.

Exports:
- Reporter(out_dir)
    .log_causal_corr(dataset, model, method, stats_dict)
    .log_curve(dataset, model, method, curve_type, points)   # curve_type: 'acc_vs_sparsity'|'acc_vs_flops'
    .log_runtime(dataset, model, method, seconds)
    .save_tables()   # writes CSV/JSON unified tables
    .plot_correlations_bar(with_ci=True)
    .plot_accuracy_vs_sparsity()
    .plot_accuracy_vs_flops()
    .build_html(title="Overnight Report") -> path
    .build_pdf(html_path=None) -> pdf_path (if weasyprint/reportlab available)

Design goals:
- Single consolidated comparison tables across methods (fixes "new table per page" issue).
- Plots saved as PNG; HTML references them in a clean, one-page layout.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import csv
import math
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

class Reporter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        _ensure_dir(out_dir)

        # Tables keyed by (dataset, model)
        self.corr_table = []    # rows: dataset,model,method,pearson,spearman,kendall,ci_low,ci_high
        self.curves_sparsity = defaultdict(list)  # key=(dataset,model,method) -> [(sparsity, acc), ...]
        self.curves_flops = defaultdict(list)     # key=(dataset,model,method) -> [(flops_red, acc), ...]
        self.runtime_table = [] # dataset,model,method,seconds

    # ----------------------------- Logging ------------------------------

    def log_causal_corr(self, dataset: str, model: str, method: str, stats: Dict[str, float]):
        row = {
            "dataset": dataset, "model": model, "method": method,
            "pearson": float(stats.get("pearson", 0.0)),
            "spearman": float(stats.get("spearman", 0.0)),
            "kendall": float(stats.get("kendall", 0.0)),
            "ci_low": float(stats.get("ci_low", stats.get("low", 0.0))),
            "ci_high": float(stats.get("ci_high", stats.get("high", 0.0))),
        }
        self.corr_table.append(row)

    def log_curve(self, dataset: str, model: str, method: str, curve_type: str, points: List[Tuple[float, float]]):
        key = (dataset, model, method)
        if curve_type == "acc_vs_sparsity":
            self.curves_sparsity[key].extend(list(points))
        elif curve_type == "acc_vs_flops":
            self.curves_flops[key].extend(list(points))
        else:
            raise ValueError("curve_type must be 'acc_vs_sparsity' or 'acc_vs_flops'")

    def log_runtime(self, dataset: str, model: str, method: str, seconds: float):
        self.runtime_table.append({
            "dataset": dataset, "model": model, "method": method, "seconds": float(seconds)
        })

    # ------------------------------ Tables ------------------------------

    def _write_csv(self, name: str, rows: List[Dict[str, Any]]):
        if not rows:
            return None
        path = os.path.join(self.out_dir, f"{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def _write_json(self, name: str, obj: Any):
        path = os.path.join(self.out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        return path

    def save_tables(self):
        # Consolidated comparisons in single tables
        corr_csv = self._write_csv("correlations", self.corr_table)
        runtime_csv = self._write_csv("runtime", self.runtime_table)
        # Curves to JSON (dict with tuple keys flattened)
        curves_obj = {
            "acc_vs_sparsity": [
                {"dataset": d, "model": m, "method": meth, "points": sorted(list(set(pts)))}
                for (d, m, meth), pts in self.curves_sparsity.items()
            ],
            "acc_vs_flops": [
                {"dataset": d, "model": m, "method": meth, "points": sorted(list(set(pts)))}
                for (d, m, meth), pts in self.curves_flops.items()
            ],
        }
        curves_json = self._write_json("curves", curves_obj)
        return {"correlations_csv": corr_csv, "runtime_csv": runtime_csv, "curves_json": curves_json}

    # ------------------------------- Plots ------------------------------

    def plot_correlations_bar(self, title: str = "Causal Validity (Spearman ρ with 95% CI)"):
        if not self.corr_table:
            return None
        # Group by (dataset, model); one figure per pair; bars are methods
        by_pair = defaultdict(list)
        for r in self.corr_table:
            key = (r["dataset"], r["model"])
            by_pair[key].append(r)

        paths = []
        for (dataset, model), rows in by_pair.items():
            rows = sorted(rows, key=lambda r: r["method"])
            methods = [r["method"].upper() for r in rows]
            vals = [r["spearman"] for r in rows]
            yerr = [max(r["spearman"] - r["ci_low"], r["ci_high"] - r["spearman"], 0.0) for r in rows]

            plt.figure(figsize=(8, 4))
            plt.title(f"{title}\n{dataset} / {model}")
            plt.bar(methods, vals, yerr=yerr, capsize=4)
            plt.ylim(0.0, 1.0)
            plt.ylabel("Spearman ρ")
            plt.tight_layout()
            p = os.path.join(self.out_dir, f"corr_{dataset}_{model}.png")
            plt.savefig(p, dpi=160)
            plt.close()
            paths.append(p)
        return paths

    def plot_accuracy_vs_sparsity(self, title: str = "Accuracy vs Sparsity"):
        if not self.curves_sparsity:
            return None
        by_pair = defaultdict(list)
        for key, pts in self.curves_sparsity.items():
            by_pair[key[:2]].append((key[2], sorted(list(set(pts)))))

        paths = []
        for (dataset, model), method_pts in by_pair.items():
            plt.figure(figsize=(8, 4))
            plt.title(f"{title}\n{dataset} / {model}")
            for method, pts in sorted(method_pts, key=lambda t: t[0]):
                xs, ys = zip(*pts)
                plt.plot(xs, ys, marker="o", label=method.upper())
            plt.xlabel("Sparsity (fraction pruned)")
            plt.ylabel("Accuracy (%)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            p = os.path.join(self.out_dir, f"acc_sparsity_{dataset}_{model}.png")
            plt.savefig(p, dpi=160)
            plt.close()
            paths.append(p)
        return paths

    def plot_accuracy_vs_flops(self, title: str = "Accuracy vs FLOPs Reduction"):
        if not self.curves_flops:
            return None
        by_pair = defaultdict(list)
        for key, pts in self.curves_flops.items():
            by_pair[key[:2]].append((key[2], sorted(list(set(pts)))))

        paths = []
        for (dataset, model), method_pts in by_pair.items():
            plt.figure(figsize=(8, 4))
            plt.title(f"{title}\n{dataset} / {model}")
            for method, pts in sorted(method_pts, key=lambda t: t[0]):
                xs, ys = zip(*pts)
                plt.plot(xs, ys, marker="o", label=method.upper())
            plt.xlabel("FLOPs Reduction (fraction)")
            plt.ylabel("Accuracy (%)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            p = os.path.join(self.out_dir, f"acc_flops_{dataset}_{model}.png")
            plt.savefig(p, dpi=160)
            plt.close()
            paths.append(p)
        return paths

    # ----------------------------- HTML/PDF -----------------------------

    def build_html(self, title: str = "Overnight Report") -> str:
        self.save_tables()
        img_corr = self.plot_correlations_bar() or []
        img_spar = self.plot_accuracy_vs_sparsity() or []
        img_flop = self.plot_accuracy_vs_flops() or []

        html_path = os.path.join(self.out_dir, "overnight_report.html")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Simple one-page HTML with linked figures and single comparison tables
        def _table_to_html(rows: List[Dict[str, Any]], title: str):
            if not rows: return ""
            cols = list(rows[0].keys())
            th = "".join(f"<th>{c}</th>" for c in cols)
            trs = []
            for r in rows:
                tds = "".join(f"<td>{r.get(c,'')}</td>" for c in cols)
                trs.append(f"<tr>{tds}</tr>")
            return f"<h3>{title}</h3><table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"

        style = """
        <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;background:#fff;color:#111;}
        h1{margin-bottom:6px} h2{margin-top:18px}
        table{border-collapse:collapse;margin:12px 0;width:100%}
        th,td{border:1px solid #ddd;padding:6px 8px;font-size:14px}
        th{background:#f7f7f7;text-align:left}
        img{max-width:100%;height:auto;border:1px solid #eee;margin:8px 0}
        .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
        .small{font-size:12px;color:#666}
        </style>
        """

        figs = "".join(f'<img src="{os.path.basename(p)}" />' for p in (img_corr + img_spar + img_flop))

        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>{style}</head>
<body>
<h1>{title}</h1>
<div class="small">Generated: {now}</div>

<h2>Key Figures</h2>
<div class="grid">{figs}</div>

<h2>Comparison Tables</h2>
{_table_to_html(self.corr_table, "Causal Validity — Correlations & 95% CI")}
{_table_to_html(self.runtime_table, "Runtime — Score Computation Cost (sec)")}

<p class="small">Notes: AUC/accuracy-at-FLOPs can be computed downstream from curves.json.
This report intentionally consolidates comparisons into unified tables (no per-page duplication).</p>
</body></html>
"""
        with open(html_path, "w") as f:
            f.write(html)
        return html_path

    def build_pdf(self, html_path: Optional[str] = None) -> Optional[str]:
        """
        Best-effort HTML->PDF using weasyprint or reportlab if available.
        Returns path or None if no backend is installed.
        """
        if html_path is None:
            html_path = os.path.join(self.out_dir, "overnight_report.html")
            if not os.path.isfile(html_path):
                html_path = self.build_html()

        pdf_path = os.path.join(self.out_dir, "overnight_report.pdf")

        # Try weasyprint
        try:
            from weasyprint import HTML
            HTML(filename=html_path).write_pdf(pdf_path)
            return pdf_path
        except Exception:
            pass

        # Try reportlab (very simple print of HTML text fallback)
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(pdf_path, pagesize=letter)
            width, height = letter
            c.setFont("Helvetica", 10)
            with open(html_path, "r") as f:
                txt = f.read()
            # Naive text dump (not pretty, but a fallback)
            y = height - 36
            for line in txt.splitlines():
                c.drawString(36, y, line[:120])
                y -= 12
                if y < 36:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - 36
            c.save()
            return pdf_path
        except Exception:
            return None
