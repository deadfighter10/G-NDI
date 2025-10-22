# reporter.py
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from collections import defaultdict
import math

def _fmt(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or not math.isfinite(x))):
        return "—"
    return f"{x:.3f}"

def generate_pdf(results, out_path="overnight_report.pdf"):
    """
    Builds a compact PDF:
      - One section per (dataset, model)
      - Single comparative table per section: rows=methods, cols=[Pearson, Spearman, Kendall]
      - No wasteful table-per-page behavior
    """
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    # Group rows by (dataset, model)
    groups = defaultdict(list)
    for row in results:
        key = (row["dataset"], row["model"])
        groups[key].append(row)

    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    flow = []
    flow.append(Paragraph("G-NDI Overnight Report", h1))
    flow.append(Paragraph("Comparative correlations vs. causal ablation ground truth", body))
    flow.append(Spacer(1, 12))

    for (ds, md) in sorted(groups.keys()):
        flow.append(Paragraph(f"{ds} — {md}", h2))
        data = [["Method", "Pearson r", "Spearman ρ", "Kendall τ"]]

        # Sort methods: GNDI-* first, then others
        rows = groups[(ds, md)]
        def _order(m):
            if m["method"].startswith("GNDI-"): return (0, m["method"])
            return (1, m["method"])
        for r in sorted(rows, key=lambda x: _order(x)):
            data.append([
                r["method"],
                _fmt(r.get("pearson")),
                _fmt(r.get("spearman")),
                _fmt(r.get("kendall")),
            ])

        t = Table(data, colWidths=[200, 90, 90, 90])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10),
            ("BOTTOMPADDING", (0,0), (-1,0), 8),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#9ca3af")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor("#f3f4f6")]),
        ]))
        flow.append(t)
        flow.append(Spacer(1, 18))

    doc.build(flow)
    print(f"[✓] PDF written to {out_path}")
