"""Render publication-ready campaign tables from the local result ledgers.

The campaign ledger is intentionally the single source of truth for our runs:
``report/jetfuel_campaign_results.csv``.  Literature values live in the small,
separately-auditable ``report/paper_baselines_gluon30.csv`` because they do not
have local run artifacts.  This script never edits either input.

Usage:
    python src/analysis/generate_campaign_tables.py
    python src/analysis/generate_campaign_tables.py --check

The default command writes three generated artifacts under ``report/generated``:

* ``jetfuel_campaign_results.tex``: every ledger result, including incomplete
  evaluations and non-qualified diagnostic arms;
* ``gluon30_paper_comparison.tex``: corrected, paper-comparable gluon-30 rows
  beside MPGAN and PC-JeDi; and
* ``jetnet30_class_comparison.tex``: paper-facing gluon, light-quark, and top
  JetNet-30 rows, including the class-specific GQT FPNDs; and
* ``jetnet30_method_table.tex``: a standalone LaTeX document with one section
  per jet class and a paper-vs-JetFUEL divider in each section; and
* ``campaign_ablation_trace.md``: controlled deltas and cautious conclusions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "report" / "jetfuel_campaign_results.csv"
DEFAULT_BASELINES = ROOT / "report" / "paper_baselines_gluon30.csv"
DEFAULT_OUT_DIR = ROOT / "report" / "generated"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    return float(raw) if raw else None


def tex(value: str) -> str:
    """Escape the small subset of TeX special characters appearing in labels."""
    return (value.replace("\\", r"\textbackslash{}")
                 .replace("_", r"\_")
                 .replace("%", r"\%")
                 .replace("&", r"\&")
                 .replace("#", r"\#"))


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return r"\textemdash{}"
    magnitude = abs(value)
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e4):
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"
    return f"{value:.{digits}f}"


def fmt_steps(value: float | None) -> str:
    if value is None:
        return r"\textemdash{}"
    if value % 1000 == 0:
        return f"{int(value // 1000)}K"
    return f"{int(value):,}"


def fmt_params(value: float | None) -> str:
    if value is None:
        return r"\textemdash{}"
    return f"{value / 1e3:.0f}K"


def physicality(row: dict[str, str]) -> str:
    invalid = number(row, "invalid_fraction")
    neg = number(row, "negative_energy_fraction")
    space = number(row, "spacelike_fraction")
    if invalid is None and neg is None and space is None:
        return r"\textemdash{}"
    if invalid == neg == space == 0:
        return "0 invalid"
    bits = []
    if invalid is not None:
        bits.append(f"{100 * invalid:.3g}\\% invalid")
    if neg not in (None, 0):
        bits.append(f"{100 * neg:.3g}\\% neg.-E")
    if space not in (None, 0):
        bits.append(f"{100 * space:.3g}\\% spacelike")
    return "; ".join(bits)


def design(row: dict[str, str]) -> str:
    name = row["run"]
    if name in {"A", "B", "C", "D", "E", "F"}:
        return {"A": "Euclidean; no refs", "B": "RFM; no refs",
                "C": "Euclidean; plain refs", "D": "RFM; plain refs",
                "E": "RFM; scalar init + ICP; no ref readout",
                "F": "RFM; scalar init + ICP; plain refs"}[name]
    if name == "G-200K":
        return "Raw refs; evolving geometry"
    if name == "H-200K":
        return "Normalized tangent refs; evolving geometry"
    if name == "I-200K":
        return "Raw refs; fixed physical geometry"
    if name == "J-200K":
        return "Normalized tangent refs; fixed physical geometry"
    if name == "H-BlockCond-200K":
        return "H + condition/time in every block"
    if name in {"H-500K", "G-500K"}:
        return f"{name[0]} recipe extended"
    if name == "G-331K":
        return "H recipe; gluon specialist"
    if name == "T-331K":
        return "H recipe; top specialist"
    if name.startswith("GQT-994K"):
        return "H recipe; joint g/q/t"
    return ""


def campaign_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Coalesce GQT's class-specific FPNDs into its one shared evaluation row."""
    normal = [row for row in rows if not row["run"].startswith("GQT-994K")]
    gqt = [row for row in rows if row["run"].startswith("GQT-994K")]
    if gqt:
        shared = dict(gqt[0])
        fpnd = r"\shortstack{" + r"\\ ".join(
            f"{row['jet_type']}={fmt(number(row, 'fpnd_value'))}"
            for row in gqt
        ) + "}"
        shared["run"] = "GQT-994K (g/q/t)"
        shared["fpnd_display"] = fpnd
        shared["fpd_display"] = r"\textemdash{}$^{\ddagger}$"
        normal.append(shared)
    return normal


def campaign_tex(rows: list[dict[str, str]]) -> str:
    out = [
        "% GENERATED by src/analysis/generate_campaign_tables.py; do not edit.",
        "% Requires: \\usepackage{booktabs,longtable,pdflscape}",
        r"\begin{landscape}", r"\scriptsize", r"\setlength{\tabcolsep}{1pt}",
        r"\begin{longtable}{@{}lp{3.4cm}rrrrrrrrp{1.7cm}@{}}",
        r"\caption{All locally recorded JetNet-30 campaign results. Lower is better for FPND, FPD, W1M, mean W1P, and MMD; higher is better for coverage. Dashes mean no completed measurement. $^\dagger$ marks historical, pre-canonical-FPND values and must not be compared to the literature. $^\ddagger$ marks the GQT full-sample FPD, which is quarantined (raw value $646{,}050$) pending a tail diagnostic.}\label{tab:campaign-results}\\",
        r"\toprule",
        r"Run & Design & Steps & Params. & FPND & FPD & W1M & W1P & Cov. & MMD & Physicality \\",
        r"\midrule", r"\endfirsthead",
        r"\toprule",
        r"Run & Design & Steps & Params. & FPND & FPD & W1M & W1P & Cov. & MMD & Physicality \\",
        r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot",
    ]
    for row in campaign_rows(rows):
        historical = row["run"] in {"A", "B", "C", "D", "E", "F"}
        fpnd = row.get("fpnd_display") or fmt(number(row, "fpnd_g"))
        if historical and fpnd != r"\textemdash{}":
            fpnd += r"$^\dagger$"
        fpd = row.get("fpd_display") or fmt(number(row, "fpd"), 4)
        values = [
            tex(row["run"]), tex(design(row)), fmt_steps(number(row, "steps")),
            fmt_params(number(row, "parameters")), fpnd, fpd,
            fmt(number(row, "w1m")), fmt(number(row, "mean_w1p")),
            fmt(number(row, "coverage"), 3), fmt(number(row, "mmd")),
            physicality(row),
        ]
        out.append(" & ".join(values) + r" \\")
    out.extend([r"\end{longtable}", r"\end{landscape}", ""])
    return "\n".join(out)


def comparable_ours(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Corrected, qualified, gluon-only rows appropriate for a g30 literature table."""
    return [row for row in rows
            if row["qualified"].lower() == "true"
            and row["jet_type"] == "g"
            and row["run"] not in {"A", "B", "C", "D", "E", "F"}]


def paper_tex(ours: list[dict[str, str]], baselines: list[dict[str, str]]) -> str:
    baselines = [row for row in baselines if row["jet_type"] == "g"]
    out = [
        "% GENERATED by src/analysis/generate_campaign_tables.py; do not edit.",
        "% Requires: \\usepackage{booktabs,threeparttable,pdflscape}",
        r"\begin{landscape}", r"\begin{table}[p]", r"\centering", r"\small", r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{threeparttable}",
        r"\caption{Gluon-30 comparison. Values for MPGAN and PC-JeDi are quoted from their published gluon-30 tables; FPD was not reported there. All JetFUEL rows use the corrected canonical FPND input.}\label{tab:gluon30-comparison}",
        r"\begin{tabular}{@{}lp{4.3cm}rrrrrrp{1.8cm}@{}}", r"\toprule",
        r"Method & Protocol / variant & FPND-g & FPD & W1M & Mean W1P & Cov. & MMD & Physicality \\", r"\midrule",
    ]
    for row in baselines:
        marker = "" if row["comparable"].lower() == "true" else r"$^\ast$"
        protocol = f"{row['training']}; {row['conditioning']}"
        values = [tex(row["method"]) + marker, tex(protocol),
                  fmt(number(row, "fpnd_g")), fmt(number(row, "fpd")),
                  fmt(number(row, "w1m")), fmt(number(row, "mean_w1p")),
                  fmt(number(row, "coverage"), 3), fmt(number(row, "mmd")),
                  tex(row["physicality"]) if row["physicality"] else r"\textemdash{}"]
        out.append(" & ".join(values) + r" \\")
    out.append(r"\midrule")
    for row in ours:
        is_joint = row["run"].startswith("GQT")
        values = [r"JetFUEL " + tex(row["run"]) + (r"$^\ddagger$" if is_joint else ""),
                  "joint g/q/t; g subset FPND" if is_joint else tex(design(row)),
                  fmt(number(row, "fpnd_g")),
                  r"\textemdash{}" if is_joint else fmt(number(row, "fpd")),
                  r"\textemdash{}" if is_joint else fmt(number(row, "w1m")),
                  r"\textemdash{}" if is_joint else fmt(number(row, "mean_w1p")),
                  r"\textemdash{}" if is_joint else fmt(number(row, "coverage"), 3),
                  r"\textemdash{}" if is_joint else fmt(number(row, "mmd")),
                  physicality(row)]
        out.append(" & ".join(values) + r" \\")
    out.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\footnotesize",
        r"\item $^\ast$ EPiC-FM and EPiC-JeDi report top-30/top-150 jets, not gluon-30; their cells are intentionally blank and they are not a numerical comparison.",
        r"\item $^\ddagger$ GQT-994K is joint g/q/t training. Its FPND-g is class-filtered and comparable; its other saved metrics are joint-sample aggregates, so they are intentionally blank here. Its raw full-sample FPD is quarantined.",
        r"\item PC-JeDi and MPGAN condition on jet attributes taken from the test set, while JetFUEL samples those attributes with Stage~1; this is an important remaining protocol difference.",
        r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", r"\end{landscape}", "",
    ])
    return "\n".join(out)


def class_comparison_tex(rows: list[dict[str, str]], baselines: list[dict[str, str]]) -> str:
    """Render all available JetNet-30 classes without mislabelling GQT aggregates."""
    by_name = {row["run"]: row for row in rows}
    ours = [
        by_name["H-200K"], by_name["GQT-994K[g]"], by_name["GQT-994K[q]"],
        by_name["T-331K"], by_name["GQT-994K[t]"],
    ]
    order = {"g": 0, "q": 1, "t": 2}
    baselines = sorted(
        (row for row in baselines if row["comparable"].lower() == "true"),
        key=lambda row: order.get(row["jet_type"], 99),
    )
    ours = sorted(ours, key=lambda row: order[row["jet_type"]])
    out = [
        "% GENERATED by src/analysis/generate_campaign_tables.py; do not edit.",
        "% Requires: \\usepackage{booktabs,longtable,pdflscape}",
        r"\begin{landscape}", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{@{}llp{3.6cm}rrrrrp{1.6cm}@{}}",
        r"\caption{Class-resolved JetNet-30 comparison. PC-JeDi reports gluon and top but not light-quark jets; MPGAN reports all three. FPD is excluded because the cited papers do not report it.}\label{tab:jetnet30-class-comparison}\\",
        r"\toprule",
        r"Class & Method & Protocol / variant & FPND & W1M & Mean W1P & Cov. & MMD & Physicality \\",
        r"\midrule", r"\endfirsthead",
        r"\toprule",
        r"Class & Method & Protocol / variant & FPND & W1M & Mean W1P & Cov. & MMD & Physicality \\",
        r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot",
    ]
    for row in baselines:
        marker = "" if row["comparable"].lower() == "true" else r"$^\ast$"
        protocol = f"{row['training']}; {row['conditioning']}"
        values = [row["jet_type"], tex(row["method"]) + marker, tex(protocol),
                  fmt(number(row, "fpnd_g")), fmt(number(row, "w1m")),
                  fmt(number(row, "mean_w1p")), fmt(number(row, "coverage"), 3),
                  fmt(number(row, "mmd")),
                  tex(row["physicality"]) if row["physicality"] else r"\textemdash{}"]
        out.append(" & ".join(values) + r" \\")
    out.append(r"\midrule")
    for row in ours:
        is_joint = row["run"].startswith("GQT")
        fpnd = number(row, "fpnd_value") or number(row, "fpnd_g")
        method = r"JetFUEL " + tex(row["run"]) + (r"$^\ddagger$" if is_joint else "")
        protocol = "joint g/q/t; class-filtered FPND" if is_joint else tex(design(row))
        values = [row["jet_type"], method, protocol, fmt(fpnd),
                  r"\textemdash{}" if is_joint else fmt(number(row, "w1m")),
                  r"\textemdash{}" if is_joint else fmt(number(row, "mean_w1p")),
                  r"\textemdash{}" if is_joint else fmt(number(row, "coverage"), 3),
                  r"\textemdash{}" if is_joint else fmt(number(row, "mmd")),
                  physicality(row)]
        out.append(" & ".join(values) + r" \\")
    out.extend([
        r"\midrule",
        r"\multicolumn{9}{@{}p{18cm}@{}}{\footnotesize $^\ddagger$ GQT-994K saves class-filtered FPND but joint-sample W1/Cov/MMD; those aggregate cells are intentionally blank. Its raw full-sample FPD is quarantined. PC-JeDi and MPGAN condition on jet attributes from the test set, while JetFUEL samples them in Stage~1.} \\",
        r"\end{longtable}", r"\end{landscape}", "",
    ])
    return "\n".join(out)


def _ours_for_class(rows: list[dict[str, str]], jet_type: str) -> list[dict[str, str]]:
    """Return every completed paper-facing JetFUEL row for one JetNet-30 class."""
    if jet_type == "g":
        wanted = {"G-331K", "GQT-994K[g]"}
    elif jet_type == "q":
        wanted = {"GQT-994K[q]"}
    else:
        wanted = {"T-331K", "GQT-994K[t]"}
    selected = [row for row in rows if row["run"] in wanted]
    if jet_type == "g":
        # The dedicated 994k-step gluon specialist is present in the paper-eval
        # matrix, but its terminal artifacts have not been downloaded/recorded.
        # Keep it visible as an explicitly blank, pending-evaluation endpoint.
        selected.append({"run": "G-994K-pending"})
    elif jet_type == "q":
        # Q-331K exists locally, but its old terminal evaluation has the same
        # label-conditioning defect as the specialist re-evaluations in flight.
        # The Q-994K source checkpoint had not yet been produced.
        selected = [
            {"run": "Q-331K-invalid"},
            {"run": "Q-994K-pending"},
            *selected,
        ]
    else:
        # The local 994K top endpoint also predates the corrected evaluator.
        selected = [
            row for row in selected if row["run"] == "T-331K"
        ] + [{"run": "T-994K-invalid"}] + [
            row for row in selected if row["run"] == "GQT-994K[t]"
        ]
    return selected


def fmt_scaled(value: float | None, scale: float) -> str:
    """Format a metric after applying a column-level scientific scale."""
    return fmt(None if value is None else value * scale)


def _method_table_row(row: dict[str, str], ours: bool) -> str:
    """One row for the standalone class-by-class method table."""
    is_joint = row["run"].startswith("GQT") if ours else False
    invalid_old_eval = ours and row["run"] in {
        "Q-331K-invalid", "T-331K", "T-994K-invalid"
    }
    pending_eval = ours and row["run"] == "G-994K-pending"
    pending_training = ours and row["run"] == "Q-994K-pending"
    if ours:
        fpnd = number(row, "fpnd_value") or number(row, "fpnd_g")
        epoch_labels = {
            "G-331K": "G-666 epochs",
            "T-331K": "T-666 epochs",
            "GQT-994K[g]": "GQT-2000 epochs [g]",
            "GQT-994K[q]": "GQT-2000 epochs [q]",
            "GQT-994K[t]": "GQT-2000 epochs [t]",
            "G-994K-pending": "G-2000 epochs",
            "Q-331K-invalid": "Q-666 epochs",
            "Q-994K-pending": "Q-2000 epochs",
            "T-994K-invalid": "T-2000 epochs",
        }
        label = (r"JetFUEL " + tex(epoch_labels.get(row["run"], row["run"]))
                 + (r"$^\dagger$" if invalid_old_eval else r"$^\S$" if pending_eval
                    else r"$^\ast$" if pending_training
                    else r"$^\ddagger$" if is_joint else ""))
        if invalid_old_eval or pending_eval or pending_training:
            fpnd = None
            w1m = w1p = cov = mmd = r"\textemdash{}"
        else:
            w1m = fmt_scaled(number(row, "w1m"), 1e4)
            w1p = fmt_scaled(number(row, "mean_w1p"), 1e4)
            cov = fmt(number(row, "coverage"), 3)
            mmd = fmt_scaled(number(row, "mmd"), 1e2)
    else:
        fpnd = number(row, "fpnd_g")
        label = tex(row["method"])
        w1m = fmt_scaled(number(row, "w1m"), 1e4)
        w1p = fmt_scaled(number(row, "mean_w1p"), 1e4)
        cov = fmt(number(row, "coverage"), 3)
        mmd = fmt_scaled(number(row, "mmd"), 1e2)
    values = [label, fmt(fpnd), w1m, w1p, cov, mmd]
    return " & ".join(values) + r" \\"


def method_table_document_tex(rows: list[dict[str, str]], baselines: list[dict[str, str]]) -> str:
    """Standalone, screenshot-style class-by-class method comparison document."""
    class_names = {"g": "Gluon jets", "q": "Light-quark jets", "t": "Top jets"}
    out = [
        "% GENERATED by src/analysis/generate_campaign_tables.py; do not edit.",
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{booktabs,pdflscape,multirow}",
        r"\begin{document}", r"\begin{landscape}", r"\thispagestyle{empty}", r"\begin{center}", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.0}",
        r"\parbox{0.80\linewidth}{\textbf{Table 1:} JetNet-30 generative-model comparison. Lower is better for FPND, W1M, mean W1P, and MMD; higher is better for coverage.}",
        r"\medskip",
        r"\begin{tabular}{@{}clrrrrr@{}}",
        r"\toprule",
        r"Jet class & Method & FPND $\downarrow$ & W1M ($\times 10^{-4}$) $\downarrow$ & Mean W1P ($\times 10^{-4}$) $\downarrow$ & Cov. $\uparrow$ & MMD ($\times 10^{-2}$) $\downarrow$ \\",
        r"\midrule",
    ]
    for index, jet_type in enumerate(("g", "q", "t")):
        if index:
            out.append(r"\midrule")
        paper_rows = [row for row in baselines if row["jet_type"] == jet_type]
        ours = _ours_for_class(rows, jet_type)
        grouped = [(row, False) for row in paper_rows] + [(row, True) for row in ours]
        for row_index, (row, is_ours) in enumerate(grouped):
            if row_index == len(paper_rows):
                out.append(r"\cmidrule(l){2-7}")
            class_cell = (rf"\multirow{{{len(grouped)}}}{{*}}{{{class_names[jet_type]}}}"
                          if row_index == 0 else "")
            out.append(class_cell + " & " + _method_table_row(row, ours=is_ours))
    out.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\parbox{0.98\linewidth}{\footnotesize\medskip $^\dagger$ The Q-666, T-666, and T-2000 epoch terminal evaluations are invalid: the old evaluator applied a global gluon condition. Their checkpoints are valid and corrected re-evaluations are pending. $^\S$ The dedicated G-2000-epoch gluon run is listed in the paper-evaluation matrix, but no terminal evaluation has been recorded locally. $^\ast$ The Q-2000-epoch source checkpoint was not yet present when the pending-evaluation manifest was written. $^\ddagger$ GQT-2000-epoch training has class-filtered FPND; its displayed W1/Cov/MMD values are valid joint g/q/t aggregates. EPiC-FM and EPiC-JeDi are scope markers only: they evaluate top-30/top-150 jets, not JetNet gluon-30. PC-JeDi reports gluon and top, but not light-quark, JetNet-30 results. PC-JeDi and MPGAN condition on test-set jet attributes, while JetFUEL samples attributes in Stage~1.}",
        r"\end{center}", r"\end{landscape}", r"\end{document}", "",
    ])
    return "\n".join(out)


def pct_change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    return f"{100 * (after / before - 1):+.1f}%"


def trace_tex_metric(rows: dict[str, dict[str, str]], left: str, right: str, key: str) -> str:
    return pct_change(number(rows[left], key), number(rows[right], key))


def ablation_trace(rows: list[dict[str, str]]) -> str:
    by_name = {row["run"]: row for row in rows}
    lines = [
        "# Campaign ablation trace",
        "",
        "Generated from `report/jetfuel_campaign_results.csv` by "
        "`python src/analysis/generate_campaign_tables.py`. Percentages are relative changes in the second run versus the first; negative is better for all listed error metrics. This document deliberately separates controlled contrasts from confounded campaign changes.",
        "",
        "## Strongest controlled findings",
        "",
        f"- **Mass-shell geometry is the dominant early change.** With no reference readout, A → B reduces W1M from {fmt(number(by_name['A'], 'w1m'))} to {fmt(number(by_name['B'], 'w1m'))} ({trace_tex_metric(by_name, 'A', 'B', 'w1m')}) and removes the recorded {100 * number(by_name['A'], 'invalid_fraction'):.3g}% invalid / {100 * number(by_name['A'], 'spacelike_fraction'):.3g}% spacelike rate. With plain references, C → D reduces W1M {trace_tex_metric(by_name, 'C', 'D', 'w1m')} and removes the {100 * number(by_name['C'], 'spacelike_fraction'):.3g}% spacelike rate. Historical FPND values are excluded from both claims.",
        f"- **Evolving auxiliary geometry beats fixed physical geometry.** G → I (raw references) changes FPND {trace_tex_metric(by_name, 'G-200K', 'I-200K', 'fpnd_g')}, W1M {trace_tex_metric(by_name, 'G-200K', 'I-200K', 'w1m')}, mean W1P {trace_tex_metric(by_name, 'G-200K', 'I-200K', 'mean_w1p')}, coverage {trace_tex_metric(by_name, 'G-200K', 'I-200K', 'coverage')}, and MMD {trace_tex_metric(by_name, 'G-200K', 'I-200K', 'mmd')}. H → J repeats the direction with tangent references: FPND {trace_tex_metric(by_name, 'H-200K', 'J-200K', 'fpnd_g')}, W1M {trace_tex_metric(by_name, 'H-200K', 'J-200K', 'w1m')}, and mean W1P {trace_tex_metric(by_name, 'H-200K', 'J-200K', 'mean_w1p')}. This is the cleanest architectural verdict after the geometry choice.",
        f"- **Normalized tangent reference readout is a small, not decisive, improvement.** At 200K with evolving geometry, G → H improves FPND {trace_tex_metric(by_name, 'G-200K', 'H-200K', 'fpnd_g')}, FPD {trace_tex_metric(by_name, 'G-200K', 'H-200K', 'fpd')}, W1M {trace_tex_metric(by_name, 'G-200K', 'H-200K', 'w1m')}, and mean W1P {trace_tex_metric(by_name, 'G-200K', 'H-200K', 'mean_w1p')}; G retains the better MMD ({fmt(number(by_name['G-200K'], 'mmd'))} vs {fmt(number(by_name['H-200K'], 'mmd'))}). The W1M difference is smaller than the stored single-run uncertainties, so use H as the balanced default rather than claiming a robust win.",
        f"- **Injecting condition/time into every block is rejected.** H → H-BlockCond changes W1M {trace_tex_metric(by_name, 'H-200K', 'H-BlockCond-200K', 'w1m')}, mean W1P {trace_tex_metric(by_name, 'H-200K', 'H-BlockCond-200K', 'mean_w1p')}, coverage {trace_tex_metric(by_name, 'H-200K', 'H-BlockCond-200K', 'coverage')}, and FPND {trace_tex_metric(by_name, 'H-200K', 'H-BlockCond-200K', 'fpnd_g')}; no listed metric improves.",
        "",
        "## Optimization-length evidence",
        "",
        f"- **H at 500K is a Pareto trade-off, not an unqualified upgrade.** H-200K → H-500K improves FPND {trace_tex_metric(by_name, 'H-200K', 'H-500K', 'fpnd_g')} and FPD {trace_tex_metric(by_name, 'H-200K', 'H-500K', 'fpd')}, but worsens W1M {trace_tex_metric(by_name, 'H-200K', 'H-500K', 'w1m')}, mean W1P {trace_tex_metric(by_name, 'H-200K', 'H-500K', 'mean_w1p')}, coverage {trace_tex_metric(by_name, 'H-200K', 'H-500K', 'coverage')}, and MMD {trace_tex_metric(by_name, 'H-200K', 'H-500K', 'mmd')}.",
        f"- **G scales a little more smoothly.** G-200K → G-500K changes FPND {trace_tex_metric(by_name, 'G-200K', 'G-500K', 'fpnd_g')}, FPD {trace_tex_metric(by_name, 'G-200K', 'G-500K', 'fpd')}, and W1M {trace_tex_metric(by_name, 'G-200K', 'G-500K', 'w1m')}, with a modest MMD regression ({trace_tex_metric(by_name, 'G-200K', 'G-500K', 'mmd')}). Both are single training trajectories, so these are checkpoint-selection observations, not independent seed estimates.",
        "",
        "## Later campaign results: important, but not causal ablations",
        "",
        f"- **GQT-994K is the strongest observed multi-class result for g/q.** It records corrected FPND g/q/t = {fmt(number(by_name['GQT-994K[g]'], 'fpnd_value'))}/{fmt(number(by_name['GQT-994K[q]'], 'fpnd_value'))}/{fmt(number(by_name['GQT-994K[t]'], 'fpnd_value'))}, shared W1M {fmt(number(by_name['GQT-994K[g]'], 'w1m'))}, and mean W1P {fmt(number(by_name['GQT-994K[g]'], 'mean_w1p'))}. Its raw full-sample FPD is quarantined and must remain omitted. The gluon gain versus G-331K is substantial, but it confounds joint-class training, extra optimization, and campaign-harness changes; it is evidence of a strong result, not evidence that any one of those factors caused it.",
        f"- **Top remains the bottleneck.** The matched single-top specialist T-331K has FPND-t {fmt(number(by_name['T-331K'], 'fpnd_value'))} and W1M {fmt(number(by_name['T-331K'], 'w1m'))}, far above gluon-specialist G-331K's W1M {fmt(number(by_name['G-331K'], 'w1m'))}. This is a class-difficulty observation, not a fair g-vs-t model comparison.",
        "",
        "## What the ledger does not identify",
        "",
        "- E/F change multiple things relative to D (scalar initialization, online coupling, and readout), so their weaker W1M does not isolate any one of those ingredients.",
        "- A–F FPND values are historical and uncorrected; never rank those arms with post-fix FPND or literature FPND.",
        "- There is one seed per arm. Any claim that a small difference is reliable needs reruns or a paired evaluation protocol.",
        "- PC-JeDi and MPGAN use test-set jet attributes as conditioning, while JetFUEL samples attributes in Stage 1. The paper table is useful context, not a fully protocol-matched leaderboard.",
        "",
    ]
    return "\n".join(lines)


def write_or_check(path: Path, content: str, check: bool) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    matches = existing == content
    if check:
        return matches
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()

    rows = read_csv(args.ledger)
    baselines = read_csv(args.baselines)
    outputs = {
        args.out_dir / "jetfuel_campaign_results.tex": campaign_tex(rows),
        args.out_dir / "gluon30_paper_comparison.tex": paper_tex(comparable_ours(rows), baselines),
        args.out_dir / "jetnet30_class_comparison.tex": class_comparison_tex(rows, baselines),
        args.out_dir / "jetnet30_method_table.tex": method_table_document_tex(rows, baselines),
        args.out_dir / "campaign_ablation_trace.md": ablation_trace(rows),
    }
    stale = [path for path, content in outputs.items() if not write_or_check(path, content, args.check)]
    if stale:
        print("Generated artifacts are stale:")
        for path in stale:
            print(path)
        return 1
    if not args.check:
        print("Wrote:")
        for path in outputs:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
