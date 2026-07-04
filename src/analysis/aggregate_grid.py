"""
analysis/aggregate_grid.py — aggregate the Phase-1 ablation grid (runs A-F) into one table.

Reads each run directory's ``summary.json`` (written by train.py), pulls the flags + final loss
+ metrics, and applies the plan's per-run isotropy abort check:

    Under an isotropic jet axis, cos(theta) of the total 3-momentum is Uniform[-1, 1]. A run
    with a symmetry-breaking mechanism on (references or an anisotropic prior) MUST depart from
    uniform — KS p-value < threshold. Run B especially: if it does not depart, the references
    are broken; stop and debug the equivariance test before spending more budget.

If a run's summary predates the eval-time KS (no isotropy_ks_costheta_p), it is recomputed
locally from ``samples_subset.pt`` with the dependency-free KS in util/ks.py.

Usage:
    python -m analysis.aggregate_grid --runs A=/path/runA B=/path/runB ...
    python -m analysis.aggregate_grid --base /mnt/data/output/grid   # auto-discovers subdirs
"""
import argparse
import json
import os

from util.ks import ks_statistic_vs_uniform

ISOTROPIC_PRIORS = {"isotropic_com", "isotropic_lognorm"}
DEFAULT_KS_THRESHOLD = 1e-3

# Columns pulled from summary["metrics"] for the table, in order.
METRIC_COLUMNS = ["final_loss", "w1m", "fpd", "frac_negative_energy", "frac_spacelike",
                  "isotropy_ks_costheta_p"]


def _get_prior_dist(summary: dict) -> str:
    full = summary.get("full_config") or {}
    training = full.get("training") or {}
    return training.get("prior_dist", "isotropic_com")


def expected_break(config: dict, prior_dist: str) -> bool:
    """Does this run have a mechanism that should break the spurious rotational symmetry?
    References (an SO(2)-breaking virtual particle) or an anisotropic prior. Node scalars alone
    (run F) keep an isotropic prior + equivariant net → still isotropic."""
    if config.get("use_reference_vectors"):
        return True
    return prior_dist not in ISOTROPIC_PRIORS


def verdict(is_expected_break: bool, ks_p, threshold: float = DEFAULT_KS_THRESHOLD) -> str:
    """Map (expected-to-break, isotropy KS p) to a human verdict per the plan's abort logic."""
    if ks_p is None or (isinstance(ks_p, float) and ks_p != ks_p):  # None or NaN
        return "NO DATA"
    departs = ks_p < threshold
    if is_expected_break:
        return "PASS (broke isotropy)" if departs else "ABORT (expected break, still isotropic)"
    return "OK (isotropic as expected)" if not departs else "NOTE (departs without a mechanism)"


def recompute_isotropy_ks(samples_path: str):
    """Recompute (D, p) of the generated jet-axis cos(theta) vs uniform from samples_subset.pt.
    Lazy-imports torch so the summary-only path needs no torch. Returns (nan, nan) if absent."""
    if not os.path.exists(samples_path):
        return float("nan"), float("nan")
    import torch  # lazy: only needed for the recompute path
    samples = torch.load(samples_path, map_location="cpu")  # (N, P, 4)
    total = samples[..., 1:4].sum(dim=1)                    # (N, 3), padding already zero
    norm = total.norm(dim=-1).clamp(min=1e-12)
    cos_theta = (total[:, 2] / norm).numpy()
    return ks_statistic_vs_uniform(cos_theta, -1.0, 1.0)


def load_run(label: str, run_dir: str, ks_threshold: float = DEFAULT_KS_THRESHOLD) -> dict:
    """Load one run's summary into a flat row dict. Missing summary => a 'MISSING' row."""
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        return {"label": label, "status": "MISSING", "dir": run_dir}

    with open(summary_path) as f:
        summary = json.load(f)

    config = summary.get("config") or {}
    metrics = summary.get("metrics") or {}
    prior_dist = _get_prior_dist(summary)

    ks_p = metrics.get("isotropy_ks_costheta_p")
    ks_d = metrics.get("isotropy_ks_costheta")
    if ks_p is None:
        # Older run without eval-time KS: recompute locally from the saved samples.
        ks_d, ks_p = recompute_isotropy_ks(os.path.join(run_dir, "samples_subset.pt"))

    is_break = expected_break(config, prior_dist)
    row = {
        "label": label,
        "status": "ok",
        "dir": run_dir,
        "prior_dist": prior_dist,
        "use_reference_vectors": bool(config.get("use_reference_vectors")),
        "use_node_scalars": bool(config.get("use_node_scalars")),
        "use_adaln": bool(config.get("use_adaln")),
        "use_attention": bool(config.get("use_attention")),
        "final_loss": summary.get("final_loss"),
        "git_commit": summary.get("git_commit"),
        "isotropy_ks_costheta": ks_d,
        "isotropy_ks_costheta_p": ks_p,
        "expected_break": is_break,
        "verdict": verdict(is_break, ks_p, ks_threshold),
    }
    for k in ("w1m", "fpd", "frac_negative_energy", "frac_spacelike"):
        row[k] = metrics.get(k)
    return row


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if v != v else (f"{v:.3e}" if (abs(v) < 1e-3 or abs(v) >= 1e4) else f"{v:.4f}")
    return str(v)


def build_markdown_table(rows: list) -> str:
    """Render the aggregated rows as a markdown table (most-informative columns)."""
    headers = ["run", "refs", "h_i", "prior", "final_loss", "w1m", "fpd",
               "E<0", "spacelike", "KS(cosθ) p", "verdict"]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        if r.get("status") == "MISSING":
            lines.append(f"| {r['label']} | " + " | ".join(["—"] * (len(headers) - 2)) + " | MISSING |")
            continue
        lines.append("| " + " | ".join([
            r["label"],
            "Y" if r["use_reference_vectors"] else "n",
            "Y" if r["use_node_scalars"] else "n",
            r["prior_dist"],
            _fmt(r["final_loss"]), _fmt(r["w1m"]), _fmt(r["fpd"]),
            _fmt(r["frac_negative_energy"]), _fmt(r["frac_spacelike"]),
            _fmt(r["isotropy_ks_costheta_p"]), r["verdict"],
        ]) + " |")
    return "\n".join(lines)


def write_csv(rows: list, path: str):
    import csv
    fieldnames = ["label", "status", "use_reference_vectors", "use_node_scalars", "use_adaln",
                  "use_attention", "prior_dist", "final_loss", "w1m", "fpd",
                  "frac_negative_energy", "frac_spacelike", "isotropy_ks_costheta",
                  "isotropy_ks_costheta_p", "expected_break", "verdict", "git_commit"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_runs(run_args, base):
    """Return an ordered list of (label, dir). --runs LABEL=PATH ...; or discover subdirs of --base."""
    runs = []
    if run_args:
        for item in run_args:
            label, sep, path = item.partition("=")
            if not sep:
                raise ValueError(f"--runs entries must be LABEL=PATH, got {item!r}")
            runs.append((label, path))
    elif base:
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            if os.path.isdir(full):
                runs.append((name, full))
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", help="LABEL=PATH entries, e.g. A=/out/runA")
    parser.add_argument("--base", help="directory whose subdirectories are runs")
    parser.add_argument("--out", default=None, help="write the markdown+csv here (default: --base or cwd)")
    parser.add_argument("--ks-threshold", type=float, default=DEFAULT_KS_THRESHOLD)
    args = parser.parse_args(argv)

    runs = _parse_runs(args.runs, args.base)
    if not runs:
        parser.error("provide --runs LABEL=PATH ... or --base DIR")

    rows = [load_run(label, path, args.ks_threshold) for label, path in runs]
    table = build_markdown_table(rows)
    print(table)

    out_dir = args.out or args.base or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ablation_table.md"), "w") as f:
        f.write(table + "\n")
    write_csv(rows, os.path.join(out_dir, "ablation_table.csv"))
    print(f"\nWrote ablation_table.md and ablation_table.csv to {out_dir}")
    return rows


if __name__ == "__main__":
    main()
