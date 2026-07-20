"""Select the latest stable mass-shell integration endpoint without sacrificing bulk fidelity."""

import argparse
import json
from pathlib import Path


ENDPOINTS = ("t095", "t098", "t099", "t0995", "t099999")


def _first(value):
    return value[0] if isinstance(value, list) else value


def collect(base: Path) -> dict:
    rows = []
    for label in ENDPOINTS:
        with open(base / label / "endpoint_tail_diagnostics.json") as handle:
            tails = json.load(handle)
        with open(base / label / "summary.json") as handle:
            summary = json.load(handle)
        metrics = summary.get("metrics") or {}
        rows.append({
            "label": label,
            "end_time": summary["generation"]["integration_end_time"],
            "n_nonfinite": tails["n_nonfinite"],
            "n_finite_max_abs_gt_1e6": tails.get("n_finite_max_abs_gt_1e6", 0),
            "fpd": _first(metrics.get("fpd")),
            "fpnd_g": metrics.get("fpnd_g"),
            "w1m": _first(metrics.get("w1m")),
        })

    reference = rows[-1]
    fidelity_keys = ("fpd", "fpnd_g", "w1m")
    for row in rows:
        row["stable"] = row["n_nonfinite"] == 0 and row["n_finite_max_abs_gt_1e6"] == 0
        row["bulk_within_5pct"] = all(
            row[key] is not None and reference[key] is not None
            and row[key] <= 1.05 * reference[key] for key in fidelity_keys)
    eligible = [row for row in rows if row["stable"] and row["bulk_within_5pct"]]
    selected = max(eligible, key=lambda row: row["end_time"]) if eligible else None
    return {
        "reference": reference["label"],
        "rows": rows,
        "selected_endpoint": selected["label"] if selected else None,
        "decision": "architecture" if selected else "stability_finetune",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    args = parser.parse_args()
    report = collect(args.base)
    (args.base / "endpoint_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
