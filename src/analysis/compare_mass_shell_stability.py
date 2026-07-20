"""Summarize a fixed-step/adaptive mass-shell inference matrix."""

import argparse
import json
from pathlib import Path


LABELS = ("euler-64", "euler-128", "euler-256", "adaptive-64-r05")


def collect(base: Path) -> dict:
    rows = []
    for label in LABELS:
        directory = base / label
        with open(directory / "endpoint_tail_diagnostics.json") as handle:
            tails = json.load(handle)
        with open(directory / "summary.json") as handle:
            summary = json.load(handle)
        metrics = summary.get("metrics") or {}
        rows.append({
            "label": label,
            "n_total": tails["n_total"],
            "n_nonfinite": tails["n_nonfinite"],
            "n_finite_max_abs_gt_1e6": tails.get("n_finite_max_abs_gt_1e6", 0),
            "max_abs_p999": tails.get("finite_max_abs_quantiles", {}).get("p999"),
            "fpd": (metrics.get("fpd") or [None])[0],
            "fpnd_g": metrics.get("fpnd_g"),
            "w1m": (metrics.get("w1m") or [None])[0],
        })
    stable = [row for row in rows
              if row["n_nonfinite"] == 0 and row["n_finite_max_abs_gt_1e6"] == 0]
    fixed = [row for row in stable if row["label"].startswith("euler-")]
    recommendation = (fixed[0]["label"] if fixed else
                      (stable[0]["label"] if stable else "none-stable"))
    return {"rows": rows, "recommended_sampler": recommendation}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    args = parser.parse_args()
    report = collect(args.base)
    with open(args.base / "stability_comparison.json", "w") as handle:
        json.dump(report, handle, indent=2)
    headers = ("label", "nonfinite", "finite >1e6", "p99.9 maxabs", "FPD", "FPND", "W1M")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in report["rows"]:
        lines.append("| " + " | ".join(str(value) for value in (
            row["label"], row["n_nonfinite"], row["n_finite_max_abs_gt_1e6"],
            row["max_abs_p999"], row["fpd"], row["fpnd_g"], row["w1m"])) + " |")
    lines.extend(["", f"Recommended stable sampler: `{report['recommended_sampler']}`", ""])
    (args.base / "stability_comparison.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
