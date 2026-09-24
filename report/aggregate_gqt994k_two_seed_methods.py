"""Average class-matched JetFUEL method metrics across two training seeds.

Use only after all 12 arms of
``as-jet-eval-gqt994k-noicp-latent-two-seeds-a40`` complete. The evaluator
writes one pointer per method/seed/class. This script averages *per-run metric
estimates*, rather than pooling samples and masking between-training-seed
variation. It reports both seed values and their sample standard deviation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METHODS = ("noicp", "latent")
CLASSES = ("g", "q", "t")
SEEDS = (42, 43)
WEIGHTS = {
    "noicp": {"g": 0.0, "q": 0.25, "t": 0.5},
    "latent": {"g": 0.0, "q": 0.25, "t": 0.75},
}


def point_estimate(value):
    return float(value[0] if isinstance(value, (list, tuple)) else value)


def metrics_for_class(summary: dict, jet_class: str) -> dict[str, float]:
    metrics = summary["metrics"]
    w1p = metrics[f"w1p_{jet_class}"]
    cov_mmd = metrics[f"cov_mmd_{jet_class}"]
    return {
        "fpnd": point_estimate(metrics[f"fpnd_{jet_class}"]),
        "fpd": point_estimate(metrics[f"fpd_{jet_class}"]),
        "w1m": point_estimate(metrics[f"w1m_{jet_class}"]),
        "mean_w1p": statistics.mean(map(float, w1p[0])),
        "cov": float(cov_mmd[0]),
        "mmd": float(cov_mmd[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pointers-dir", type=Path,
        default=Path("/mnt/data/output/run-pointers/gqt994k-noicp-latent-two-seeds"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/mnt/data/output/analysis/gqt994k_noicp_latent_two_seed_metrics.csv"),
    )
    args = parser.parse_args()

    rows = []
    for method in METHODS:
        for jet_class in CLASSES:
            per_seed = {}
            for seed in SEEDS:
                pointer = args.pointers_dir / f"{method}-seed{seed}-{jet_class}.txt"
                run_dir = Path(pointer.read_text().strip())
                summary_path = run_dir / "eval/summary.json"
                per_seed[seed] = metrics_for_class(
                    json.loads(summary_path.read_text()), jet_class
                )
            for metric in ("fpnd", "fpd", "w1m", "mean_w1p", "cov", "mmd"):
                values = [per_seed[seed][metric] for seed in SEEDS]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Nonfinite {metric} for {method}/{jet_class}: {values}")
                rows.append({
                    "method": method,
                    "jet_class": jet_class,
                    "cfg_weight": WEIGHTS[method][jet_class],
                    "metric": metric,
                    "seed42": values[0],
                    "seed43": values[1],
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} two-seed metric rows to {args.output}")


if __name__ == "__main__":
    main()
