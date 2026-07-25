"""Compare the four mass-shell auxiliary-loss continuation arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("base", "total", "gram", "combined")


def _mean(values) -> float:
    return sum(float(value) for value in values) / len(values)


def load_metrics(run_dir: Path) -> dict[str, float]:
    summary_path = run_dir / "eval" / "summary.json"
    if not summary_path.is_file():
        summary_path = run_dir / "summary.json"
    payload = json.loads(summary_path.read_text())
    metrics = payload["metrics"]
    return {
        "invalid_fraction": float(metrics["frac_generated_invalid"]),
        "fpnd": float(metrics["fpnd_g"]),
        "fpd": float(metrics["fpd"][0]),
        "w1efp_mean": _mean(metrics["w1efp"][0]),
        "w1p_mean": _mean(metrics["w1p"][0]),
        "w1m": float(metrics["w1m"][0]),
        "coverage": float(metrics["cov_mmd"][0]),
        "mmd": float(metrics["cov_mmd"][1]),
    }


def compare(runs: dict[str, Path]) -> dict:
    if set(runs) != set(ARMS):
        raise ValueError(f"runs must contain exactly {ARMS}")
    metrics = {arm: load_metrics(path) for arm, path in runs.items()}
    baseline = metrics["base"]
    decisions = {}
    for arm in ARMS:
        current = metrics[arm]
        if arm == "base":
            decisions[arm] = {
                "stable": current["invalid_fraction"] <= 0.001,
                "promising": False,
                "reason": "continued-training control",
            }
            continue
        stable = current["invalid_fraction"] <= 0.001
        joint_improvement = (
            current["fpnd"] < baseline["fpnd"]
            and current["w1efp_mean"] < baseline["w1efp_mean"]
        )
        particle_guard = current["w1p_mean"] <= 1.10 * baseline["w1p_mean"]
        mass_guard = current["w1m"] <= 1.10 * baseline["w1m"]
        decisions[arm] = {
            "stable": stable,
            "joint_improvement": joint_improvement,
            "particle_guard": particle_guard,
            "mass_guard": mass_guard,
            "promising": stable and joint_improvement and particle_guard and mass_guard,
        }

    promising = [arm for arm in ARMS[1:] if decisions[arm]["promising"]]
    winner = None
    if promising:
        # Balance perceptual and EFP gains without letting either metric dominate
        # merely because of its numerical units.
        winner = min(
            promising,
            key=lambda arm: math.sqrt(
                metrics[arm]["fpnd"] / baseline["fpnd"]
                * metrics[arm]["w1efp_mean"] / baseline["w1efp_mean"]
            ),
        )
    return {"metrics": metrics, "decisions": decisions, "winner": winner}


def render_markdown(result: dict) -> str:
    lines = [
        "# Mass-shell auxiliary continuation comparison",
        "",
        "| arm | invalid | FPND | FPD | mean W1EFP | mean W1P | W1M | COV | MMD | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        metric = result["metrics"][arm]
        decision = result["decisions"][arm]
        label = (
            "ADVANCE" if decision["promising"]
            else "CONTROL" if arm == "base"
            else "DO NOT ADVANCE"
        )
        lines.append(
            f"| {arm} | {metric['invalid_fraction']:.3g} | {metric['fpnd']:.6g} "
            f"| {metric['fpd']:.6g} | {metric['w1efp_mean']:.6g} "
            f"| {metric['w1p_mean']:.6g} | {metric['w1m']:.6g} "
            f"| {metric['coverage']:.6g} | {metric['mmd']:.6g} | {label} |"
        )
    lines.extend([
        "",
        f"Selected winner: **{result['winner'] or 'none'}**.",
        "",
        "Advance requires ≤0.1% invalid samples, lower FPND and mean W1EFP "
        "than the continued-training control, and no >10% regression in mean W1P or W1M.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True,
        help="arm=/path/to/run (required once for each arm)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        arm, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"invalid --run value {item!r}")
        runs[arm] = Path(path)
    result = compare(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aux_grid_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output_dir / "aux_grid_comparison.md").write_text(render_markdown(result))


if __name__ == "__main__":
    main()
