import json

from analysis.compare_mass_shell_stability import LABELS, collect


def _write_result(base, label, nonfinite, extreme, fpd):
    directory = base / label
    directory.mkdir()
    (directory / "endpoint_tail_diagnostics.json").write_text(json.dumps({
        "n_total": 10000,
        "n_nonfinite": nonfinite,
        "n_finite_max_abs_gt_1e6": extreme,
        "finite_max_abs_quantiles": {"p999": 100.0},
    }))
    (directory / "summary.json").write_text(json.dumps({
        "metrics": {"fpd": [fpd, 0.1], "fpnd_g": 10.0, "w1m": [0.1, 0.01]}
    }))


def test_compare_prefers_lowest_cost_stable_fixed_sampler(tmp_path):
    for label, bad in zip(LABELS, (True, False, False, False)):
        _write_result(tmp_path, label, int(bad), int(bad), 1.0)
    report = collect(tmp_path)
    assert report["recommended_sampler"] == "euler-128"


def test_compare_falls_back_to_adaptive(tmp_path):
    for label in LABELS:
        adaptive = label.startswith("adaptive")
        _write_result(tmp_path, label, 0 if adaptive else 1, 0 if adaptive else 1, 1.0)
    assert collect(tmp_path)["recommended_sampler"] == "adaptive-64-r05"
