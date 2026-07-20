import json

from analysis.compare_mass_shell_endpoints import ENDPOINTS, collect


def _write(base, label, end_time, bad=False, scale=1.0):
    directory = base / label
    directory.mkdir()
    (directory / "endpoint_tail_diagnostics.json").write_text(json.dumps({
        "n_nonfinite": int(bad), "n_finite_max_abs_gt_1e6": int(bad)}))
    (directory / "summary.json").write_text(json.dumps({
        "generation": {"integration_end_time": end_time},
        "metrics": {"fpd": [50 * scale, 0.1], "fpnd_g": 100 * scale,
                    "w1m": [0.05 * scale, 0.001]},
    }))


def test_selects_latest_stable_endpoint_with_preserved_bulk(tmp_path):
    times = (0.95, 0.98, 0.99, 0.995, 0.99999)
    for label, time in zip(ENDPOINTS, times):
        _write(tmp_path, label, time, bad=time > 0.99)
    report = collect(tmp_path)
    assert report["selected_endpoint"] == "t099"
    assert report["decision"] == "architecture"


def test_routes_to_finetune_when_stable_endpoints_degrade_bulk(tmp_path):
    times = (0.95, 0.98, 0.99, 0.995, 0.99999)
    for label, time in zip(ENDPOINTS, times):
        _write(tmp_path, label, time, bad=time >= 0.995, scale=1.2 if time < 0.995 else 1.0)
    assert collect(tmp_path)["decision"] == "stability_finetune"
