import json

from analysis.compare_aux_grid import compare, render_markdown


def _write_run(path, *, invalid, fpnd, efp, w1p, w1m, fpd=10.0):
    eval_dir = path / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.json").write_text(json.dumps({
        "metrics": {
            "frac_generated_invalid": invalid,
            "fpnd_g": fpnd,
            "fpd": [fpd, 0.1],
            "w1efp": [[efp] * 5, [0.1] * 5],
            "w1p": [[w1p] * 3, [0.1] * 3],
            "w1m": [w1m, 0.1],
            "cov_mmd": [0.5, 0.03],
        }
    }))


def test_compare_selects_best_arm_that_passes_guards(tmp_path):
    values = {
        "base": (0.0, 20.0, 2.0, 1.0, 1.0),
        "total": (0.0, 18.0, 1.8, 1.05, 1.05),
        "gram": (0.0, 15.0, 1.5, 1.2, 1.0),
        "combined": (0.0, 16.0, 1.6, 1.0, 1.0),
    }
    runs = {}
    for arm, value in values.items():
        runs[arm] = tmp_path / arm
        _write_run(
            runs[arm], invalid=value[0], fpnd=value[1], efp=value[2],
            w1p=value[3], w1m=value[4],
        )
    result = compare(runs)
    assert result["decisions"]["total"]["promising"]
    assert not result["decisions"]["gram"]["promising"]
    assert result["decisions"]["combined"]["promising"]
    assert result["winner"] == "combined"
    assert "ADVANCE" in render_markdown(result)


def test_compare_rejects_unstable_arm(tmp_path):
    runs = {}
    for arm in ("base", "total", "gram", "combined"):
        runs[arm] = tmp_path / arm
        _write_run(
            runs[arm], invalid=0.002 if arm == "total" else 0.0,
            fpnd=10.0 if arm == "base" else 9.0,
            efp=1.0 if arm == "base" else 0.9,
            w1p=1.0, w1m=1.0,
        )
    result = compare(runs)
    assert not result["decisions"]["total"]["promising"]
