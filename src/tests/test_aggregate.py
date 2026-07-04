"""Results tooling: dependency-free KS (util/ks.py) + grid aggregator (analysis/aggregate_grid.py)."""
import json
import os

import numpy as np
import pytest

from util.ks import ks_statistic_vs_uniform, ks_pvalue
from analysis.aggregate_grid import (
    expected_break,
    verdict,
    load_run,
    build_markdown_table,
    _parse_runs,
    DEFAULT_KS_THRESHOLD,
)


# ── KS ──────────────────────────────────────────────────────────────────────

def test_ks_uniform_sample_has_large_pvalue():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=5000)
    d, p = ks_statistic_vs_uniform(x, -1.0, 1.0)
    assert d < 0.05
    assert p > 0.05  # cannot reject uniformity


def test_ks_concentrated_sample_rejects_uniform():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.02, size=5000)  # tightly peaked, very non-uniform on [-1,1]
    d, p = ks_statistic_vs_uniform(x, -1.0, 1.0)
    assert d > 0.3
    assert p < 1e-3


def test_ks_matches_analytic_shift():
    """A sample equal to the reference's own quantiles has ~zero KS statistic."""
    n = 1000
    x = (np.arange(n) + 0.5) / n * 2 - 1  # evenly spaced on [-1,1] ~ uniform quantiles
    d, p = ks_statistic_vs_uniform(x, -1.0, 1.0)
    assert d < 1.0 / n + 1e-9
    assert p > 0.99


def test_ks_pvalue_monotonic_and_bounded():
    assert ks_pvalue(0.0, 100) == pytest.approx(1.0)
    assert 0.0 <= ks_pvalue(0.5, 100) <= 1.0
    assert ks_pvalue(0.5, 100) < ks_pvalue(0.05, 100)


def test_ks_empty_is_nan():
    d, p = ks_statistic_vs_uniform([], -1.0, 1.0)
    assert np.isnan(d) and np.isnan(p)


# ── Aggregator logic ────────────────────────────────────────────────────────

def test_expected_break_rules():
    assert expected_break({"use_reference_vectors": True}, "isotropic_com") is True
    assert expected_break({"use_reference_vectors": False}, "axis_aligned") is True
    assert expected_break({"use_reference_vectors": False}, "isotropic_com") is False
    # Node scalars alone (run F): isotropic prior + refs off -> not expected to break.
    assert expected_break({"use_node_scalars": True}, "isotropic_com") is False


def test_verdict_matrix():
    assert verdict(True, 1e-6).startswith("PASS")
    assert verdict(True, 0.5).startswith("ABORT")
    assert verdict(False, 0.5).startswith("OK")
    assert verdict(False, 1e-6).startswith("NOTE")
    assert verdict(True, None) == "NO DATA"
    assert verdict(True, float("nan")) == "NO DATA"


def _write_summary(tmp_path, label, use_refs, prior, ks_p):
    d = tmp_path / label
    d.mkdir()
    summary = {
        "final_loss": 0.123,
        "git_commit": "deadbeef",
        "config": {"use_reference_vectors": use_refs, "use_node_scalars": False,
                   "use_adaln": False, "use_attention": False},
        "full_config": {"training": {"prior_dist": prior}},
        "metrics": {"w1m": 0.01, "fpd": 0.5, "frac_negative_energy": 0.0,
                    "frac_spacelike": 0.02, "isotropy_ks_costheta": 0.4,
                    "isotropy_ks_costheta_p": ks_p},
    }
    (d / "summary.json").write_text(json.dumps(summary))
    return str(d)


def test_load_run_pass_and_abort(tmp_path):
    # Run B: refs on, isotropy broken (tiny p) -> PASS.
    b_dir = _write_summary(tmp_path, "B", use_refs=True, prior="isotropic_com", ks_p=1e-8)
    row_b = load_run("B", b_dir)
    assert row_b["status"] == "ok"
    assert row_b["expected_break"] is True
    assert row_b["verdict"].startswith("PASS")

    # Run B': refs on but still isotropic (large p) -> ABORT signal.
    b2_dir = _write_summary(tmp_path, "Bflat", use_refs=True, prior="isotropic_com", ks_p=0.4)
    assert load_run("Bflat", b2_dir)["verdict"].startswith("ABORT")

    # Run A: baseline, isotropic as expected.
    a_dir = _write_summary(tmp_path, "A", use_refs=False, prior="isotropic_com", ks_p=0.5)
    assert load_run("A", a_dir)["verdict"].startswith("OK")


def test_load_run_missing_dir(tmp_path):
    row = load_run("Z", str(tmp_path / "does_not_exist"))
    assert row["status"] == "MISSING"


def test_build_markdown_table_includes_rows(tmp_path):
    b_dir = _write_summary(tmp_path, "B", use_refs=True, prior="isotropic_com", ks_p=1e-8)
    rows = [load_run("B", b_dir), {"label": "C", "status": "MISSING", "dir": "x"}]
    table = build_markdown_table(rows)
    assert "| run |" in table
    assert "PASS" in table
    assert "MISSING" in table


def test_parse_runs_labelled():
    runs = _parse_runs(["A=/p/a", "B=/p/b"], None)
    assert runs == [("A", "/p/a"), ("B", "/p/b")]
    with pytest.raises(ValueError):
        _parse_runs(["noequals"], None)
