"""Regression tests for the controlled frozen-versus-fresh path audit."""

import numpy as np
import pytest
import torch

from tests.lorentz_test_utils import apply_transform, random_proper_transform
from util.mass_shell import mass_shell_loss, project_to_shell, pushforward_to_tangent
from util.off_path_audit import (
    cluster_bootstrap_h1,
    deterministic_indices,
    latin_hypercube_times,
    per_jet_field_metrics,
    transported_draw_dispersion,
)


def _tangent_batch(seed=0, batch=5, particles=6, mass=0.3):
    generator = torch.Generator().manual_seed(seed)
    spatial = torch.randn(batch, particles, 3, generator=generator, dtype=torch.float64)
    points = project_to_shell(
        torch.cat([torch.zeros(batch, particles, 1, dtype=torch.float64), spatial], -1),
        mass,
    )
    raw = torch.randn(batch, particles, 4, generator=generator, dtype=torch.float64)
    return points, pushforward_to_tangent(points, raw, mass)


def test_deterministic_selection_and_latin_hypercube_times():
    indices = deterministic_indices(100, 20, 71)
    assert np.array_equal(indices, deterministic_indices(100, 20, 71))
    assert len(np.unique(indices)) == 20
    assert np.all(np.diff(indices) > 0)

    times = latin_hypercube_times(20, 72)
    assert np.array_equal(times, latin_hypercube_times(20, 72))
    assert np.array_equal(np.sort(times), (np.arange(20) + 0.5) / 20)
    assert np.all((times > 0) & (times < 1))


def test_per_jet_loss_matches_global_mass_shell_loss_and_excludes_padding():
    points, target = _tangent_batch()
    generator = torch.Generator().manual_seed(2)
    delta = pushforward_to_tangent(
        points, torch.randn(target.shape, generator=generator, dtype=torch.float64), 0.3
    )
    prediction = target + 0.2 * delta
    mask = torch.ones(target.shape[:-1], dtype=torch.float64)
    mask[0, -2:] = 0
    metrics = per_jet_field_metrics(prediction, target, mask)
    weighted = (metrics["loss"] * mask.sum(1)).sum() / mask.sum()
    assert weighted == pytest.approx(
        mass_shell_loss(prediction, target, mask).item(), rel=1e-12
    )

    altered = prediction.clone()
    altered[0, -2:] += 1e6
    assert torch.equal(
        metrics["loss"], per_jet_field_metrics(altered, target, mask)["loss"]
    )


def test_per_jet_metrics_are_lorentz_invariant():
    points, target = _tangent_batch(seed=3)
    generator = torch.Generator().manual_seed(4)
    delta = pushforward_to_tangent(
        points, torch.randn(target.shape, generator=generator, dtype=torch.float64), 0.3
    )
    prediction = target + 0.1 * delta
    mask = torch.ones(target.shape[:-1], dtype=torch.float64)
    transform = random_proper_transform(seed=8)
    base = per_jet_field_metrics(prediction, target, mask)
    moved = per_jet_field_metrics(
        apply_transform(prediction, transform), apply_transform(target, transform), mask
    )
    for key in base:
        assert torch.allclose(base[key], moved[key], atol=1e-8, rtol=1e-7)


def test_cluster_bootstrap_is_deterministic_and_detects_excess_train_gap():
    rng = np.random.default_rng(4)
    train_cached = rng.lognormal(0, 0.1, 250)
    valid_frozen = rng.lognormal(0, 0.1, 250)
    train_fresh = np.stack([train_cached * 2.0] * 4, axis=1)
    valid_fresh = np.stack([valid_frozen] * 4, axis=1)
    first = cluster_bootstrap_h1(
        train_fresh, train_cached, valid_fresh, valid_frozen, seed=19, samples=200
    )
    second = cluster_bootstrap_h1(
        train_fresh, train_cached, valid_fresh, valid_frozen, seed=19, samples=200
    )
    assert first == second
    assert first["ratio"] == pytest.approx(2.0)
    assert first["ci95_ratio_low"] == pytest.approx(2.0)


def test_transported_draw_dispersion_zero_for_identical_draws():
    points, tangent = _tangent_batch(seed=7)
    states = points.unsqueeze(0).repeat(3, 1, 1, 1)
    predictions = tangent.unsqueeze(0).repeat(3, 1, 1, 1)
    mask = torch.ones(points.shape[:-1], dtype=torch.float64)
    dispersion = transported_draw_dispersion(states, predictions, points, mask, 0.3)
    assert torch.allclose(dispersion, torch.zeros_like(dispersion), atol=1e-14)


def test_audit_off_path_script_imports_standalone():
    """Running the audit as a script must resolve the src-rooted imports.

    The NRP job invokes ``python experiments/audit_off_path.py`` directly, which
    puts the script's own directory on ``sys.path[0]`` rather than ``src``. This is
    a regression guard for the ``ModuleNotFoundError: No module named 'cache_icp'``
    that killed the H1 off-path audit job twice; it is caught only when the script
    is exercised as a standalone process, not via the conftest-bootstrapped imports.
    """
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "experiments" / "audit_off_path.py"
    # Run from a directory that is not ``src`` so only the in-script bootstrap can
    # make the src-rooted module-level imports resolve.
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
