"""Mass-shell geodesic ICP assignment in cache_icp.py (experiment plan Phase 4, revision 3).

The worker functions are importable without jetnet (heavy deps are lazy-imported in __main__).
Covers: geodesic Hungarian recovers a known permutation; identity rotation; and the masking
guard — only real particles enter the cost, so no real particle is matched to apex padding.
"""
import numpy as np
import pytest
import torch

# cache_icp imports scipy at module top (Hungarian assignment / Kabsch). The geodesic *cost*
# math is covered scipy-free in test_mass_shell.py; these worker tests need the assignment.
pytest.importorskip("scipy")

from cache_icp import _permute_geodesic, _icp_permute_worker


def test_permute_geodesic_recovers_known_permutation():
    """If x_0 is a shuffled copy of x_1, the assignment recovers the inverse shuffle exactly."""
    g = torch.Generator().manual_seed(0)
    n = 6
    p3 = (torch.rand(n, 3, generator=g, dtype=torch.float64) * 2 - 1) * 0.8
    x_1 = torch.cat([torch.zeros(n, 1, dtype=torch.float64), p3], dim=-1)
    perm_applied = torch.randperm(n, generator=g)
    x_0 = x_1[perm_applied]

    perm, rot = _permute_geodesic(x_0, x_1, m=0.1)
    # Applying perm to x_0 should reproduce x_1.
    assert np.allclose(x_0.numpy()[perm], x_1.numpy(), atol=1e-9)
    assert np.allclose(rot, np.eye(3), atol=0)


def test_worker_mass_shell_geometry():
    """The worker dispatches to the geodesic path and returns identity rotation + valid perm."""
    max_p = 8
    n_real = 5
    rng = np.random.default_rng(1)
    x_0 = np.zeros((max_p, 4), dtype=np.float32)
    x_1 = np.zeros((max_p, 4), dtype=np.float32)
    x_0[:n_real, 1:] = rng.standard_normal((n_real, 3)).astype(np.float32) * 0.5
    x_1[:n_real, 1:] = rng.standard_normal((n_real, 3)).astype(np.float32) * 0.5

    idx, perm_full, rot = _icp_permute_worker(
        (7, x_0, x_1, n_real, 100, "mass_shell", 0.1)
    )
    assert idx == 7
    assert np.allclose(rot, np.eye(3))
    # Real slice is a valid permutation of range(n_real); padding rows keep identity.
    assert sorted(perm_full[:n_real].tolist()) == list(range(n_real))
    assert perm_full[n_real:].tolist() == list(range(n_real, max_p))


def test_masking_guard_real_never_matched_to_padding():
    """Only the n_real real particles enter the cost, so every assigned index is < n_real —
    a real particle can never be matched to apex-parked padding."""
    max_p = 10
    n_real = 4
    rng = np.random.default_rng(2)
    x_0 = np.zeros((max_p, 4), dtype=np.float32)
    x_1 = np.zeros((max_p, 4), dtype=np.float32)
    x_0[:n_real, 1:] = rng.standard_normal((n_real, 3)).astype(np.float32) * 0.5
    x_1[:n_real, 1:] = rng.standard_normal((n_real, 3)).astype(np.float32) * 0.5
    # Padding rows are exactly zero (they will project to the apex).

    _, perm_full, _ = _icp_permute_worker((0, x_0, x_1, n_real, 100, "mass_shell", 0.1))
    # No real row is assigned to a padding index.
    assert (perm_full[:n_real] < n_real).all()


def test_worker_empty_jet():
    x = np.zeros((5, 4), dtype=np.float32)
    idx, perm_full, rot = _icp_permute_worker((3, x, x, 0, 100, "mass_shell", 0.1))
    assert idx == 3
    assert perm_full.tolist() == list(range(5))
    assert np.allclose(rot, np.eye(3))
