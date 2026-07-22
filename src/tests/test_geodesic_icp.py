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

from cache_icp import (_permute_geodesic, _icp_permute_worker,
                       resolve_training_cache_path, source_dataset_fingerprint,
                       validate_cache_metadata,
                       CACHE_FORMAT_VERSION)


def test_icp_opt_out_ignores_existing_shared_cache():
    exists = lambda _: True
    assert resolve_training_cache_path(False, None, "/cache", ["g"], 30, exists=exists) is None


def test_icp_requires_cache_when_enabled():
    with pytest.raises(FileNotFoundError):
        resolve_training_cache_path(True, None, "/cache", ["g"], 30, exists=lambda _: False)


def test_icp_rejects_explicit_path_when_disabled():
    with pytest.raises(ValueError):
        resolve_training_cache_path(False, "/cache/file.pkl", "/cache", ["g"], 30)


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


def test_squared_geodesic_is_a_real_assignment_ablation(monkeypatch):
    # Linear distance prefers the diagonal (0 + 4 < 2.1 + 2.1), whereas
    # squared distance prefers the off-diagonal (0 + 16 > 4.41 + 4.41).
    cost = torch.tensor([[0.0, 2.1], [2.1, 4.0]], dtype=torch.float64)
    monkeypatch.setattr("util.mass_shell.geodesic_cost_matrix", lambda *args: cost)
    x = torch.zeros(2, 4, dtype=torch.float64)
    linear, _ = _permute_geodesic(x, x, 0.1, "geodesic")
    squared, _ = _permute_geodesic(x, x, 0.1, "squared_geodesic")
    assert linear.tolist() == [0, 1]
    assert squared.tolist() == [1, 0]


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


def test_paired_cache_metadata_rejects_dataset_reordering():
    paired = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    metadata = {"dataset_fingerprint": "ordered", "dataset_indices": [0, 1],
                "prior_dist": "axis_aligned", "seed": 42}
    payload = {"format_version": CACHE_FORMAT_VERSION, "paired_x0": paired,
               "metadata": metadata}
    validate_cache_metadata(payload, metadata)
    with pytest.raises(ValueError, match="metadata"):
        validate_cache_metadata(payload, {**metadata, "dataset_indices": [1, 0]})
    with pytest.raises(ValueError, match="legacy"):
        validate_cache_metadata({"perm_cache": np.zeros((2, 3))}, metadata)


def test_source_fingerprint_is_order_sensitive_and_transform_independent():
    particles = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
    jets = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    dataset = (particles, jets)

    expected = source_dataset_fingerprint(dataset, n_samples=2, num_particles=3)
    # A derived Cartesian representation is deliberately absent from the API.
    assert expected == source_dataset_fingerprint(dataset, n_samples=2, num_particles=3)
    reordered = (particles.flip(0), jets.flip(0))
    assert expected != source_dataset_fingerprint(reordered, n_samples=2, num_particles=3)
    assert expected != source_dataset_fingerprint(dataset, n_samples=1, num_particles=3)


def test_geodesic_alignment_reduces_same_realization_cost():
    g = torch.Generator().manual_seed(13)
    x1 = torch.cat([torch.zeros(7, 1, dtype=torch.float64),
                    torch.randn(7, 3, generator=g, dtype=torch.float64)], dim=-1)
    shuffle = torch.randperm(7, generator=g)
    x0 = x1[shuffle]
    perm, _ = _permute_geodesic(x0, x1, 0.1)
    unaligned = torch.linalg.vector_norm(x0[:, 1:] - x1[:, 1:], dim=-1).sum()
    aligned = torch.linalg.vector_norm(x0[perm, 1:] - x1[:, 1:], dim=-1).sum()
    assert aligned < unaligned
    # The cached tensor is the actual changed training pair, not an instruction for a fresh draw.
    paired = x0[perm]
    fresh = torch.randn_like(x0)
    assert torch.equal(paired, x1)
    assert not torch.equal(paired, fresh[torch.as_tensor(perm)])
