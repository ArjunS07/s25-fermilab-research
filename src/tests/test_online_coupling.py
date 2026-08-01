"""Tests for the online per-batch geodesic ICP coupling (fresh-noise flow matching)."""

import numpy as np
import torch

from tests.lorentz_test_utils import apply_transform, random_proper_transform
from util.online_coupling import geodesic_permutation, online_geodesic_coupling

M = 0.1


def _clouds(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, 4, generator=g, dtype=torch.float64)
    x1 = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return x0, x1


def test_recovers_exact_permutation_when_target_is_a_permuted_prior():
    # If x1 is exactly a permutation of x0, the zero-cost matching is unique and the
    # returned perm must reproduce it: x0[perm] == x1.
    g = torch.Generator().manual_seed(3)
    n = 9
    x0 = torch.randn(n, 4, generator=g, dtype=torch.float64)
    true_perm = torch.randperm(n, generator=g)
    x1 = x0[true_perm]
    perm = geodesic_permutation(x0, x1, M)
    assert torch.allclose(x0[torch.as_tensor(perm, dtype=torch.long)], x1, atol=1e-12)


def test_permutation_is_a_valid_permutation_and_beats_identity():
    x0, x1 = _clouds(10, seed=1)
    perm = geodesic_permutation(x0, x1, M)
    assert sorted(perm.tolist()) == list(range(10))  # a genuine permutation
    from util.mass_shell import geodesic_cost_matrix
    cost = geodesic_cost_matrix(x0, x1, M).square()
    paired_cost = cost[torch.as_tensor(perm, dtype=torch.long), torch.arange(10)].sum()
    identity_cost = cost.diagonal().sum()
    assert paired_cost <= identity_cost + 1e-12


def test_deterministic():
    x0, x1 = _clouds(8, seed=2)
    assert np.array_equal(geodesic_permutation(x0, x1, M), geodesic_permutation(x0, x1, M))


def test_boost_invariance_of_assignment():
    # The geodesic cost is Lorentz invariant, so a proper transform applied to both
    # clouds must leave the assignment unchanged.
    x0, x1 = _clouds(7, seed=4)
    L = random_proper_transform(seed=4)
    perm = geodesic_permutation(x0, x1, M)
    perm_boosted = geodesic_permutation(apply_transform(x0, L), apply_transform(x1, L), M)
    assert np.array_equal(perm, perm_boosted)


def test_batch_preserves_target_and_padding_and_pairs_only_real():
    B, P = 4, 12
    g = torch.Generator().manual_seed(5)
    x0 = torch.randn(B, P, 4, generator=g, dtype=torch.float64)
    x1 = torch.randn(B, P, 4, generator=g, dtype=torch.float64)
    mask = torch.zeros(B, P, dtype=torch.float64)
    n_real = [12, 6, 1, 0]
    for b, n in enumerate(n_real):
        mask[b, :n] = 1.0
    x1_before = x1.clone()
    paired = online_geodesic_coupling(x0, x1, mask, M)

    assert torch.equal(x1, x1_before)  # target never mutated
    for b, n in enumerate(n_real):
        # padding rows untouched
        assert torch.allclose(paired[b, n:], x0[b, n:])
        # real rows are a permutation of the original real rows
        real_before = {tuple(r.tolist()) for r in x0[b, :n]}
        real_after = {tuple(r.tolist()) for r in paired[b, :n]}
        assert real_before == real_after
    # n_real <= 1 jets are identity
    assert torch.allclose(paired[2], x0[2])
    assert torch.allclose(paired[3], x0[3])


def test_batch_matches_per_jet_permutation():
    B, P = 3, 8
    g = torch.Generator().manual_seed(6)
    x0 = torch.randn(B, P, 4, generator=g, dtype=torch.float64)
    x1 = torch.randn(B, P, 4, generator=g, dtype=torch.float64)
    mask = torch.ones(B, P, dtype=torch.float64)
    paired = online_geodesic_coupling(x0, x1, mask, M)
    for b in range(B):
        perm = geodesic_permutation(x0[b], x1[b], M)
        assert torch.allclose(paired[b], x0[b][torch.as_tensor(perm, dtype=torch.long)])
