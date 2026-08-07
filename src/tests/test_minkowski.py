"""Low-level invariant checks: normsq4 / dotsq4 and the E_i = <x, e_t> identity."""
import math

import torch

from util.geometry.minkowski_utils import normsq4, dotsq4
from tests.lorentz_test_utils import random_proper_transform, apply_transform


def test_normsq_matches_metric():
    x = torch.randn(4, 7, 4, dtype=torch.float64)
    expected = x[..., 0] ** 2 - x[..., 1:].pow(2).sum(-1)
    assert torch.allclose(normsq4(x), expected, atol=1e-12)


def test_dotsq_matches_metric():
    x = torch.randn(3, 5, 4, dtype=torch.float64)
    y = torch.randn(3, 5, 4, dtype=torch.float64)
    expected = x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(-1)
    assert torch.allclose(dotsq4(x, y), expected, atol=1e-12)


def test_energy_is_inner_product_with_e_t():
    """<x, e_t> with e_t = (1,0,0,0) must equal the energy component exactly.

    This underpins the per-node seed feature psi(E_i) = psi(<x, e_t>)."""
    x = torch.randn(2, 6, 4, dtype=torch.float64)
    e_t = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).expand(2, 1, 4)
    assert torch.allclose(dotsq4(x, e_t), x[..., 0], atol=1e-12)


def test_invariants_are_lorentz_invariant():
    """normsq4 and dotsq4 are unchanged by a joint Lorentz transform."""
    x = torch.randn(2, 6, 4, dtype=torch.float64) * 0.5
    y = torch.randn(2, 6, 4, dtype=torch.float64) * 0.5
    L = random_proper_transform(seed=3)
    xL, yL = apply_transform(x, L), apply_transform(y, L)
    assert torch.allclose(normsq4(x), normsq4(xL), atol=1e-9)
    assert torch.allclose(dotsq4(x, y), dotsq4(xL, yL), atol=1e-9)


def test_axis_alignment_changes_under_particles_only_rotation():
    """<x_i, x_jet> (the symmetry-breaking feature) must change when particles rotate but
    the jet-axis reference is held fixed — off the jet axis."""
    x = torch.randn(1, 5, 4, dtype=torch.float64) * 0.5
    x_jet = torch.tensor([[[3.0, 0.0, 0.0, 2.5]]], dtype=torch.float64)  # along z
    R = random_proper_transform(seed=7)  # generic, not about z
    before = dotsq4(x, x_jet)
    after = dotsq4(apply_transform(x, R), x_jet)
    assert not torch.allclose(before, after, atol=1e-3)
