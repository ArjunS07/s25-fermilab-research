"""Lorentz covariance of the mass-shell geometry (experiment plan Phase 4).

The hyperboloid H_m and the (+,-,-,-) metric are preserved by proper orthochronous Lorentz
transforms Λ, so the geometry ops must be covariant:
    geodesic_distance(Λp, Λq)      == geodesic_distance(p, q)          (invariant)
    exp_map(Λp, Λu)                == Λ · exp_map(p, u)                (covariant)
    log_map(Λp, Λq)                == Λ · log_map(p, q)
    pushforward_to_tangent(Λp, Λv) == Λ · pushforward_to_tangent(p, v)
    geodesic_interpolant / conditional_vector_field   likewise covariant
    mass_shell_loss                invariant

This is the geometric analogue of the network-level equivariance gate: if these fail, the
mass-shell RFM target is frame-dependent and training on it is ill-posed.

(project_to_shell is intentionally NOT covariant — it fixes energy from |p_vec|, a
frame-dependent lift of massless data — so it is not tested here.)
"""
import pytest
import torch

from util.mass_shell import (
    project_to_shell,
    geodesic_distance,
    exp_map,
    log_map,
    pushforward_to_tangent,
    geodesic_interpolant,
    conditional_vector_field,
    mass_shell_loss,
)
from tests.lorentz_test_utils import random_proper_transform, apply_transform

M = 0.3
SEEDS = [0, 1, 2]


def _on_shell_and_tangent(batch=3, n=6, seed=0, m=M):
    """On-shell points p, q and a tangent field u at p (all float64)."""
    g = torch.Generator().manual_seed(seed)
    def shell(s):
        gg = torch.Generator().manual_seed(s)
        p3 = (torch.rand(batch, n, 3, generator=gg, dtype=torch.float64) * 2 - 1) * 0.7
        return project_to_shell(torch.cat([torch.zeros(batch, n, 1, dtype=torch.float64), p3], dim=-1), m)
    p = shell(seed)
    q = shell(seed + 100)
    v = (torch.rand(batch, n, 4, generator=g, dtype=torch.float64) * 2 - 1) * 0.4
    u = pushforward_to_tangent(p, v, m)
    return p, q, u, v


@pytest.mark.parametrize("seed", SEEDS)
def test_geodesic_distance_is_invariant(seed):
    p, q, _, _ = _on_shell_and_tangent(seed=seed)
    L = random_proper_transform(seed=seed)
    d = geodesic_distance(p, q, M)
    dL = geodesic_distance(apply_transform(p, L), apply_transform(q, L), M)
    assert torch.allclose(d, dL, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_exp_map_is_covariant(seed):
    p, _, u, _ = _on_shell_and_tangent(seed=seed)
    L = random_proper_transform(seed=seed)
    lhs = apply_transform(exp_map(p, u, M), L)
    rhs = exp_map(apply_transform(p, L), apply_transform(u, L), M)
    assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_log_map_is_covariant(seed):
    p, q, _, _ = _on_shell_and_tangent(seed=seed)
    L = random_proper_transform(seed=seed)
    lhs = apply_transform(log_map(p, q, M), L)
    rhs = log_map(apply_transform(p, L), apply_transform(q, L), M)
    assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_pushforward_is_covariant(seed):
    p, _, _, v = _on_shell_and_tangent(seed=seed)
    L = random_proper_transform(seed=seed)
    lhs = apply_transform(pushforward_to_tangent(p, v, M), L)
    rhs = pushforward_to_tangent(apply_transform(p, L), apply_transform(v, L), M)
    assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_interpolant_and_field_are_covariant(seed):
    p, q, _, _ = _on_shell_and_tangent(seed=seed)
    L = random_proper_transform(seed=seed)
    t = torch.full((p.shape[0],), 0.35, dtype=torch.float64)

    x_t = geodesic_interpolant(p, q, t, M)
    x_t_L = geodesic_interpolant(apply_transform(p, L), apply_transform(q, L), t, M)
    assert torch.allclose(apply_transform(x_t, L), x_t_L, atol=1e-6, rtol=1e-5)

    u_t = conditional_vector_field(x_t, q, t, M)
    u_t_L = conditional_vector_field(x_t_L, apply_transform(q, L), t, M)
    assert torch.allclose(apply_transform(u_t, L), u_t_L, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_loss_is_invariant(seed):
    p, _, u, _ = _on_shell_and_tangent(seed=seed)
    # Both vectors must inhabit the same tangent space; subtracting tangents based at
    # different shell points is not a defined Riemannian loss.
    _, _, _, v2 = _on_shell_and_tangent(seed=seed + 7)
    u2 = pushforward_to_tangent(p, v2, M)
    mask = torch.ones(p.shape[0], p.shape[1], dtype=torch.float64)
    L = random_proper_transform(seed=seed)
    base = mass_shell_loss(u, u2, mask, M)
    boosted = mass_shell_loss(apply_transform(u, L), apply_transform(u2, L), mask, M)
    assert torch.allclose(base, boosted, atol=1e-6, rtol=1e-5)
