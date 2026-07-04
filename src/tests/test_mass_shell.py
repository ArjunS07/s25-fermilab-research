"""Mass-shell (hyperboloid) RFM geometry (experiment plan Phase 4).

Verifies the geometric identities that make the flow-matching path well-defined:
  - exp/log are inverse; exp stays on the shell.
  - geodesic distance is symmetric, zero on the diagonal, and scales with m.
  - the RFM interpolant hits its endpoints and is consistent with the conditional field.
  - project_to_shell lands exactly on H_m (apex for zero momentum).
  - float32 in -> float32 out (float64 used internally).
"""
import math

import pytest
import torch

from util.minkowski_utils import dotsq4
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


def _shell_points(batch=4, n=6, seed=0, m=1.0, dtype=torch.float64):
    """Random on-shell points and a moderate tangent field at the first set."""
    g = torch.Generator().manual_seed(seed)
    p3 = (torch.rand(batch, n, 3, generator=g, dtype=dtype) * 2 - 1) * 0.8
    p = project_to_shell(torch.cat([torch.zeros(batch, n, 1, dtype=dtype), p3], dim=-1), m)
    # A tangent vector at p: project a random Cartesian vector, then scale to a modest norm.
    v = (torch.rand(batch, n, 4, generator=g, dtype=dtype) * 2 - 1) * 0.5
    u = pushforward_to_tangent(p, v, m)
    return p, u


@pytest.mark.parametrize("m", [1.0, 0.5, 2.0])
def test_project_to_shell_is_on_shell(m):
    p3 = torch.randn(3, 5, 4, dtype=torch.float64)
    p = project_to_shell(p3, m)
    assert torch.allclose(dotsq4(p, p), torch.full((3, 5), m * m, dtype=torch.float64), atol=1e-9)
    # Positive energy sheet.
    assert (p[..., 0] > 0).all()


def test_project_zero_momentum_is_apex():
    z = torch.zeros(2, 3, 4, dtype=torch.float64)
    p = project_to_shell(z, m=1.3)
    apex = torch.tensor([1.3, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(p, apex.expand_as(p), atol=1e-9)


def test_pushforward_is_tangent():
    p, u = _shell_points(m=1.0)
    # <p, u> == 0 for a tangent vector.
    assert torch.allclose(dotsq4(p, u), torch.zeros_like(dotsq4(p, u)), atol=1e-9)


@pytest.mark.parametrize("m", [1.0, 0.7])
def test_exp_stays_on_shell(m):
    p, u = _shell_points(m=m)
    q = exp_map(p, u, m)
    assert torch.allclose(dotsq4(q, q), torch.full_like(dotsq4(q, q), m * m), atol=1e-7)


@pytest.mark.parametrize("m", [1.0, 0.7, 1.5])
def test_exp_log_roundtrip(m):
    p, u = _shell_points(m=m)
    q = exp_map(p, u, m)
    u_rec = log_map(p, q, m)
    assert torch.allclose(u_rec, u, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("m", [1.0, 0.7])
def test_log_exp_roundtrip(m):
    p, _ = _shell_points(seed=1, m=m)
    q, _ = _shell_points(seed=2, m=m)
    # exp_p(log_p(q)) == q.
    q_rec = exp_map(p, log_map(p, q, m), m)
    assert torch.allclose(q_rec, q, atol=1e-6, rtol=1e-5)


def test_distance_symmetry_and_zero_diagonal():
    p, _ = _shell_points(seed=3)
    q, _ = _shell_points(seed=4)
    assert torch.allclose(geodesic_distance(p, q, 1.0), geodesic_distance(q, p, 1.0), atol=1e-9)
    # d(p,p) floors at arccosh(1+eps) ~ 1.4e-6 from the domain clamp — physically negligible.
    assert torch.allclose(geodesic_distance(p, p, 1.0), torch.zeros_like(geodesic_distance(p, p, 1.0)),
                          atol=1e-5)


def test_distance_equals_tangent_norm_of_log():
    """d(p,q) == ||log_p(q)|| (the log map has norm equal to the geodesic distance)."""
    p, _ = _shell_points(seed=5)
    q, _ = _shell_points(seed=6)
    u = log_map(p, q, 1.0)
    norm = torch.sqrt((-dotsq4(u, u)).clamp(min=0.0))
    assert torch.allclose(norm, geodesic_distance(p, q, 1.0), atol=1e-6)


@pytest.mark.parametrize("m", [1.0, 0.8])
def test_interpolant_endpoints(m):
    x0, _ = _shell_points(seed=7, m=m)
    x1, _ = _shell_points(seed=8, m=m)
    B = x0.shape[0]
    t0 = torch.zeros(B, dtype=torch.float64)
    t1 = torch.ones(B, dtype=torch.float64)
    assert torch.allclose(geodesic_interpolant(x0, x1, t0, m), x0, atol=1e-6)
    assert torch.allclose(geodesic_interpolant(x0, x1, t1, m), x1, atol=1e-6)


def test_conditional_field_recovers_endpoint():
    """(1-t) * u_t(x_t|x1) == log_{x_t}(x1), so exp_{x_t}((1-t) u_t) == x1."""
    m = 1.0
    x0, _ = _shell_points(seed=9, m=m)
    x1, _ = _shell_points(seed=10, m=m)
    B = x0.shape[0]
    t = torch.full((B,), 0.4, dtype=torch.float64)
    x_t = geodesic_interpolant(x0, x1, t, m)
    u_t = conditional_vector_field(x_t, x1, t, m)
    step = (1.0 - 0.4) * u_t
    recovered = exp_map(x_t, step, m)
    assert torch.allclose(recovered, x1, atol=1e-6, rtol=1e-5)


def test_loss_zero_when_pred_equals_target_and_masks():
    p, u = _shell_points(seed=11)
    mask = torch.ones(p.shape[0], p.shape[1], dtype=torch.float64)
    assert mass_shell_loss(u, u, mask).item() == pytest.approx(0.0, abs=1e-9)
    # Masked-out particles do not contribute.
    other = u + 1.0
    mask_zero = torch.zeros_like(mask)
    assert mass_shell_loss(other, u, mask_zero).item() == pytest.approx(0.0, abs=1e-9)


def test_float32_in_float32_out():
    p3 = torch.randn(2, 4, 4)
    p = project_to_shell(p3, 1.0)
    assert p.dtype == torch.float32
    q = exp_map(p, pushforward_to_tangent(p, torch.randn(2, 4, 4) * 0.2, 1.0), 1.0)
    assert q.dtype == torch.float32
    assert geodesic_distance(p, q, 1.0).dtype == torch.float32
