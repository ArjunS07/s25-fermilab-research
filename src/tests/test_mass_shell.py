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
    geodesic_cost_matrix,
    mass_shell_loss,
    massless_energy_view,
    tangent_error_diagnostics,
)
from util.minkowski_utils import normsq4
from tests.lorentz_test_utils import build_model, sample_inputs


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


def test_massless_energy_view_preserves_spatial_momentum():
    p = project_to_shell(torch.randn(2, 3, 4, dtype=torch.float64), m=0.3)
    view = massless_energy_view(p)
    assert torch.equal(view[..., 1:4], p[..., 1:4])
    assert torch.allclose(view[..., 0], torch.linalg.vector_norm(p[..., 1:4], dim=-1))
    assert torch.allclose(normsq4(view), torch.zeros_like(normsq4(view)), atol=1e-10)


def test_tangent_error_diagnostics_detects_healthy_and_bad_vectors():
    p, target = _shell_points(batch=2, n=4, m=0.5)
    mask = torch.ones(2, 4, dtype=torch.float64)
    healthy = tangent_error_diagnostics(p, target.clone(), target, mask, m=0.5, dt=1 / 64)
    assert healthy["raw_loss_negative_fraction"] == 0.0
    assert healthy["step_clamp_fraction"] == 0.0
    assert healthy["nonfinite_fraction"] == 0.0

    bad = target.clone()
    bad[0, 0, 0] = float("nan")
    unhealthy = tangent_error_diagnostics(p, bad, target, mask, m=0.5, dt=1 / 64)
    assert unhealthy["nonfinite_fraction"] > 0.0


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


def test_geodesic_cost_matrix_matches_pairwise_distance():
    """cost[i,j] equals the geodesic distance between shell-lifted x0_i and x1_j; diagonal of a
    self-cost is (near) zero, so an identity assignment is optimal."""
    g = torch.Generator().manual_seed(12)
    n, m = 5, 0.1
    x = torch.cat([torch.zeros(n, 1, dtype=torch.float64),
                   (torch.rand(n, 3, generator=g, dtype=torch.float64) * 2 - 1) * 0.6], dim=-1)
    cost = geodesic_cost_matrix(x, x, m)
    assert cost.shape == (n, n)
    # Diagonal (same point) is the minimum of its row.
    assert torch.allclose(torch.diagonal(cost), cost.min(dim=1).values, atol=1e-6)
    # Matches an explicit pairwise call.
    expected = geodesic_distance(project_to_shell(x, m).unsqueeze(1),
                                 project_to_shell(x, m).unsqueeze(0), m)
    assert torch.allclose(cost, expected, atol=1e-12)


def test_exp_map_clamps_stiff_regime():
    """A pathologically large tangent step (small m, near-light-like shell) stays FINITE thanks
    to the invariant clamp on s = ||u||/m (guards cosh/sinh overflow). At the resulting scale
    (|q| ~ 1e13) the point is still on the shell in exact arithmetic, but <q,q> = m^2 is below
    the cancellation floor of normsq4, so we only assert finiteness + that <q,q> is negligible
    relative to |q|^2 (a near-light-like on-shell point). Tight on-shell is covered elsewhere."""
    m = 0.05
    p = project_to_shell(torch.tensor([[0.0, 3.0, -2.0, 1.0]], dtype=torch.float64), m)
    v = torch.tensor([[0.0, 500.0, -400.0, 300.0]], dtype=torch.float64)  # huge -> s clamped
    u = pushforward_to_tangent(p, v, m)
    q = exp_map(p, u, m)
    assert torch.isfinite(q).all()
    assert (q.abs().max() > 1e6)  # clamp engaged, step is large but finite
    assert normsq4(q).abs().item() < 1e-6 * float((q * q).sum())


def test_model_mass_shell_step_stays_on_shell():
    """model.step_hyperbolic(hyperbolic_model='mass_shell') keeps real particles on H_m."""
    m = 0.1
    model = build_model(use_reference_vectors=False, use_node_scalars=False, seed=0)
    x, _, cond, mask, _ = sample_inputs(seed=2)
    y0 = project_to_shell(x * mask.unsqueeze(-1), m)
    t0 = torch.tensor(0.0, dtype=torch.float64)
    t1 = torch.tensor(0.1, dtype=torch.float64)
    y1 = model.step_hyperbolic(y0, cond, mask, t0, t1,
                               hyperbolic_model="mass_shell", regulator_mass=m)
    assert torch.isfinite(y1).all()
    onshell = normsq4(y1)                                  # <y,y>
    real = mask > 0
    assert torch.allclose(onshell[real], torch.full_like(onshell[real], m * m), atol=1e-6)


def test_float32_inputs_produce_float64_geometry():
    p3 = torch.randn(2, 4, 4)
    p = project_to_shell(p3, 1.0)
    assert p.dtype == torch.float64
    q = exp_map(p, pushforward_to_tangent(p, torch.randn(2, 4, 4) * 0.2, 1.0), 1.0)
    assert q.dtype == torch.float64
    assert geodesic_distance(p, q, 1.0).dtype == torch.float64


@pytest.mark.parametrize("m", [0.1, 0.03])
def test_stiff_loss_is_float64_tangent_and_has_gradient(m):
    p, target = _shell_points(batch=2, n=5, seed=31, m=m)
    raw = torch.randn_like(target, dtype=torch.float32, requires_grad=True)
    pred = pushforward_to_tangent(p, raw, m)
    assert pred.dtype == torch.float64
    assert dotsq4(p, pred).abs().max() < 1e-8
    loss = mass_shell_loss(pred, target, torch.ones(2, 5), m)
    assert loss.dtype == torch.float64 and loss > 0
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0


def test_massless_endpoint_zeroes_padding_and_removes_constituent_mass_floor():
    m = 0.1
    mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
    raw = torch.tensor([[[0., 1., 0., 0.], [0., .5, .3, .2], [0., 0., 0., 0.]]],
                       dtype=torch.float64)
    shell = project_to_shell(raw, m)
    physical = massless_energy_view(shell, mask)
    real = mask.bool()
    assert (physical[..., 0][real] > 0).all()
    assert torch.allclose(normsq4(physical)[real], torch.zeros(2, dtype=torch.float64), atol=1e-12)
    assert torch.equal(physical[..., 1:4][real], shell[..., 1:4][real])
    assert torch.equal(physical[~real], torch.zeros_like(physical[~real]))
    shell_jet_mass = torch.sqrt(normsq4((shell * mask.unsqueeze(-1)).sum(1)).clamp(min=0))
    physical_jet_mass = torch.sqrt(normsq4(physical.sum(1)).clamp(min=0))
    assert physical_jet_mass < shell_jet_mass
