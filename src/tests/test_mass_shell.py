"""Compact physical-geometry gate for the regulated mass-shell sampler."""

import torch

from tests.lorentz_test_utils import build_model, sample_inputs
from util.geometry.mass_shell import exp_map, project_to_shell, pushforward_to_tangent
from util.geometry.minkowski_utils import normsq4


def test_projection_and_geodesic_step_stay_on_shell():
    mass = 1.0
    state = project_to_shell(torch.randn(2, 5, 4, dtype=torch.float64), mass)
    tangent = 0.01 * pushforward_to_tangent(state, torch.randn_like(state), mass)
    stepped = exp_map(state, tangent, mass)

    assert torch.allclose(
        normsq4(state), torch.full_like(normsq4(state), mass**2), atol=1e-9
    )
    assert torch.allclose(
        normsq4(stepped), torch.full_like(normsq4(stepped), mass**2), atol=1e-7
    )


def test_generation_euler_path_stays_finite_and_on_shell():
    mass = 1.0
    model = build_model(seed=0, regulator_mass=mass)
    x, _, conditions, mask, refs = sample_inputs(seed=3, mass=mass)
    state = project_to_shell(x * mask.unsqueeze(-1), mass)
    times = torch.linspace(0, 1, 16, dtype=torch.float64)

    for start, end in zip(times[:-1], times[1:]):
        state = model.step_hyperbolic(
            state, conditions, mask, start, end, ref_vectors=refs
        )

    real = mask.bool()
    assert torch.isfinite(state).all()
    assert torch.allclose(
        normsq4(state)[real], torch.full_like(normsq4(state)[real], mass**2),
        atol=1e-6,
    )


def test_step_returns_per_trajectory_nonfinites_for_sampler_isolation(monkeypatch):
    model = build_model(seed=31)
    x, t, conditions, mask, refs = sample_inputs(batch=2, seed=32)

    def zero_velocity(state, *_args, **_kwargs):
        return torch.zeros_like(state)

    def one_bad_trajectory(state, _velocity, _mass):
        result = state.clone()
        result[1] = float("nan")
        return result

    monkeypatch.setattr(model, "_mass_shell_velocity", zero_velocity)
    monkeypatch.setattr("models.lorentznet_flow.exp_map", one_bad_trajectory)
    stepped = model.step_hyperbolic(
        x, conditions, mask, t[0], t[0] + 0.1, ref_vectors=refs,
    )

    assert torch.isfinite(stepped[0]).all()
    assert not torch.isfinite(stepped[1]).all()
