"""Regression gates for the checkpointed H readout ablations."""

import pytest
import torch

from models.lorentznet_flow import build_lorentznet
from tests.lorentz_test_utils import (
    apply_transform, build_model, random_proper_transform, sample_inputs,
)
from util.geometry.mass_shell import pushforward_to_tangent
from util.geometry.minkowski_utils import dotsq4


@pytest.mark.parametrize("mode", ["physical_logmap", "latent_displacement"])
def test_particle_direction_modes_are_jointly_equivariant(mode):
    model = build_model(seed=7, particle_direction_mode=mode)
    x, t, conditions, mask, refs = sample_inputs(seed=8)
    transform = random_proper_transform(seed=9)

    velocity = model(x, t, conditions, mask, ref_vectors=refs)
    transformed_velocity = model(
        apply_transform(x, transform) * mask.unsqueeze(-1), t, conditions, mask,
        ref_vectors=apply_transform(refs, transform),
    )

    assert torch.allclose(
        apply_transform(velocity, transform), transformed_velocity, atol=1e-5, rtol=1e-5
    )


def test_latent_readout_is_projected_to_the_physical_tangent_space():
    model = build_model(seed=10, particle_direction_mode="latent_displacement")
    x, t, conditions, mask, refs = sample_inputs(seed=11)

    velocity = model(x, t, conditions, mask, ref_vectors=refs)
    tangent_residual = dotsq4(x, velocity)[mask.bool()].abs().max()

    assert tangent_residual < 1e-8


def test_latent_readout_supports_the_float32_training_model():
    model = build_lorentznet(5, hidden_dim=16, num_layers=2, regulator_mass=1.0,
                             particle_direction_mode="latent_displacement").eval()
    x, t, conditions, mask, refs = sample_inputs(seed=15, dtype=torch.float32, mass=1.0)

    velocity = model(x, t, conditions, mask, ref_vectors=refs)

    assert velocity.dtype == torch.float64
    assert torch.isfinite(velocity).all()


def test_terminal_projection_bypass_matches_physical_h_to_roundoff():
    projected = build_model(seed=12)
    bypassed = build_model(seed=13, final_tangent_projection=False)
    bypassed.load_state_dict(projected.state_dict())
    x, t, conditions, mask, refs = sample_inputs(seed=14)

    projected_velocity = projected(x, t, conditions, mask, ref_vectors=refs)
    bypassed_velocity = bypassed(x, t, conditions, mask, ref_vectors=refs)
    reprojection = pushforward_to_tangent(x, bypassed_velocity, projected.regulator_mass)

    assert torch.allclose(projected_velocity, reprojection, atol=1e-10, rtol=1e-10)
    assert torch.allclose(projected_velocity, bypassed_velocity, atol=1e-10, rtol=1e-10)
