"""Checks for the axis-aligned prior (jetnet-dependent transform, so guarded)."""
import math

import pytest
import torch


def _jet_features(batch, device="cpu"):
    # [eta, pt, mass, num_particles, type]
    eta = torch.zeros(batch)
    pt = torch.ones(batch)
    return torch.stack([eta, pt, torch.zeros(batch), torch.full((batch,), 8.0), torch.zeros(batch)], dim=-1)


def test_axis_aligned_prior_shape_energy_and_collimation():
    pytest.importorskip("jetnet")
    from util.data.distributions import gen_initial_distribution

    torch.manual_seed(0)
    batch, num_particles = 64, 30
    jf = _jet_features(batch)

    x = gen_initial_distribution(
        batch_size=batch, num_particles=num_particles,
        prior_dist="axis_aligned", jet_features=jf, device="cpu",
    )
    assert x.shape == (batch, num_particles, 4)
    # Positive (lightlike) energies.
    assert (x[..., 0] > 0).all()

    # Collimation: the spread of momentum directions is much tighter than isotropic.
    iso = gen_initial_distribution(
        batch_size=batch, num_particles=num_particles, prior_dist="isotropic_com", device="cpu",
    )

    def direction_spread(p):
        p3 = p[..., 1:]
        n = p3 / p3.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        mean_dir = n.mean(dim=1, keepdim=True)
        mean_dir = mean_dir / mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos = (n * mean_dir).sum(-1).clamp(-1, 1)
        return cos.mean().item()  # closer to 1 => more collimated

    assert direction_spread(x) > direction_spread(iso)


def test_unknown_prior_raises():
    pytest.importorskip("jetnet")  # util.data.distributions imports jetnet at module load
    from util.data.distributions import gen_initial_distribution
    with pytest.raises(ValueError):
        gen_initial_distribution(batch_size=4, num_particles=10, prior_dist="does_not_exist")
