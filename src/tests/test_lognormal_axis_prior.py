"""Contracts for the masked lognormal-share conditioned-axis prior."""

import torch

from util.data.distributions import JET_FEATURE_PRIORS, gen_initial_distribution
from util.geometry.mass_shell import project_to_shell
from util.geometry.minkowski_utils import normsq4


def _case(multiplicities=(3, 6), max_particles=8):
    dtype = torch.float64
    counts = torch.tensor(multiplicities, dtype=torch.long)
    features = torch.zeros(len(counts), 5, dtype=dtype)
    features[:, 0] = torch.tensor([0.3, -0.7], dtype=dtype)[:len(counts)]
    features[:, 1] = torch.tensor([1000.0, 725.0], dtype=dtype)[:len(counts)]
    features[:, 3] = counts
    phi = torch.tensor([0.2, 2.1], dtype=dtype)[:len(counts)]
    mask = (torch.arange(max_particles).unsqueeze(0) < counts.unsqueeze(1)).to(dtype)
    return features, phi, mask


def _sample(features, phi, mask, scale=50.0, x1=None):
    kwargs = dict(
        prior_dist="axis_aligned_lognormal",
        jet_features=features,
        jet_phi=phi,
        model_scale=scale,
        particle_mask=mask,
    )
    if x1 is None:
        kwargs.update(batch_size=features.shape[0], num_particles=mask.shape[1])
    else:
        kwargs["x_1"] = x1
    return gen_initial_distribution(**kwargs)


def test_lognormal_axis_prior_is_routed_jet_features_during_generation():
    assert "axis_aligned_lognormal" in JET_FEATURE_PRIORS


def test_lognormal_axis_prior_is_masked_positive_massless_and_exact_pt():
    features, phi, mask = _case()
    scale = 50.0
    torch.manual_seed(71)
    prior = _sample(features, phi, mask, scale)
    assert torch.equal(prior[mask == 0], torch.zeros_like(prior[mask == 0]))
    assert (prior[..., 0][mask > 0] > 0).all()
    assert torch.allclose(
        normsq4(prior)[mask > 0], torch.zeros_like(normsq4(prior)[mask > 0]),
        atol=2e-12, rtol=2e-12,
    )
    total_pt = torch.linalg.vector_norm(prior[..., 1:3].sum(1), dim=-1)
    assert torch.allclose(total_pt, features[:, 1] / scale, atol=2e-12, rtol=2e-12)


def test_lognormal_axis_prior_has_random_unequal_real_shares():
    features, phi, mask = _case(multiplicities=(8, 8), max_particles=8)
    torch.manual_seed(72)
    prior = _sample(features, phi, mask)
    particle_pt = torch.linalg.vector_norm(prior[..., 1:3], dim=-1)
    assert (particle_pt.std(dim=1) > 0.01 * particle_pt.mean(dim=1)).all()


def test_lognormal_axis_prior_is_batch_independent_with_assigned_phi():
    features, phi, mask = _case()
    torch.manual_seed(73)
    batched = _sample(features, phi, mask)
    torch.manual_seed(73)
    alone = _sample(features[:1], phi[:1], mask[:1])
    assert torch.equal(batched[0], alone[0])


def test_lognormal_axis_prior_variable_multiplicity_and_train_infer_match():
    features, phi, mask = _case(multiplicities=(1, 7), max_particles=9)
    torch.manual_seed(74)
    inference = _sample(features, phi, mask, scale=37.0)
    torch.manual_seed(74)
    training = _sample(
        features, phi, mask, scale=37.0,
        x1=torch.zeros(features.shape[0], mask.shape[1], 4, dtype=features.dtype),
    )
    assert torch.equal(training, inference)
    total_pt = torch.linalg.vector_norm(inference[..., 1:3].sum(1), dim=-1)
    assert torch.allclose(total_pt, features[:, 1] / 37.0, atol=2e-12, rtol=2e-12)


def test_lognormal_axis_prior_rfm_lift_is_on_shell():
    features, phi, mask = _case()
    torch.manual_seed(75)
    prior = _sample(features, phi, mask)
    shell = project_to_shell(prior, 0.1) * mask.unsqueeze(-1)
    assert torch.allclose(
        normsq4(shell)[mask > 0],
        torch.full_like(normsq4(shell)[mask > 0], 0.01),
        atol=2e-12, rtol=2e-12,
    )
    assert torch.equal(shell[mask == 0], torch.zeros_like(shell[mask == 0]))
