"""Contract tests for the simple equal-share axis-aligned prior."""

import torch

from util.data.distributions import gen_initial_distribution
from util.geometry.mass_shell import project_to_shell
from util.geometry.minkowski_utils import normsq4


def _case(multiplicities=(3, 6), max_particles=8):
    dtype = torch.float64
    multiplicities = torch.tensor(multiplicities, dtype=torch.long)
    batch = len(multiplicities)
    features = torch.zeros(batch, 5, dtype=dtype)
    features[:, 0] = torch.tensor([0.3, -0.7], dtype=dtype)[:batch]
    features[:, 1] = torch.tensor([1000.0, 725.0], dtype=dtype)[:batch]
    features[:, 3] = multiplicities.to(dtype)
    phi = torch.tensor([0.2, 2.1], dtype=dtype)[:batch]
    mask = (
        torch.arange(max_particles).unsqueeze(0)
        < multiplicities.unsqueeze(1)
    ).to(dtype)
    return features, phi, mask


def _sample(features, phi, mask, scale=50.0, x1=None):
    kwargs = dict(
        prior_dist="axis_aligned_equal",
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


def test_equal_axis_prior_padding_is_exactly_zero():
    features, phi, mask = _case()
    prior = _sample(features, phi, mask)
    assert torch.equal(prior[mask == 0], torch.zeros_like(prior[mask == 0]))


def test_equal_axis_prior_real_energies_are_positive():
    features, phi, mask = _case()
    prior = _sample(features, phi, mask)
    assert (prior[..., 0][mask > 0] > 0).all()


def test_equal_axis_prior_is_massless_before_rfm_lift():
    features, phi, mask = _case()
    prior = _sample(features, phi, mask)
    assert torch.allclose(
        normsq4(prior)[mask > 0], torch.zeros_like(normsq4(prior)[mask > 0]),
        atol=2e-12, rtol=2e-12,
    )


def test_equal_axis_prior_matches_conditioned_transverse_vector_sum():
    features, phi, mask = _case()
    scale = 50.0
    prior = _sample(features, phi, mask, scale)
    total_pt = torch.linalg.vector_norm(prior[..., 1:3].sum(dim=1), dim=-1)
    assert torch.allclose(total_pt, features[:, 1] / scale, atol=2e-12, rtol=2e-12)
    # Every real particle receives the same scalar pT share after the common correction.
    particle_pt = torch.linalg.vector_norm(prior[..., 1:3], dim=-1)
    for row, count in zip(particle_pt, mask.sum(dim=1).long()):
        assert torch.allclose(row[:count], row[0].expand(count), atol=2e-12, rtol=2e-12)


def test_equal_axis_prior_is_independent_of_other_jets_in_batch():
    features, phi, mask = _case()
    torch.manual_seed(101)
    batched = _sample(features, phi, mask)
    torch.manual_seed(101)
    alone = _sample(features[:1], phi[:1], mask[:1])
    assert torch.equal(batched[0], alone[0])


def test_equal_axis_prior_variable_multiplicity_retains_requested_pt():
    features, phi, mask = _case(multiplicities=(1, 7), max_particles=9)
    scale = 37.0
    prior = _sample(features, phi, mask, scale)
    total_pt = torch.linalg.vector_norm(prior[..., 1:3].sum(dim=1), dim=-1)
    assert torch.allclose(total_pt, features[:, 1] / scale, atol=2e-12, rtol=2e-12)
    assert torch.equal(prior[mask == 0], torch.zeros_like(prior[mask == 0]))


def test_training_and_inference_prior_calls_share_orientation_convention():
    features, phi, mask = _case()
    torch.manual_seed(303)
    inference_prior = _sample(features, phi, mask)
    torch.manual_seed(303)
    training_prior = _sample(
        features, phi, mask,
        x1=torch.zeros(features.shape[0], mask.shape[1], 4, dtype=features.dtype),
    )
    assert torch.equal(training_prior, inference_prior)


def test_equal_axis_prior_rfm_lift_lies_on_requested_shell():
    features, phi, mask = _case()
    prior = _sample(features, phi, mask)
    regulator_mass = 0.1
    shell = project_to_shell(prior, regulator_mass) * mask.unsqueeze(-1)
    shell_m2 = normsq4(shell)
    assert torch.allclose(
        shell_m2[mask > 0],
        torch.full_like(shell_m2[mask > 0], regulator_mass**2),
        atol=2e-12,
        rtol=2e-12,
    )
    assert torch.equal(shell[mask == 0], torch.zeros_like(shell[mask == 0]))
