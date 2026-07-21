import torch

from util.coordinates import build_reference_vectors
from util.distributions import gen_initial_distribution
from util.minkowski_utils import normsq4


def test_massive_reference_has_requested_invariant_mass():
    eta = torch.tensor([0.3, -0.8])
    pt = torch.tensor([900.0, 1100.0])
    mass = torch.tensor([80.0, 120.0])
    refs = build_reference_vectors(
        eta, pt, final_scale=50.0, device=eta.device,
        jet_phi=torch.tensor([0.2, 1.4]), jet_mass=mass,
    )
    assert torch.allclose(normsq4(refs[:, 1]), (mass / 50.0).square(), atol=2e-5)
    assert torch.equal(refs[:, 0], torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]))


def test_per_jet_prior_matches_conditioned_transverse_momentum():
    torch.manual_seed(19)
    features = torch.tensor([
        [0.2, 1000.0, 70.0, 30.0, 0.0],
        [-0.4, 800.0, 50.0, 18.0, 0.0],
    ])
    scale = 50.0
    prior = gen_initial_distribution(
        batch_size=2, num_particles=30, prior_dist="axis_aligned_per_jet",
        jet_features=features, jet_phi=torch.tensor([0.3, 2.1]), model_scale=scale,
    )
    transverse = prior[..., 1:3].sum(dim=1).norm(dim=-1)
    assert torch.allclose(transverse, features[:, 1] / scale, atol=2e-5, rtol=2e-5)
    assert (prior[..., 0] > 0).all()
