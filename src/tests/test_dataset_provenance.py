import torch

from util.geometry.coordinates import deterministic_jet_phi


def test_deterministic_jet_phi_is_independent_of_global_rng():
    expected = deterministic_jet_phi(16, seed=42)
    torch.manual_seed(999)
    _ = torch.rand(1000)
    actual = deterministic_jet_phi(16, seed=42)
    assert torch.equal(actual, expected)


def test_deterministic_jet_phi_preserves_global_rng_state():
    torch.manual_seed(7)
    before = torch.get_rng_state()
    _ = deterministic_jet_phi(8, seed=42)
    after = torch.get_rng_state()
    assert torch.equal(after, before)


def test_deterministic_jet_phi_seed_is_provenance():
    assert not torch.equal(
        deterministic_jet_phi(8, seed=41),
        deterministic_jet_phi(8, seed=42),
    )
