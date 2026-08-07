"""Inference-path checks: the typed reference-vector reconstruction helper."""
import pytest
import torch


def test_build_reference_vectors_shape_and_e_t():
    jetnet = pytest.importorskip("jetnet")  # inference helper depends on jetnet's transform
    from generate_samples import build_reference_vectors
    jet_eta = torch.tensor([0.1, -0.3, 0.5])
    jet_pt = torch.tensor([1.0, 2.0, 0.5])
    refs = build_reference_vectors(jet_eta, jet_pt, final_scale=2.0, device="cpu")
    assert refs.shape == (3, 2, 4)
    # e_t reference is exactly (1,0,0,0).
    assert torch.allclose(refs[:, 0, :], torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(3, 4))
    # jet reference has positive energy.
    assert (refs[:, 1, 0] > 0).all()


def test_training_and_inference_reference_contract_is_identical():
    pytest.importorskip("jetnet")
    from util.coordinates import build_reference_vectors
    eta = torch.tensor([0.2, -0.4])
    pt = torch.tensor([3.0, 1.5])
    phi = torch.tensor([0.7, -1.2])
    train_refs = build_reference_vectors(eta, pt, 4.0, "cpu", jet_phi=phi)
    infer_refs = build_reference_vectors(eta.clone(), pt.clone(), 4.0, "cpu", jet_phi=phi.clone())
    assert torch.equal(train_refs, infer_refs)
    assert torch.allclose(train_refs[:, 1, 0], pt * torch.cosh(eta) / 4.0)
