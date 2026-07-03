"""Inference-path checks: references flow through the samplers and the reconstruction helper."""
import pytest
import torch

from tests.lorentz_test_utils import build_model, sample_inputs


def test_step_accepts_reference_vectors():
    model = build_model(use_reference_vectors=True, use_node_scalars=True)
    x, _, cond, mask, refs = sample_inputs()
    t0 = torch.tensor(0.0, dtype=torch.float64)
    t1 = torch.tensor(0.1, dtype=torch.float64)
    x_next = model.step(x, cond, mask, t0, t1, use_cfg=True, ref_vectors=refs)
    assert x_next.shape == x.shape
    assert torch.isfinite(x_next).all()


def test_step_hyperbolic_accepts_reference_vectors():
    model = build_model(use_reference_vectors=True, use_node_scalars=False)
    x, _, cond, mask, refs = sample_inputs()
    from util.hyperbolic import to_poincare_ball
    y = to_poincare_ball(x, c=1.0)
    t0 = torch.tensor(0.0, dtype=torch.float64)
    t1 = torch.tensor(0.1, dtype=torch.float64)
    y_next = model.step_hyperbolic(y, cond, mask, t0, t1, c=1.0, ref_vectors=refs)
    assert y_next.shape == x.shape
    assert torch.isfinite(y_next).all()


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
