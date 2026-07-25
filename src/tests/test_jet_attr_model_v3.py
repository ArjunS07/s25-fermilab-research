import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("normflows")

from jet_attr_model_v3 import JetAttributeFlowV3
from util.jet_attributes import load_model


def tiny_model():
    return JetAttributeFlowV3(
        max_particles=30, num_flows=2, hidden_layers=1, hidden_units=16
    )


def test_v3_samples_exact_categorical_multiplicity_and_positive_scales():
    model = tiny_model().eval()
    model.multiplicity_probs.zero_()
    model.multiplicity_probs[0, 30] = 1
    model.multiplicity_probs[1, 12] = 1
    context = torch.zeros(20, 5)
    context[:10, 0] = 1
    context[10:, 1] = 1
    attrs, log_prob = model.sample(20, context)
    assert torch.equal(attrs[:10, 3], torch.full((10,), 30.0))
    assert torch.equal(attrs[10:, 3], torch.full((10,), 12.0))
    assert (attrs[:, 1:3] > 0).all()
    assert log_prob.shape == (20,)


def test_v3_nll_has_finite_gradients_in_ratio_coordinates():
    model = tiny_model()
    attrs = torch.tensor([
        [0.2, 900.0, 70.0, 30.0],
        [-0.3, 1100.0, 90.0, 18.0],
    ])
    context = torch.zeros(2, 5)
    context[:, 0] = 1
    loss = model.nll(attrs, context)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.grad is not None
    )


def test_v3_portable_payload_round_trip():
    model = tiny_model()
    payload = {
        "format": "jet_attribute_v3_state_dict",
        "config": model.portable_config(),
        "state_dict": model.state_dict(),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pth"
        torch.save(payload, path)
        loaded = load_model(path)
    assert isinstance(loaded, JetAttributeFlowV3)
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.state_dict().values(), loaded.state_dict().values())
    )
