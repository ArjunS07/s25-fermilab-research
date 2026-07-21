import torch
import tempfile
from pathlib import Path

from jet_attr_model_v2 import JetAttributeModelV2
from util.jet_attributes import load_model


def test_discrete_multiplicity_sampling_respects_support_and_context():
    model = JetAttributeModelV2(max_particles=30, mixtures=2, hidden=16).eval()
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


def test_v2_attribute_nll_has_finite_gradients():
    model = JetAttributeModelV2(max_particles=30, mixtures=2, hidden=16)
    attrs = torch.tensor([[0.2, 900.0, 70.0, 30.0], [-0.3, 1100.0, 90.0, 18.0]])
    context = torch.zeros(2, 5)
    context[:, 0] = 1
    loss = model.nll(attrs, context)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_v2_state_dict_payload_round_trip():
    model = JetAttributeModelV2(max_particles=30, mixtures=2, hidden=16)
    payload = {
        "format": "jet_attribute_v2_state_dict",
        "config": {"max_particles": 30, "num_types": 5, "mixtures": 2, "hidden": 16},
        "state_dict": model.state_dict(),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pth"
        torch.save(payload, path)
        loaded = load_model(path)
    assert isinstance(loaded, JetAttributeModelV2)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(),
                                                 loaded.state_dict().values()))
