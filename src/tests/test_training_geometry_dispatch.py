from types import SimpleNamespace

import torch

from training import flow_matching_loss


class ConstantField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.01))

    def forward(self, x, t, jet_conditions, mask, ref_vectors=None):
        del t, jet_conditions, ref_vectors
        return torch.ones_like(x) * self.scale * mask.unsqueeze(-1)


def _batch():
    torch.manual_seed(17)
    x0 = torch.randn(3, 5, 4) * 0.1
    x1 = torch.randn(3, 5, 4) * 0.1
    mask = torch.ones(3, 5)
    mask[0, -1] = 0
    x0[mask == 0] = 0
    x1[mask == 0] = 0
    return {
        "x0": x0, "x1": x1, "t": torch.tensor([0.2, 0.5, 0.8]),
        "mask": mask, "conditions": torch.randn(3, 8),
        "references": torch.randn(3, 2, 4),
    }


def test_mass_shell_geometry_dispatch_is_finite_and_differentiable():
    model = ConstantField()
    config = SimpleNamespace(
        use_hyperbolic=True, hyperbolic_model="mass_shell",
        regulator_mass=0.3, backbone="tangent_attention",
        train_space="cartesian", sigma_min=1e-4,
    )
    loss = flow_matching_loss(model=model, raw_model=model, config=config, **_batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
