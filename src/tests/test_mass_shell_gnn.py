import pytest
import torch
from models.mass_shell_gnn import LEFTJeN
from models.mass_shell_gnn import MassShellGNNBlock, Geometry
from tests.lorentz_test_utils import apply_transform, random_proper_transform
from util.geometry.mass_shell import project_to_shell
from util.geometry.minkowski_utils import dotsq4
from config import InferRunConfig
from util.infra.checkpoint_config import resolve_architecture


def inputs(mass=.1):
    torch.manual_seed(71)
    x = project_to_shell(torch.randn(2, 5, 4, dtype=torch.float64), mass)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.float64)
    x[0, 4] = torch.tensor([mass, 0, 0, 0], dtype=torch.float64)
    refs = torch.randn(2, 2, 4, dtype=torch.float64); refs[:, 0] = torch.tensor([1., 0, 0, 0])
    return x, torch.tensor([.2, .8], dtype=torch.float64), torch.randn(2, 8), mask, refs


def model(mass=.1):
    torch.manual_seed(19)
    return LEFTJeN(5, hidden_dim=32, num_layers=2, include_pt=True,
                   use_reference_vectors=True, include_mass_condition=True,
                   regulator_mass=mass).eval()


def test_covariance_permutation_padding_tangency():
    x, t, c, mask, refs = inputs(); net = model(); out = net(x, t, c, mask, refs)
    L = random_proper_transform(22)
    assert torch.allclose(net(apply_transform(x, L), t, c, mask, apply_transform(refs, L)),
                          apply_transform(out, L), atol=3e-6, rtol=3e-6)
    p = torch.tensor([2, 0, 3, 1, 4])
    assert torch.allclose(net(x[:, p], t, c, mask[:, p], refs), out[:, p], atol=3e-6, rtol=3e-6)
    assert torch.equal(out[mask == 0], torch.zeros_like(out[mask == 0]))
    assert torch.allclose(dotsq4(x, out) * mask, torch.zeros_like(mask), atol=1e-9)


@pytest.mark.parametrize("mass", [.1, .03])
def test_finite_forward_backward_and_backbone_gradients(mass):
    x, t, c, mask, refs = inputs(mass); net = model(mass).train()
    loss = net(x, t, c, mask, refs).square().sum(); loss.backward()
    assert torch.isfinite(loss)
    assert net.tangent_backbone.blocks[0].edge_mlp[0].weight.grad.abs().sum() > 0
    assert net.tangent_backbone.blocks[0].node_mlp[0].weight.grad.abs().sum() > 0


def test_signed_sum_is_not_softmax_or_convex_average():
    """Message aggregation is a signed sum / sqrt(degree), not a softmax convex average:
    a constant negative message aggregates to a value below any single message."""
    block = MassShellGNNBlock(4)
    with torch.no_grad(): block.edge_mlp[-1].weight.zero_(); block.edge_mlp[-1].bias.fill_(-2)
    captured = {}
    def hook(_, args): captured["x"] = args[0]
    handle = block.node_mlp.register_forward_pre_hook(hook)
    support = ~torch.eye(3, dtype=torch.bool).unsqueeze(0)
    count = support.sum(-1).clamp_min(1).to(torch.float64).sqrt()
    g = Geometry(x=torch.ones(1, 3, 4, dtype=torch.float64), cond=torch.zeros(1, 4),
                 mask=torch.ones(1, 3), edge=torch.zeros(1, 3, 3, 3),
                 direction=torch.zeros(1, 3, 3, 4, dtype=torch.float64),
                 projected_refs=torch.zeros(1, 3, 2, 4, dtype=torch.float64),
                 typed_refs=torch.zeros(1, 2, 4), support=support, count=count,
                 count_model=count.to(torch.float32), mass=.1)
    block(torch.zeros(1, 3, 4), g)
    handle.remove()
    # 2 neighbours × (-2) / sqrt(2) = -2*sqrt(2): a signed sum, below the min single message (-2).
    assert torch.allclose(captured["x"][..., 4:8], torch.full((1, 3, 4), -2 * 2 ** .5))


def test_checkpoint_reconstruction():
    """Architecture reconstructed from an embedded full_config reproduces the model exactly."""
    a = model()
    model_config = {
        "n_hidden": 32, "n_layers": 2, "regulator_mass": .1,
        "use_reference_vectors": True, "include_mass_condition": True,
    }
    model_cfg, _ = resolve_architecture(InferRunConfig().model,
                                        {"full_config": {"model": model_config}})
    b = LEFTJeN(5, hidden_dim=model_cfg.n_hidden, num_layers=model_cfg.n_layers,
                include_pt=True, use_reference_vectors=model_cfg.use_reference_vectors,
                include_mass_condition=model_cfg.include_mass_condition,
                regulator_mass=model_cfg.regulator_mass)
    b.load_state_dict(a.state_dict())
    x, t, c, mask, refs = inputs()
    assert torch.equal(a(x, t, c, mask, refs), b(x, t, c, mask, refs))
