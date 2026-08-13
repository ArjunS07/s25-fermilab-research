import torch

from util.infra.grad_stats import collect_gradient_stats


def test_collect_gradient_stats_matches_individual_scalar_reductions():
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.SiLU(),
        torch.nn.Linear(4, 2),
    )
    model(torch.randn(5, 3)).square().sum().backward()

    expected = {}
    for name, param in model.named_parameters():
        weight_norm = param.data.norm(2).item()
        grad_norm = param.grad.norm(2).item()
        grad_mean = param.grad.abs().mean().item()
        expected[name] = {
            "norm": grad_norm,
            "mean": grad_mean,
            "weight_norm": weight_norm,
            "update_ratio": grad_norm / (weight_norm + 1e-8),
        }

    assert collect_gradient_stats(model) == expected


def test_collect_gradient_stats_ignores_parameters_without_gradients():
    model = torch.nn.Linear(2, 2)
    assert collect_gradient_stats(model) == {}
