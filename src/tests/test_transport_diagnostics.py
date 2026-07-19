import pytest
import torch

pytest.importorskip("jetnet")

from experiments.diagnose_mass_shell_checkpoint import paired_endpoint_diagnostics
from util.mass_shell import (project_to_shell, pushforward_to_tangent,
                             tangent_error_diagnostics, exp_map)


def test_paired_endpoint_diagnostic_detects_controlled_angular_motion():
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    prior = torch.tensor([[[1., 1., 0., 0.], [1., .8, .2, .1], [0., 0., 0., 0.]]],
                         dtype=torch.float64)
    generated = prior.clone()
    angle = 0.3
    px, py = generated[..., 1].clone(), generated[..., 2].clone()
    generated[..., 1] = px * torch.cos(torch.tensor(angle)) - py * torch.sin(torch.tensor(angle))
    generated[..., 2] = px * torch.sin(torch.tensor(angle)) + py * torch.cos(torch.tensor(angle))
    report = paired_endpoint_diagnostics(prior, generated, mask)
    assert report["angular_displacement_quantiles"][0] > 0.1
    assert report["spatial_momentum_displacement_quantiles"][0] > 0.1


def test_tangent_diagnostic_exact_zero_and_wrong_fields():
    m = 0.1
    p = project_to_shell(torch.tensor([[[0., 1., .2, .1]]]), m)
    target = pushforward_to_tangent(p, torch.tensor([[[0., .2, -.1, .3]]]), m)
    mask = torch.ones(1, 1)
    exact = tangent_error_diagnostics(p, target, target, mask, m)
    zero = tangent_error_diagnostics(p, torch.zeros_like(target), target, mask, m)
    wrong = -target
    bad = tangent_error_diagnostics(p, wrong, target, mask, m)
    assert exact["loss_zero_fraction"] == 1.0
    assert zero["pred_tangent_norm_quantiles"][0] == 0.0
    assert bad["pred_target_alignment_quantiles"][0] < -0.99
    moved = exp_map(p, wrong * 0.05, m)
    assert not torch.allclose(moved[..., 1:4], p[..., 1:4])
