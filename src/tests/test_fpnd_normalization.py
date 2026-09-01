"""Scientific gate for the FPND pT-coordinate convention."""

import torch

from util.data.fpnd_input import build_fpnd_input


def test_fpnd_uses_true_jet_pt_and_zeroes_padding():
    # The retained constituents sum to 90 GeV, while the clustered jet is 100 GeV.
    # Dividing by the former recreates the historical FPND artifact.
    eta = torch.tensor([[1.10, 0.90, 1.05, 0.0]])
    phi = torch.tensor([[0.10, -0.10, 0.05, float("nan")]])
    pt = torch.tensor([[40.0, 30.0, 20.0, 0.0]])
    polar_abs = torch.stack((eta, phi, pt, pt * torch.cosh(eta)), dim=-1)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

    output = build_fpnd_input(
        polar_abs, torch.tensor([1.0]), torch.tensor([100.0]), mask
    )

    assert torch.allclose(output[0, :3, 2], torch.tensor([0.4, 0.3, 0.2]))
    assert output[0, :3, 2].sum() < 0.95
    assert torch.isfinite(output).all()
    assert torch.equal(output[0, 3], torch.zeros(3))
