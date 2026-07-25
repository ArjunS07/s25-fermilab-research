import math

import pytest
import torch

from util.aux_calibration import GradientCalibration, shared_backbone_parameters


def test_gradient_calibration_recovers_known_scales_and_cosines():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    calibration = GradientCalibration()
    for _ in range(3):
        calibration.update(
            {
                "base": parameter[0] + 2 * parameter[1],
                "gram": 2 * parameter[0] + 4 * parameter[1],
                "total_momentum": -parameter[1],
            },
            [parameter],
        )

    result = calibration.result()
    assert result["batches"] == 3
    assert result["gradient_rms_norm"]["base"] == pytest.approx(math.sqrt(5))
    assert result["gradient_rms_norm"]["gram"] == pytest.approx(2 * math.sqrt(5))
    assert result["weights"]["gram_only"] == pytest.approx(0.1)
    assert result["weights"]["combined_gram"] == pytest.approx(0.05)
    assert result["gradient_cosine"]["base__gram"] == pytest.approx(1.0)
    assert result["gradient_cosine"]["base__total_momentum"] == pytest.approx(
        -2 / math.sqrt(5)
    )


def test_gradient_calibration_clamps_weights():
    parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    calibration = GradientCalibration()
    calibration.update(
        {
            "base": parameter,
            "gram": parameter * 1e8,
            "total_momentum": parameter * 1e-8,
        },
        [parameter],
    )
    result = calibration.result()
    assert result["weights"]["gram_only"] == 1e-8
    assert result["weights"]["total_momentum_only"] == 10.0


class _ToyTangentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tangent_backbone = torch.nn.Module()
        self.tangent_backbone.body = torch.nn.Linear(2, 2)
        self.tangent_backbone.readout = torch.nn.Linear(2, 1)
        self.other = torch.nn.Linear(2, 2)


def test_shared_backbone_parameters_excludes_readout_and_other_parameters():
    model = _ToyTangentModel()
    selected = shared_backbone_parameters(model)
    assert selected == list(model.tangent_backbone.body.parameters())


def test_gradient_calibration_rejects_zero_base_gradient():
    parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    calibration = GradientCalibration()
    calibration.update(
        {
            "base": parameter * 0,
            "gram": parameter,
            "total_momentum": parameter,
        },
        [parameter],
    )
    with pytest.raises(ValueError, match="base gradient norm is zero"):
        calibration.result()
