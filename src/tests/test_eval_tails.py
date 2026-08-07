import math

import torch

from util.metrics.tail_diagnostics import endpoint_tail_diagnostics


def test_endpoint_tail_diagnostics_separates_nonfinite_and_finite_explosions():
    samples = torch.zeros(5, 2, 4, dtype=torch.float64)
    samples[0, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    samples[1, 0] = torch.tensor([2e3, 2e3, 0.0, 0.0])
    samples[2, 0] = torch.tensor([2e6, 2e6, 0.0, 0.0])
    samples[3, 0, 0] = math.nan
    samples[4, 0, 0] = math.inf

    report = endpoint_tail_diagnostics(samples)
    assert report["n_total"] == 5
    assert report["n_nonfinite"] == 2
    assert report["n_finite_max_abs_gt_1e3"] == 2
    assert report["n_finite_max_abs_gt_1e6"] == 1
    assert report["finite_max_abs_quantiles"]["max"] == 2e6
