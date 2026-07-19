import torch

from analysis.analyze_mass_shell_samples import summarize
from util.mass_shell import project_to_shell


def test_sample_summary_counts_invalid_and_preserves_spatial_view():
    samples = project_to_shell(torch.randn(4, 3, 4), 0.3)
    samples[0, 0, 0] = float("nan")
    report = summarize(samples)
    assert report["n_total"] == 4
    assert report["n_invalid"] == 1
    assert report["spatial_momentum_identical_in_massless_view"] is True
