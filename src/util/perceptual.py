"""Auditable perceptual-feature helpers for jet evaluation.

JetNet's public ``fpnd`` compares generated activations with cached global
statistics.  Conditional comparisons instead need activations for both samples;
this module deliberately labels that quantity cFPND.
"""

from __future__ import annotations

import numpy as np
import torch


def frechet_from_features(real, generated) -> float:
    """Fréchet distance between two 2-D feature arrays."""
    from scipy import linalg

    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("Fréchet inputs must be 2-D with matching feature width")
    if min(len(real), len(generated)) < 2:
        raise ValueError("at least two samples are required")
    mu_r, mu_g = real.mean(0), generated.mean(0)
    cov_r = np.cov(real, rowvar=False)
    cov_g = np.cov(generated, rowvar=False)
    covmean = linalg.sqrtm(cov_r.dot(cov_g))
    if not np.isfinite(covmean).all():
        eps = np.eye(cov_r.shape[0]) * 1e-6
        covmean = linalg.sqrtm((cov_r + eps).dot(cov_g + eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_g
    return float(diff.dot(diff) + np.trace(cov_r) + np.trace(cov_g)
                 - 2 * np.trace(covmean))


def particlenet_activations(jets, jet_type="g", device=None, batch_size=64):
    """Return the exact pretrained ParticleNet activations used by JetNet FPND."""
    from torch.utils.data import DataLoader
    from jetnet.datasets import JetNet
    from jetnet.evaluation import gen_metrics

    jets = torch.as_tensor(jets, dtype=torch.float32).clone()
    if jets.ndim != 3 or jets.shape[1:] != (30, 3):
        raise ValueError("ParticleNet jets must have shape [N, 30, 3]")
    JetNet.fpnd_norm(jets, inplace=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if (
        "jetnet" not in gen_metrics.fpnd_dict
        or 30 not in gen_metrics.fpnd_dict["jetnet"]
        or jet_type not in gen_metrics.fpnd_dict["jetnet"][30]
    ):
        gen_metrics._init_fpnd_dict("jetnet", jet_type, 30, 3, device)
    model = gen_metrics.fpnd_dict["jetnet"][30][jet_type]["pnet"].to(device).eval()
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(jets, batch_size=batch_size):
            outputs.append(model(batch.to(device), ret_activations=True).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def conditional_fpnd(real_activations, generated_activations) -> float:
    """Condition-matched ParticleNet Fréchet distance (cFPND)."""
    return frechet_from_features(real_activations, generated_activations)
