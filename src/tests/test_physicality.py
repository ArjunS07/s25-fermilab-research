"""Physics underlying the physicality/isotropy diagnostics (no jetnet import needed).

util.metrics imports jetnet at module load, so its helper cannot be imported on a machine
without jetnet. These tests cover the numeric building blocks the helper relies on.
"""
import math

import torch

from util.minkowski_utils import normsq4, relative_mass_shell_residual, spacelike_mask


def test_msq_sign_classifies_particles():
    # timelike (E > |p|): m^2 > 0 ; lightlike (E = |p|): 0 ; spacelike (E < |p|): m^2 < 0
    timelike = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
    lightlike = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    spacelike = torch.tensor([[0.5, 1.0, 0.0, 0.0]])
    assert normsq4(timelike).item() > 0
    assert abs(normsq4(lightlike).item()) < 1e-12
    assert normsq4(spacelike).item() < 0
    assert not spacelike_mask(lightlike).item()
    assert spacelike_mask(spacelike).item()


def test_roundoff_near_light_cone_is_not_called_spacelike():
    spatial = torch.tensor([[1234.567, 2345.678, 3456.789]], dtype=torch.float32)
    energy = torch.linalg.vector_norm(spatial, dim=-1, keepdim=True)
    rounded_lightlike = torch.cat([energy, spatial], dim=-1)
    assert not spacelike_mask(rounded_lightlike).item()
    assert relative_mass_shell_residual(rounded_lightlike).item() < 1e-6


def test_physicality_fractions_on_known_cloud():
    # 3 real particles: 1 negative-energy, 1 spacelike, 1 physical; plus a padded zero row.
    x = torch.tensor([[
        [-1.0, 0.2, 0.0, 0.0],   # E < 0
        [0.3, 1.0, 0.0, 0.0],    # spacelike (m^2 < 0)
        [2.0, 0.5, 0.0, 0.0],    # physical timelike
        [0.0, 0.0, 0.0, 0.0],    # padding
    ]])
    real = x.abs().sum(dim=-1) > 1e-12
    E = x[..., 0]
    msq = normsq4(x)
    frac_neg_E = (E < 0)[real].float().mean().item()
    frac_spacelike = (msq < 0)[real].float().mean().item()
    assert real.sum().item() == 3
    assert abs(frac_neg_E - 1 / 3) < 1e-6
    assert abs(frac_spacelike - 1 / 3) < 1e-6


def _direction(p3):
    norm = p3.norm(dim=-1).clamp(min=1e-12)
    return p3[:, 2] / norm, torch.atan2(p3[:, 1], p3[:, 0])


def test_total_momentum_direction_known_vectors():
    p3 = torch.tensor([
        [0.0, 0.0, 1.0],   # +z  -> cos=1, phi undefined-ish (atan2(0,0)=0)
        [1.0, 0.0, 0.0],   # +x  -> cos=0, phi=0
        [0.0, 1.0, 0.0],   # +y  -> cos=0, phi=pi/2
    ])
    cos, phi = _direction(p3)
    assert torch.allclose(cos, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert abs(phi[1].item() - 0.0) < 1e-6
    assert abs(phi[2].item() - math.pi / 2) < 1e-6
