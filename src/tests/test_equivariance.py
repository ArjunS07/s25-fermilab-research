"""Blocking equivariance / symmetry-breaking gate (plan 0.2) for H LorentzNet.

No Phase >=1 run is valid until these pass. They encode:
  1. Joint Lorentz equivariance  f(Lx, L refs) = L f(x, refs)  under rotations AND boosts.
  2. Symmetry breaking (THE blocking one): with references on, a particles-only rotation MUST
     change the output — guards against reference invariants silently not reaching the
     scalar/geometry path.
  3. Residual SO(2): rotating particles about the jet axis (refs fixed) rotates the output.
"""
import pytest
import torch

from tests.lorentz_test_utils import (
    build_model,
    sample_inputs,
    apply_transform,
    rotation_4x4,
    random_proper_transform,
)


@pytest.mark.parametrize("transform_seed", [0, 1, 2])
def test_joint_lorentz_equivariance(transform_seed):
    """f(Lx, L refs) == L f(x, refs) under random rotations AND boosts, refs transformed jointly."""
    model = build_model(seed=0)
    x, t, cond, mask, refs = sample_inputs(seed=transform_seed + 5)
    L = random_proper_transform(seed=transform_seed)

    v = model(x, t, cond, mask, ref_vectors=refs)
    Lv = apply_transform(v, L)

    xL = apply_transform(x, L) * mask.unsqueeze(-1)
    refL = apply_transform(refs, L)
    v_of_L = model(xL, t, cond, mask, ref_vectors=refL)

    assert torch.allclose(Lv, v_of_L, atol=1e-5, rtol=1e-5)


def test_symmetry_breaking_particles_only_rotation():
    """BLOCKING: with references on, rotating particles only (refs fixed, off the jet axis)
    must change the output. If this fails, the reference invariants are not reaching the
    scalar/geometry path."""
    model = build_model(seed=0)
    x, t, cond, mask, refs = sample_inputs(jet_axis="z", seed=4)
    R = rotation_4x4("x", 0.7)  # about x, not the jet (z) axis; fixes e_t but not jet_p

    v = model(x, t, cond, mask, ref_vectors=refs)
    xR = apply_transform(x, R) * mask.unsqueeze(-1)
    v_rot = model(xR, t, cond, mask, ref_vectors=refs)  # refs held FIXED

    diff = (v - v_rot).abs().max().item()
    assert diff > 1e-4, f"output barely changed under particles-only rotation (max|Δ|={diff:.2e})"


def test_residual_so2_about_jet_axis():
    """With the jet reference aligned to z, rotating particles about z (refs fixed) rotates the
    output by the same rotation — the residual SO(2) of relative-coordinate jets."""
    model = build_model(seed=0)
    x, t, cond, mask, refs = sample_inputs(jet_axis="z", seed=4)
    Rz = rotation_4x4("z", 0.9)  # fixes e_t and the z-aligned jet reference

    v = model(x, t, cond, mask, ref_vectors=refs)
    Rzv = apply_transform(v, Rz)
    xR = apply_transform(x, Rz) * mask.unsqueeze(-1)
    v_rot = model(xR, t, cond, mask, ref_vectors=refs)

    assert torch.allclose(Rzv, v_rot, atol=1e-5, rtol=1e-5)
