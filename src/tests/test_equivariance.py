"""Blocking equivariance / symmetry-breaking gate (plan 0.2).

No Phase >=1 run is valid until these pass. They encode:
  1. Joint Lorentz equivariance  f(Lx,...,Lrefs) = L f(x,...,refs)  (all flag settings).
  2. Symmetry breaking (THE blocking one): with refs on, a particles-only rotation MUST
     change the output — guards against silently dropping ref invariants (reproducing run A).
  3. Residual SO(2): rotating particles about the jet axis (refs fixed) rotates the output.
  4. Run-F guard: refs off + node scalars on stays exactly rotation-equivariant.
"""
import math

import pytest
import torch

from tests.lorentz_test_utils import (
    build_model,
    sample_inputs,
    apply_transform,
    rotation_4x4,
    boost_4x4,
    random_proper_transform,
)

FLAG_GRID = [
    (False, False),  # A
    (True, False),   # B
    (True, True),    # C
    (False, True),   # F
]


@pytest.mark.parametrize("use_refs,use_h", FLAG_GRID)
@pytest.mark.parametrize("transform_seed", [0, 1, 2])
def test_joint_lorentz_equivariance(use_refs, use_h, transform_seed):
    """f(Lx, Lrefs) == L f(x, refs) for random rotations AND boosts, refs transformed jointly."""
    model = build_model(use_reference_vectors=use_refs, use_node_scalars=use_h, seed=0)
    x, t, cond, mask, refs = sample_inputs(seed=transform_seed + 5)
    L = random_proper_transform(seed=transform_seed)

    ref_in = refs if use_refs else None
    v = model(x, t, cond, mask, ref_vectors=ref_in)
    Lv = apply_transform(v, L)

    xL = apply_transform(x, L) * mask.unsqueeze(-1)
    refL = apply_transform(refs, L) if use_refs else None
    v_of_L = model(xL, t, cond, mask, ref_vectors=refL)

    assert torch.allclose(Lv, v_of_L, atol=1e-6, rtol=1e-5)


def test_symmetry_breaking_particles_only_rotation():
    """BLOCKING: with references on, rotating particles only (refs fixed, off the jet axis)
    must change the output. If this fails, reference invariants are not reaching the scalar/
    displacement path and the model has silently reproduced run A."""
    model = build_model(use_reference_vectors=True, use_node_scalars=True, seed=0)
    x, t, cond, mask, refs = sample_inputs(jet_axis="z", seed=4)
    R = rotation_4x4("x", 0.7)  # about x, not the jet (z) axis

    v = model(x, t, cond, mask, ref_vectors=refs)
    xR = apply_transform(x, R) * mask.unsqueeze(-1)
    v_rot = model(xR, t, cond, mask, ref_vectors=refs)  # refs held FIXED

    diff = (v - v_rot).abs().max().item()
    assert diff > 1e-4, f"output barely changed under particles-only rotation (max|Δ|={diff:.2e})"


def test_residual_so2_about_jet_axis():
    """With the jet reference aligned to z, rotating particles about z (refs fixed) rotates
    the output by the same rotation — the residual symmetry of relative-coordinate jets."""
    model = build_model(use_reference_vectors=True, use_node_scalars=True, seed=0)
    x, t, cond, mask, refs = sample_inputs(jet_axis="z", seed=4)
    Rz = rotation_4x4("z", 0.9)  # fixes e_t and the z-aligned jet reference

    v = model(x, t, cond, mask, ref_vectors=refs)
    Rzv = apply_transform(v, Rz)
    xR = apply_transform(x, Rz) * mask.unsqueeze(-1)
    v_rot = model(xR, t, cond, mask, ref_vectors=refs)

    assert torch.allclose(Rzv, v_rot, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("transform_seed", [0, 1, 2])
def test_run_f_stays_rotation_equivariant(transform_seed):
    """Run F (refs off, node scalars on): still exactly rotation-equivariant, since h is
    seeded from m^2 only."""
    model = build_model(use_reference_vectors=False, use_node_scalars=True, seed=0)
    x, t, cond, mask, _ = sample_inputs(seed=transform_seed + 5)
    R = rotation_4x4("y", 0.5 + 0.3 * transform_seed)

    v = model(x, t, cond, mask)
    Rv = apply_transform(v, R)
    xR = apply_transform(x, R) * mask.unsqueeze(-1)
    v_of_R = model(xR, t, cond, mask)

    assert torch.allclose(Rv, v_of_R, atol=1e-6, rtol=1e-5)
