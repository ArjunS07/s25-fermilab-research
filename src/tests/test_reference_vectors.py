"""Shape, wiring, masking, and back-compat checks for reference vectors + node scalars."""
import pytest
import torch

from tests.lorentz_test_utils import build_model, sample_inputs

FLAG_GRID = [
    (False, False),  # A: current model
    (True, False),   # B: references only
    (True, True),    # C: references + node scalars
    (False, True),   # F: node scalars only
]


@pytest.mark.parametrize("use_refs,use_h", FLAG_GRID)
def test_forward_shape_and_finite(use_refs, use_h):
    model = build_model(use_reference_vectors=use_refs, use_node_scalars=use_h)
    x, t, cond, mask, refs = sample_inputs()
    out = model(x, t, cond, mask, ref_vectors=refs if use_refs else None)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("use_refs,use_h", FLAG_GRID)
def test_padding_rows_are_zero(use_refs, use_h):
    model = build_model(use_reference_vectors=use_refs, use_node_scalars=use_h)
    x, t, cond, mask, refs = sample_inputs()
    out = model(x, t, cond, mask, ref_vectors=refs if use_refs else None)
    padded = (mask == 0).unsqueeze(-1).expand_as(out)
    assert torch.allclose(out[padded], torch.zeros_like(out[padded]), atol=1e-12)


def test_reference_flag_requires_vectors():
    model = build_model(use_reference_vectors=True, use_node_scalars=False)
    x, t, cond, mask, _ = sample_inputs()
    with pytest.raises(ValueError):
        model(x, t, cond, mask, ref_vectors=None)


def test_default_model_unchanged_by_ref_argument():
    """With references disabled, passing ref_vectors must be ignored (byte-for-byte run A)."""
    model = build_model(use_reference_vectors=False, use_node_scalars=False)
    x, t, cond, mask, refs = sample_inputs()
    out_none = model(x, t, cond, mask, ref_vectors=None)
    out_refs = model(x, t, cond, mask, ref_vectors=refs)
    assert torch.allclose(out_none, out_refs, atol=1e-12)


def test_seed_features_zero_axis_columns_without_refs():
    """Run F: without references the E and axis seed columns are exactly zero, so only the
    true invariant m^2 seeds h (keeping the model rotation-equivariant)."""
    model = build_model(use_reference_vectors=False, use_node_scalars=True)
    x, _, _, _, _ = sample_inputs()
    feats = model._node_seed_features(x, refs=None)
    assert feats.shape[-1] == 3
    assert torch.allclose(feats[..., 1:], torch.zeros_like(feats[..., 1:]), atol=1e-12)


def test_seed_features_energy_column_matches_energy():
    """With references, the second seed column is psi(E) = psi(<x, e_t>)."""
    from models.LEFT_JeN import psi
    model = build_model(use_reference_vectors=True, use_node_scalars=True)
    x, _, _, _, refs = sample_inputs()
    feats = model._node_seed_features(x, refs=refs)
    assert torch.allclose(feats[..., 1], psi(x[..., 0]), atol=1e-12)


def test_references_change_output_vs_baseline():
    """Turning references on must actually change the velocity field (they are wired in)."""
    x, t, cond, mask, refs = sample_inputs()
    base = build_model(use_reference_vectors=False, use_node_scalars=False, seed=0)
    withref = build_model(use_reference_vectors=True, use_node_scalars=False, seed=0)
    out_base = base(x, t, cond, mask)
    out_ref = withref(x, t, cond, mask, ref_vectors=refs)
    assert not torch.allclose(out_base, out_ref, atol=1e-6)
