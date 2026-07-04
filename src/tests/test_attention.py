"""Invariant-logit softmax attention aggregation (experiment plan 3.1).

Covers:
  1. Joint Lorentz equivariance still holds with attention on (invariant logits).
  2. masked_neighbor_softmax: weights sum to 1 over valid neighbors; 0 on diagonal/masked.
  3. Padded-vs-unpadded invariance: padding rows do not change real-particle outputs.
  4. Finite output when a node has no valid neighbor (single-particle jet).
  5. use_attention=False adds no parameters and leaves the baseline path intact.
"""
import pytest
import torch

from models.LEFT_JeN import masked_neighbor_softmax
from tests.lorentz_test_utils import (
    build_model,
    sample_inputs,
    apply_transform,
    rotation_4x4,
    random_proper_transform,
)


@pytest.mark.parametrize("use_refs,use_h", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("transform_seed", [0, 1, 2])
def test_attention_is_lorentz_equivariant(use_refs, use_h, transform_seed):
    """With attention on, f(Lx, Lrefs) == L f(x, refs) — logits are invariant scalars."""
    model = build_model(use_reference_vectors=use_refs, use_node_scalars=use_h,
                        use_attention=True, seed=0)
    x, t, cond, mask, refs = sample_inputs(seed=transform_seed + 5)
    L = random_proper_transform(seed=transform_seed)

    ref_in = refs if use_refs else None
    Lv = apply_transform(model(x, t, cond, mask, ref_vectors=ref_in), L)

    xL = apply_transform(x, L) * mask.unsqueeze(-1)
    refL = apply_transform(refs, L) if use_refs else None
    v_of_L = model(xL, t, cond, mask, ref_vectors=refL)

    assert torch.allclose(Lv, v_of_L, atol=1e-6, rtol=1e-5)


def test_masked_softmax_normalization_and_support():
    """Weights sum to 1 over valid j!=i neighbors; exactly 0 on the diagonal and padded pairs."""
    torch.manual_seed(0)
    B, P = 2, 5
    n_real = 4  # last particle is padding in both batch elements
    logits = torch.randn(B, P, P, dtype=torch.float64)
    mask = torch.zeros(B, P, dtype=torch.float64)
    mask[:, :n_real] = 1.0
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, P, P)

    attn = masked_neighbor_softmax(logits, pair_mask)

    # Diagonal always zero (no self-loops).
    diag = torch.diagonal(attn, dim1=1, dim2=2)
    assert torch.allclose(diag, torch.zeros_like(diag))
    # Padded pairs zero.
    assert torch.allclose(attn * (1 - pair_mask), torch.zeros_like(attn))
    # Each real row sums to 1 over its neighbors (real rows have >= 1 valid neighbor here).
    row_sums = attn.sum(dim=2)
    assert torch.allclose(row_sums[:, :n_real], torch.ones(B, n_real, dtype=torch.float64))
    # Padded rows sum to 0.
    assert torch.allclose(row_sums[:, n_real:], torch.zeros(B, P - n_real, dtype=torch.float64))


def test_no_valid_neighbor_is_zero_and_finite():
    """A single real particle has no j!=i neighbor → its attention row is all zeros, finite."""
    logits = torch.randn(1, 4, 4, dtype=torch.float64)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    attn = masked_neighbor_softmax(logits, pair_mask)
    assert torch.isfinite(attn).all()
    assert torch.allclose(attn, torch.zeros_like(attn))


def test_padded_vs_unpadded_invariance():
    """The real-particle output is unaffected by extra padding rows (masking correctness)."""
    model = build_model(use_attention=True, seed=0)
    x, t, cond, mask, _ = sample_inputs(batch=1, n_real=5, max_particles=8, seed=3)
    # sample_inputs pads batch element 0 to n_real-1 real rows; recompute the true count.
    n_real = int(mask[0].sum().item())

    v_full = model(x, t, cond, mask)

    # Trim the tensors to exactly the real particles (no padding rows at all).
    xt = x[:, :n_real]
    maskt = mask[:, :n_real]
    condt = cond.clone()
    v_trim = model(xt, t, condt, maskt)

    assert torch.allclose(v_full[:, :n_real], v_trim, atol=1e-8)


def test_attention_off_adds_no_params():
    """use_attention=False must not create phi_attn params (baseline stays byte-identical)."""
    off = build_model(use_attention=False, seed=0)
    on = build_model(use_attention=True, seed=0)
    off_keys = [k for k in off.state_dict() if "phi_attn" in k]
    on_keys = [k for k in on.state_dict() if "phi_attn" in k]
    assert off_keys == []
    assert len(on_keys) > 0


def test_attention_changes_output_vs_sigmoid():
    """Attention is a genuinely different aggregation from the sigmoid gate."""
    x, t, cond, mask, _ = sample_inputs(seed=2)
    v_sigmoid = build_model(use_attention=False, seed=0)(x, t, cond, mask)
    v_attn = build_model(use_attention=True, seed=0)(x, t, cond, mask)
    assert not torch.allclose(v_sigmoid, v_attn, atol=1e-6)
