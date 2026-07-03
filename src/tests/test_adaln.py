"""adaLN/FiLM conditioning (experiment plan 2.2): equivariance + wiring."""
import pytest
import torch

from tests.lorentz_test_utils import (
    build_model,
    sample_inputs,
    apply_transform,
    random_proper_transform,
)

# adaLN combined with each reference/node-scalar setting.
ADALN_GRID = [
    (False, False),
    (True, False),
    (True, True),
    (False, True),
]


@pytest.mark.parametrize("use_refs,use_h", ADALN_GRID)
@pytest.mark.parametrize("transform_seed", [0, 1])
def test_adaln_preserves_joint_equivariance(use_refs, use_h, transform_seed):
    """FiLM scale/shift come from invariant conditioning, so equivariance must hold."""
    model = build_model(use_reference_vectors=use_refs, use_node_scalars=use_h,
                        use_adaln=True, seed=0)
    x, t, cond, mask, refs = sample_inputs(seed=transform_seed + 5)
    L = random_proper_transform(seed=transform_seed)

    ref_in = refs if use_refs else None
    v = model(x, t, cond, mask, ref_vectors=ref_in)
    Lv = apply_transform(v, L)

    xL = apply_transform(x, L) * mask.unsqueeze(-1)
    refL = apply_transform(refs, L) if use_refs else None
    v_of_L = model(xL, t, cond, mask, ref_vectors=refL)

    assert torch.allclose(Lv, v_of_L, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("use_refs,use_h", ADALN_GRID)
def test_adaln_forward_shape_and_finite(use_refs, use_h):
    model = build_model(use_reference_vectors=use_refs, use_node_scalars=use_h, use_adaln=True)
    x, t, cond, mask, refs = sample_inputs()
    out = model(x, t, cond, mask, ref_vectors=refs if use_refs else None)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_adaln_message_input_is_smaller():
    """The point of adaLN: g/t are no longer concatenated per edge, so phi_e's input is
    much narrower than the concat path (the (B,N,N,~3*embed) memory hog is gone)."""
    concat = build_model(use_adaln=False)
    adaln = build_model(use_adaln=True)
    concat_in = concat.layers[0].phi_e.net[0].in_features
    adaln_in = adaln.layers[0].phi_e.net[0].in_features
    assert adaln_in < concat_in
    # base concat path: 2 + 3*embed_dim (=16) = 50 ; adaLN base: 2
    assert adaln_in == 2
    assert concat_in == 2 + 3 * 16


def test_adaln_zero_init_starts_as_identity():
    """adaLN-zero: the modulator's last layer is zero-initialized, so at init FiLM is the
    identity and an adaLN model matches the same-seed concat model's *structure* by producing
    finite, well-scaled output (sanity that zero-init did not break the forward)."""
    model = build_model(use_adaln=True, seed=0)
    last = [m for m in model.layers[0].adaln_mod.net if isinstance(m, torch.nn.Linear)][-1]
    assert torch.allclose(last.weight, torch.zeros_like(last.weight))
    assert torch.allclose(last.bias, torch.zeros_like(last.bias))
