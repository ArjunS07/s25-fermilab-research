"""Heun (RK2) sampler (experiment plan 2.3)."""
import pytest
import torch

from tests.lorentz_test_utils import build_model, sample_inputs


def _scalar(v):
    return torch.tensor(v, dtype=torch.float64)


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_step_shape_and_masking(method):
    model = build_model(use_reference_vectors=True, use_node_scalars=True)
    x, _, cond, mask, refs = sample_inputs()
    out = model.step(x, cond, mask, _scalar(0.0), _scalar(0.1), method=method, ref_vectors=refs)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_heun_matches_manual_rk2():
    """step('heun') == x + dt/2 (v(x,t0) + v(x + dt*v(x,t0), t1)), computed via forward."""
    model = build_model(use_reference_vectors=False, use_node_scalars=True)
    x, _, cond, mask, _ = sample_inputs()
    t0, t1 = _scalar(0.2), _scalar(0.35)
    dt = t1 - t0

    def vel(state, t):
        tb = t.unsqueeze(0).expand(state.shape[0])
        return model(state, tb, cond, mask)

    v1 = vel(x, t0)
    x_euler = x + v1 * dt
    v2 = vel(x_euler, t1)
    expected = x + 0.5 * (v1 + v2) * dt

    got = model.step(x, cond, mask, t0, t1, method="heun")
    assert torch.allclose(got, expected, atol=1e-10)


def test_euler_and_heun_differ_generically():
    """With a non-constant field the two integrators should not coincide."""
    model = build_model(use_reference_vectors=True, use_node_scalars=False)
    x, _, cond, mask, refs = sample_inputs()
    e = model.step(x, cond, mask, _scalar(0.0), _scalar(0.3), method="euler", ref_vectors=refs)
    h = model.step(x, cond, mask, _scalar(0.0), _scalar(0.3), method="heun", ref_vectors=refs)
    assert not torch.allclose(e, h, atol=1e-6)


def test_unknown_sampler_raises():
    model = build_model()
    x, _, cond, mask, _ = sample_inputs()
    with pytest.raises(NotImplementedError):
        model.step(x, cond, mask, _scalar(0.0), _scalar(0.1), method="rk4")
