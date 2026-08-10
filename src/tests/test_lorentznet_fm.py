"""Focused gates for the shared LorentzNet Euclidean/RFM ablation."""

from types import SimpleNamespace

import pytest
import torch

from models.lorentznet_flow import build_lorentznet
from tests.lorentz_test_utils import apply_transform, random_proper_transform
from training import flow_matching_loss
from training.euclidean import euclidean_interpolant_and_target
from util.geometry.mass_shell import project_to_shell
from util.geometry.euclidean import euclidean_ode_step
from util.geometry.minkowski_utils import dotsq4


MASS = 1.0


def _model(geometry="euclidean", references="none", seed=17):
    torch.manual_seed(seed)
    return build_lorentznet(
        5, num_layers=2, hidden_dim=24, include_pt=True,
        include_mass_condition=True, regulator_mass=MASS,
        flow_geometry=geometry, reference_mode=references,
    ).double()


def _inputs(on_shell=False):
    torch.manual_seed(29)
    x = torch.randn(2, 5, 4, dtype=torch.float64) * 0.25
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float64)
    if on_shell:
        x = project_to_shell(x, MASS)
        x[0, 3:] = torch.tensor([MASS, 0.0, 0.0, 0.0])
    else:
        x = x * mask.unsqueeze(-1)
    t = torch.tensor([0.2, 0.7], dtype=torch.float64)
    conditions = torch.randn(2, 8, dtype=torch.float64)
    refs = torch.randn(2, 2, 4, dtype=torch.float64)
    refs[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    return x, t, conditions, mask, refs


def test_raw_euclidean_vector_field_is_lorentz_equivariant():
    model = _model().eval()
    x, t, conditions, mask, _ = _inputs()
    transform = random_proper_transform(11)
    field = model.raw_field(x, t, conditions, mask)
    transformed = model.raw_field(apply_transform(x, transform), t, conditions, mask)
    assert torch.allclose(transformed, apply_transform(field, transform), atol=2e-8, rtol=2e-8)


def test_plain_reference_readout_is_jointly_lorentz_equivariant():
    model = _model(references="plain_readout").eval()
    x, t, conditions, mask, refs = _inputs()
    transform = random_proper_transform(12)
    field = model.raw_field(x, t, conditions, mask, refs)
    transformed = model.raw_field(
        apply_transform(x, transform), t, conditions, mask,
        apply_transform(refs, transform),
    )
    assert torch.allclose(transformed, apply_transform(field, transform), atol=2e-8, rtol=2e-8)


def test_rfm_output_is_tangent():
    model = _model("mass_shell", "plain_readout").eval()
    x, t, conditions, mask, refs = _inputs(on_shell=True)
    field = model(x, t, conditions, mask, refs)
    assert torch.allclose(dotsq4(x, field) * mask, torch.zeros_like(mask), atol=2e-10)


def test_rfm_projected_field_is_lorentz_equivariant():
    model = _model("mass_shell", "plain_readout").eval()
    x, t, conditions, mask, refs = _inputs(on_shell=True)
    transform = random_proper_transform(13)
    field = model(x, t, conditions, mask, refs)
    transformed = model(
        apply_transform(x, transform), t, conditions, mask,
        apply_transform(refs, transform),
    )
    assert torch.allclose(transformed, apply_transform(field, transform), atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("references", ["none", "plain_readout"])
def test_permutation_equivariance(references):
    model = _model(references=references).eval()
    x, t, conditions, mask, refs = _inputs()
    refs = refs if references == "plain_readout" else None
    permutation = torch.tensor([2, 4, 0, 3, 1])
    field = model(x, t, conditions, mask, refs)
    permuted = model(x[:, permutation], t, conditions, mask[:, permutation], refs)
    assert torch.allclose(permuted, field[:, permutation], atol=2e-9, rtol=2e-9)


def test_padding_is_zero_and_cannot_affect_real_particles():
    model = _model(references="plain_readout").eval()
    x, t, conditions, mask, refs = _inputs()
    field = model(x, t, conditions, mask, refs)
    changed = x.clone()
    changed[mask == 0] = torch.randn_like(changed[mask == 0]) * 1e5
    changed_field = model(changed, t, conditions, mask, refs)
    assert torch.equal(field[mask == 0], torch.zeros_like(field[mask == 0]))
    assert torch.allclose(changed_field[mask > 0], field[mask > 0], atol=0, rtol=0)


def test_euclidean_interpolation_and_target_are_exact():
    x0 = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    x1 = torch.tensor([[[5.0, 8.0, 11.0, 14.0]]])
    state, target = euclidean_interpolant_and_target(x0, x1, torch.tensor([0.25]))
    assert torch.equal(state, torch.tensor([[[2.0, 3.5, 5.0, 6.5]]]))
    assert torch.equal(target, x1 - x0)


@pytest.mark.parametrize("references", ["none", "plain_readout"])
def test_geometry_variants_share_identical_parameterization(references):
    euclidean = _model("euclidean", references, seed=41)
    rfm = _model("mass_shell", references, seed=41)
    assert list(euclidean.state_dict()) == list(rfm.state_dict())
    assert all(
        left.shape == right.shape
        for left, right in zip(euclidean.state_dict().values(), rfm.state_dict().values())
    )
    assert sum(p.numel() for p in euclidean.parameters()) == sum(
        p.numel() for p in rfm.parameters()
    )


@pytest.mark.parametrize("geometry", ["euclidean", "mass_shell"])
@pytest.mark.parametrize("references", ["none", "plain_readout"])
def test_all_four_variants_have_finite_forward_backward(geometry, references):
    model = _model(geometry, references).train()
    x0, t, conditions, mask, refs = _inputs()
    x1 = torch.randn_like(x0) * mask.unsqueeze(-1)
    refs = refs if references == "plain_readout" else None
    config = SimpleNamespace(flow_geometry=geometry, regulator_mass=MASS)
    loss = flow_matching_loss(
        model=model, raw_model=model, config=config, x0=x0, x1=x1, t=t,
        mask=mask, conditions=conditions, references=refs,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_euclidean_sampling_smoke_has_no_projection_or_repair():
    model = _model("euclidean", "none").eval()
    state, _, conditions, mask, _ = _inputs()
    initial = state.clone()
    times = torch.linspace(0, 1, 5, dtype=torch.float64)
    for start, end in zip(times[:-1], times[1:]):
        state = euclidean_ode_step(
            model, state, conditions, mask, start, end, sampler="euler"
        )
    assert torch.isfinite(state).all()
    assert torch.equal(state[mask == 0], torch.zeros_like(state[mask == 0]))
    # No shell/energy projection: a generic ambient start remains a generic 4-vector.
    assert not torch.allclose(state[..., 0], torch.linalg.vector_norm(state[..., 1:], dim=-1))
    assert not torch.equal(state, initial)


def test_mass_shell_sampling_smoke_stays_on_shell():
    model = _model("mass_shell", "none").eval()
    state, _, conditions, mask, _ = _inputs(on_shell=True)
    stepped = model.step_hyperbolic(
        state, conditions, mask,
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.1, dtype=torch.float64),
        hyperbolic_model="mass_shell", regulator_mass=MASS,
    )
    residual = dotsq4(stepped, stepped) - MASS**2
    assert torch.isfinite(stepped).all()
    assert torch.allclose(residual[mask > 0], torch.zeros_like(residual[mask > 0]), atol=2e-9)


def test_requested_width96_parameter_counts_are_in_budget():
    none = build_lorentznet(5, hidden_dim=96, num_layers=6, reference_mode="none")
    refs = build_lorentznet(5, hidden_dim=96, num_layers=6, reference_mode="plain_readout")
    assert sum(p.numel() for p in none.parameters()) == 441_621
    assert sum(p.numel() for p in refs.parameters()) == 451_319
