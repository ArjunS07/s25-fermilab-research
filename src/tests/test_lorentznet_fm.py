"""Focused gates for the shared LorentzNet Euclidean/RFM ablation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.lorentznet_flow import LorentzNetLGEB, build_lorentznet, signed_log
from tests.lorentz_test_utils import apply_transform, random_proper_transform
from training import flow_matching_loss
from training.euclidean import euclidean_interpolant_and_target
from util.geometry.mass_shell import log_map, project_to_shell, pushforward_to_tangent
from util.geometry.euclidean import euclidean_ode_step
from util.geometry.minkowski_utils import dotsq4, normsq4


MASS = 1.0


def _model(
    geometry="euclidean",
    references="none",
    seed=17,
    activate_head=True,
    scalar_init_mode="normsq",
    particle_readout_mode="ambient",
    geometry_mode="evolving_auxiliary",
    field_degree_normalization="none",
    inject_condition_time_each_block=False,
):
    torch.manual_seed(seed)
    model = build_lorentznet(
        5, num_layers=2, hidden_dim=24, include_pt=True,
        include_mass_condition=True, regulator_mass=MASS,
        flow_geometry=geometry, reference_mode=references,
        scalar_init_mode=scalar_init_mode,
        particle_readout_mode=particle_readout_mode,
        geometry_mode=geometry_mode,
        field_degree_normalization=field_degree_normalization,
        inject_condition_time_each_block=inject_condition_time_each_block,
    ).double()
    if activate_head:
        # The production field starts at exactly zero.  Activate its invariant
        # coefficient heads here so the covariance tests exercise nonzero vectors.
        generator = torch.Generator().manual_seed(seed + 10_000)
        with torch.no_grad():
            model.lorentznet_backbone.field_mlp[-1].weight.normal_(
                std=1e-3, generator=generator
            )
            if references != "none":
                model.lorentznet_backbone.reference_mlp[-1].weight.normal_(
                    std=1e-3, generator=generator
                )
    return model


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


@pytest.mark.parametrize("references", ["none", "plain_readout"])
def test_field_heads_use_small_normal_weights_and_zero_biases(references):
    model = _model(references=references, activate_head=False).eval()
    backbone = model.lorentznet_backbone
    heads = [backbone.field_mlp[-1]]
    if references != "none":
        heads.append(backbone.reference_mlp[-1])
    for head in heads:
        assert torch.count_nonzero(head.weight) > 0
        assert 2e-4 < head.weight.std() < 2e-3
        assert torch.equal(head.bias, torch.zeros_like(head.bias))


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


def test_reference_contraction_scalar_init_matches_requested_formula():
    model = _model(
        "mass_shell", "none", activate_head=False,
        scalar_init_mode="reference_contractions",
    ).eval()
    x, t, conditions, mask, refs = _inputs(on_shell=True)
    y = x * mask.unsqueeze(-1)
    backbone = model.lorentznet_backbone
    actual = backbone.initial_scalar_state(y, t, conditions, mask, refs)
    contractions = signed_log(dotsq4(y.unsqueeze(2), refs.unsqueeze(1)))
    time_features = torch.stack(
        (t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)), dim=-1
    )
    expected = (
        backbone.node_seed(contractions)
        - backbone.time_embed(time_features).unsqueeze(1)
        + backbone.condition_embed(conditions).unsqueeze(1)
    ) * mask.unsqueeze(-1)
    assert torch.equal(actual, expected)


def test_reference_contraction_scalar_state_is_jointly_lorentz_invariant():
    model = _model(
        "mass_shell", "none", activate_head=False,
        scalar_init_mode="reference_contractions",
    ).eval()
    x, t, conditions, mask, refs = _inputs(on_shell=True)
    y = x * mask.unsqueeze(-1)
    transform = random_proper_transform(14)
    backbone = model.lorentznet_backbone
    initial = backbone.initial_scalar_state(y, t, conditions, mask, refs)
    transformed = backbone.initial_scalar_state(
        apply_transform(y, transform), t, conditions, mask,
        apply_transform(refs, transform),
    )
    assert torch.allclose(transformed, initial, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("references", ["none", "plain_readout"])
def test_reference_contraction_rfm_field_is_jointly_equivariant(references):
    model = _model(
        "mass_shell", references,
        scalar_init_mode="reference_contractions",
    ).eval()
    x, t, conditions, mask, refs = _inputs(on_shell=True)
    transform = random_proper_transform(15)
    field = model(x, t, conditions, mask, refs)
    transformed = model(
        apply_transform(x, transform), t, conditions, mask,
        apply_transform(refs, transform),
    )
    assert torch.allclose(transformed, apply_transform(field, transform), atol=3e-8, rtol=3e-8)


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


class _ConstantMessages(nn.Module):
    def __init__(self, width, value):
        super().__init__()
        self.width = width
        self.value = value

    def forward(self, inputs):
        return torch.full(
            (*inputs.shape[:-1], self.width), self.value,
            dtype=inputs.dtype, device=inputs.device,
        )


class _CaptureUpdate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs.detach().clone()
        return torch.zeros_like(inputs[..., :self.width])


def test_lgeb_uses_signed_sqrt_degree_sum_without_message_gate():
    width = 4
    block = LorentzNetLGEB(width, geometry_mode="fixed_physical_geodesic").double()
    assert not hasattr(block, "message_gate")
    assert block.node_mlp[0].in_features == 2 * width
    assert block.node_mlp[0].out_features == 2 * width
    assert block.node_mlp[-1].in_features == 2 * width
    assert block.node_mlp[-1].out_features == width
    block.message_mlp = _ConstantMessages(width, -2.0)
    capture = _CaptureUpdate(width)
    block.node_mlp = capture
    h = torch.zeros(1, 3, width, dtype=torch.float64)
    y = project_to_shell(torch.randn(1, 3, 4, dtype=torch.float64), MASS)
    mask = torch.ones(1, 3, dtype=torch.float64)
    block(h, y, mask, fixed_edge=torch.zeros(1, 3, 3, 3, dtype=torch.float64))
    aggregate = capture.inputs[..., width:]
    assert torch.allclose(
        aggregate,
        torch.full_like(aggregate, -2.0 * 2.0**0.5),
        atol=1e-12, rtol=1e-12,
    )


def test_lgeb_cached_support_matches_internal_support_bitwise():
    torch.manual_seed(29)
    width = 8
    block = LorentzNetLGEB(width).double().eval()
    h = torch.randn(2, 5, width, dtype=torch.float64)
    y = torch.randn(2, 5, 4, dtype=torch.float64)
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.float64)
    real = mask.bool()
    support = (
        real.unsqueeze(2)
        & real.unsqueeze(1)
        & ~torch.eye(5, dtype=torch.bool).unsqueeze(0)
    )
    support_f = support.unsqueeze(-1).to(h.dtype)
    sqrt_degree = support.sum(2).clamp_min(1).to(h.dtype).sqrt().unsqueeze(-1)

    expected_h, expected_y = block(h, y, mask)
    actual_h, actual_y = block(
        h, y, mask,
        support=support,
        support_f=support_f,
        sqrt_degree=sqrt_degree,
    )

    assert torch.equal(actual_h, expected_h)
    assert torch.equal(actual_y, expected_y)


def test_output_support_excludes_self_terms():
    model = _model(references="none").eval()
    x = torch.tensor([[[1.2, 0.3, 0.4, 0.5]]], dtype=torch.float64)
    t = torch.tensor([0.2], dtype=torch.float64)
    conditions = torch.randn(1, 8, dtype=torch.float64)
    mask = torch.ones(1, 1, dtype=torch.float64)
    assert torch.equal(model.raw_field(x, t, conditions, mask), torch.zeros_like(x))


class _UnitCoefficients(nn.Module):
    def __init__(self, outputs=1):
        super().__init__()
        self.outputs = outputs

    def forward(self, inputs):
        return torch.ones(
            (*inputs.shape[:-1], self.outputs),
            dtype=inputs.dtype, device=inputs.device,
        )


def test_normalized_logmap_readout_matches_formula_and_sqrt_degree():
    model = _model(
        "mass_shell", "none", geometry_mode="fixed_physical_geodesic",
        particle_readout_mode="normalized_logmap",
        field_degree_normalization="sqrt",
    ).eval()
    x, t, conditions, mask, _ = _inputs(on_shell=True)
    model.lorentznet_backbone.field_mlp = _UnitCoefficients()
    actual = model.raw_field(x, t, conditions, mask)
    relative, distance = log_map(
        x.unsqueeze(2), x.unsqueeze(1), MASS, return_distance=True
    )
    n = x.shape[1]
    support = (
        mask.bool().unsqueeze(2) & mask.bool().unsqueeze(1)
        & ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
    )
    expected = (
        (relative / (MASS + distance) * support.unsqueeze(-1)).sum(2)
        / support.sum(2).clamp_min(1).sqrt().unsqueeze(-1)
    ) * mask.unsqueeze(-1)
    assert torch.allclose(actual, expected, atol=1e-8, rtol=1e-8)
    assert torch.allclose(dotsq4(x, actual) * mask, torch.zeros_like(mask), atol=2e-10)


def test_normalized_tangent_reference_readout_matches_formula():
    model = _model(
        "mass_shell", "normalized_tangent_readout",
        particle_readout_mode="normalized_logmap",
        field_degree_normalization="sqrt",
    ).eval()
    x, t, conditions, mask, refs = _inputs(on_shell=True)
    with torch.no_grad():
        model.lorentznet_backbone.field_mlp[-1].weight.zero_()
        model.lorentznet_backbone.field_mlp[-1].bias.zero_()
    model.lorentznet_backbone.reference_mlp = _UnitCoefficients(outputs=2)
    actual = model.raw_field(x, t, conditions, mask, refs)
    expanded = refs.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
    projected = pushforward_to_tangent(x.unsqueeze(2), expanded, MASS)
    reference_norm = torch.sqrt((-normsq4(projected)).clamp_min(0))
    expected = (projected / (MASS + reference_norm).unsqueeze(-1)).sum(2)
    expected = expected * mask.unsqueeze(-1)
    assert torch.allclose(actual, expected, atol=2e-10, rtol=2e-10)
    assert torch.allclose(dotsq4(x, actual) * mask, torch.zeros_like(mask), atol=2e-10)


def test_fixed_physical_geometry_has_no_auxiliary_coordinate_heads():
    model = _model(
        "mass_shell", "plain_readout",
        geometry_mode="fixed_physical_geodesic",
        particle_readout_mode="normalized_logmap",
        field_degree_normalization="sqrt",
    )
    assert all(
        not hasattr(block, "coordinate_mlp")
        for block in model.lorentznet_backbone.blocks
    )


def test_condition_time_injection_expands_every_block_inputs():
    model = _model(
        "mass_shell", "normalized_tangent_readout",
        scalar_init_mode="reference_contractions",
        particle_readout_mode="normalized_logmap",
        field_degree_normalization="sqrt",
        inject_condition_time_each_block=True,
    )
    for block in model.lorentznet_backbone.blocks:
        assert block.inject_condition_time is True
        assert block.message_mlp[0].in_features == 3 * 24 + 2
        assert block.node_mlp[0].in_features == 3 * 24


def test_condition_time_injection_factorial_is_equivariant_finite_and_on_shell():
    model = _model(
        "mass_shell", "normalized_tangent_readout",
        scalar_init_mode="reference_contractions",
        particle_readout_mode="normalized_logmap",
        field_degree_normalization="sqrt",
        inject_condition_time_each_block=True,
    ).train()
    x, t, conditions, mask, refs = _inputs(on_shell=True)
    transform = random_proper_transform(91)
    field = model(x, t, conditions, mask, refs)
    transformed = model(
        apply_transform(x, transform), t, conditions, mask,
        apply_transform(refs, transform),
    )
    assert torch.allclose(
        transformed, apply_transform(field, transform), atol=5e-8, rtol=5e-8
    )
    loss = (field.square() * mask.unsqueeze(-1)).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    model.eval()
    stepped = model.step_hyperbolic(
        x, conditions, mask,
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.05, dtype=torch.float64),
        ref_vectors=refs,
        hyperbolic_model="mass_shell", regulator_mass=MASS,
    )
    residual = dotsq4(stepped, stepped) - MASS**2
    assert torch.isfinite(stepped).all()
    assert torch.allclose(
        residual[mask > 0], torch.zeros_like(residual[mask > 0]), atol=2e-9
    )


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


@pytest.mark.parametrize("references", ["none", "plain_readout"])
def test_reference_contraction_rfm_forward_backward_and_sampling(references):
    model = _model(
        "mass_shell", references,
        scalar_init_mode="reference_contractions",
    ).train()
    x0, t, conditions, mask, refs = _inputs()
    x1 = torch.randn_like(x0) * mask.unsqueeze(-1)
    loss = flow_matching_loss(
        model=model,
        raw_model=model,
        config=SimpleNamespace(flow_geometry="mass_shell", regulator_mass=MASS),
        x0=x0,
        x1=x1,
        t=t,
        mask=mask,
        conditions=conditions,
        references=refs,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    model.eval()
    shell = project_to_shell(x0, MASS)
    stepped = model.step_hyperbolic(
        shell,
        conditions,
        mask,
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.1, dtype=torch.float64),
        ref_vectors=refs,
        hyperbolic_model="mass_shell",
        regulator_mass=MASS,
    )
    residual = dotsq4(stepped, stepped) - MASS**2
    assert torch.isfinite(stepped).all()
    assert torch.allclose(
        residual[mask > 0], torch.zeros_like(residual[mask > 0]), atol=2e-9
    )


@pytest.mark.parametrize(
    "reference_mode,geometry_mode",
    [
        ("plain_readout", "evolving_auxiliary"),
        ("normalized_tangent_readout", "evolving_auxiliary"),
        ("plain_readout", "fixed_physical_geodesic"),
        ("normalized_tangent_readout", "fixed_physical_geodesic"),
    ],
)
def test_logmap_factorial_is_equivariant_finite_and_samples_on_shell(
    reference_mode, geometry_mode
):
    model = _model(
        "mass_shell", reference_mode,
        scalar_init_mode="reference_contractions",
        particle_readout_mode="normalized_logmap",
        geometry_mode=geometry_mode,
        field_degree_normalization="sqrt",
    ).train()
    x, t, conditions, mask, refs = _inputs(on_shell=True)
    transform = random_proper_transform(81)
    field = model(x, t, conditions, mask, refs)
    transformed = model(
        apply_transform(x, transform), t, conditions, mask,
        apply_transform(refs, transform),
    )
    assert torch.allclose(
        transformed, apply_transform(field, transform), atol=5e-8, rtol=5e-8
    )
    loss = (field.square() * mask.unsqueeze(-1)).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    model.eval()
    stepped = model.step_hyperbolic(
        x, conditions, mask,
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.05, dtype=torch.float64),
        ref_vectors=refs,
        hyperbolic_model="mass_shell",
        regulator_mass=MASS,
    )
    residual = dotsq4(stepped, stepped) - MASS**2
    assert torch.isfinite(stepped).all()
    assert torch.allclose(
        residual[mask > 0], torch.zeros_like(residual[mask > 0]), atol=2e-9
    )


def test_requested_width96_parameter_counts_are_in_budget():
    none = build_lorentznet(5, hidden_dim=96, num_layers=6, reference_mode="none")
    refs = build_lorentznet(5, hidden_dim=96, num_layers=6, reference_mode="plain_readout")
    assert sum(p.numel() for p in none.parameters()) == 607_503
    assert sum(p.numel() for p in refs.parameters()) == 617_201

    contraction_none = build_lorentznet(
        5, hidden_dim=96, num_layers=6, reference_mode="none",
        scalar_init_mode="reference_contractions",
    )
    contraction_refs = build_lorentznet(
        5, hidden_dim=96, num_layers=6, reference_mode="plain_readout",
        scalar_init_mode="reference_contractions",
    )
    assert sum(p.numel() for p in contraction_none.parameters()) == 607_599
    assert sum(p.numel() for p in contraction_refs.parameters()) == 617_297

    for reference_mode in ("plain_readout", "normalized_tangent_readout"):
        evolving = build_lorentznet(
            5, hidden_dim=96, num_layers=6, flow_geometry="mass_shell",
            reference_mode=reference_mode,
            scalar_init_mode="reference_contractions",
            particle_readout_mode="normalized_logmap",
            geometry_mode="evolving_auxiliary",
            field_degree_normalization="sqrt",
        )
        fixed = build_lorentznet(
            5, hidden_dim=96, num_layers=6, flow_geometry="mass_shell",
            reference_mode=reference_mode,
            scalar_init_mode="reference_contractions",
            particle_readout_mode="normalized_logmap",
            geometry_mode="fixed_physical_geodesic",
            field_degree_normalization="sqrt",
        )
        assert sum(p.numel() for p in evolving.parameters()) == 617_297
        assert sum(p.numel() for p in fixed.parameters()) == 561_515

    block_context = build_lorentznet(
        5, hidden_dim=96, num_layers=6, flow_geometry="mass_shell",
        reference_mode="normalized_tangent_readout",
        scalar_init_mode="reference_contractions",
        particle_readout_mode="normalized_logmap",
        geometry_mode="evolving_auxiliary",
        field_degree_normalization="sqrt",
        inject_condition_time_each_block=True,
    )
    assert sum(p.numel() for p in block_context.parameters()) == 783_185
