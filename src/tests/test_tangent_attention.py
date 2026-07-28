import torch

from models.LEFT_JeN import LEFTJeN
from tests.lorentz_test_utils import apply_transform, random_proper_transform
from util.mass_shell import parallel_transport, project_to_shell
from util.minkowski_utils import dotsq4, normsq4


def _inputs(dtype=torch.float64):
    torch.manual_seed(8)
    batch, particles, mass = 2, 6, 0.3
    x = project_to_shell(torch.randn(batch, particles, 4, dtype=dtype), mass)
    mask = torch.ones(batch, particles, dtype=dtype)
    mask[0, -1] = 0
    x[0, -1] = project_to_shell(torch.zeros(4, dtype=dtype), mass)
    cond = torch.randn(batch, 8, dtype=dtype)
    refs = torch.randn(batch, 2, 4, dtype=dtype)
    refs[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    t = torch.tensor([0.2, 0.7], dtype=dtype)
    return x, t, cond, mask, refs, mass


def _model(mass):
    torch.manual_seed(4)
    return LEFTJeN(
        max_num_jet_types=5, max_particles=6, hidden_dim=32, num_layers=2,
        include_pt=True, use_reference_vectors=True, backbone="tangent_attention",
        include_mass_condition=True, num_attention_heads=4, vector_channels=4,
        regulator_mass=mass,
    ).eval()


def test_parallel_transport_is_tangent_and_norm_preserving():
    p = project_to_shell(torch.randn(12, 4), 0.3)
    q = project_to_shell(torch.randn(12, 4), 0.3)
    raw = torch.randn(12, 4, dtype=torch.float64)
    v = raw - (dotsq4(p, raw) / 0.3**2).unsqueeze(-1) * p
    transported = parallel_transport(p, q, v, 0.3)
    assert torch.allclose(dotsq4(q, transported), torch.zeros(12, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(normsq4(v), normsq4(transported), atol=1e-8, rtol=1e-8)


def test_tangent_attention_covariance_tangency_and_padding():
    x, t, cond, mask, refs, mass = _inputs()
    model = _model(mass)
    transform = random_proper_transform(seed=12)
    out = model(x, t, cond, mask, refs)
    transformed = model(
        apply_transform(x, transform), t, cond, mask,
        apply_transform(refs, transform),
    )
    assert torch.allclose(transformed, apply_transform(out, transform), atol=2e-6, rtol=2e-6)
    assert torch.allclose(dotsq4(x, out) * mask, torch.zeros_like(mask), atol=1e-9)
    assert torch.equal(out[mask == 0], torch.zeros_like(out[mask == 0]))


def test_tangent_attention_particle_permutation_equivariance():
    x, t, cond, mask, refs, mass = _inputs()
    model = _model(mass)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    expected = model(x, t, cond, mask, refs)[:, permutation]
    actual = model(x[:, permutation], t, cond, mask[:, permutation], refs)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_reference_roles_are_not_exchangeable():
    x, t, cond, mask, refs, mass = _inputs()
    model = _model(mass)
    original = model(x, t, cond, mask, refs)
    swapped = model(x, t, cond, mask, refs[:, [1, 0]])
    assert not torch.allclose(original, swapped, atol=1e-6)


def test_mass_condition_changes_output():
    x, t, cond, mask, refs, mass = _inputs()
    model = _model(mass)
    changed = cond.clone()
    changed[:, -1] += 1.0
    assert not torch.allclose(
        model(x, t, cond, mask, refs), model(x, t, changed, mask, refs), atol=1e-7
    )


def test_zero_readout_initialization_produces_zero_initial_velocity():
    x, t, cond, mask, refs, mass = _inputs()
    torch.manual_seed(4)
    zero_model = LEFTJeN(
        max_num_jet_types=5, max_particles=6, hidden_dim=32, num_layers=2,
        include_pt=True, use_reference_vectors=True, backbone="tangent_attention",
        include_mass_condition=True, num_attention_heads=4, vector_channels=4,
        regulator_mass=mass, velocity_readout_init="zero",
    ).eval()
    assert torch.equal(zero_model(x, t, cond, mask, refs), torch.zeros_like(x))
    assert torch.count_nonzero(zero_model.tangent_backbone.readout.weight) == 0


def test_tangent_sampler_accepts_generation_keyword_contract():
    x, _, cond, mask, refs, mass = _inputs()
    model = _model(mass)
    state = project_to_shell(x * mask.unsqueeze(-1), mass)
    output, diagnostics = model.step_hyperbolic(
        y_t=state, jet_conditions=cond, mask=mask,
        t_start=torch.tensor(0.0), t_end=torch.tensor(0.01),
        hyperbolic_model="mass_shell", regulator_mass=mass,
        ref_vectors=refs, return_diagnostics=True,
    )
    assert output.shape == state.shape
    assert diagnostics["substeps"] == 1
