"""Mass-shell inference/sampling path (experiment plan Phase 4).

Exercises the integrator that generate_samples() drives for a mass_shell run — project the
prior onto H_m, then repeatedly call model.step_hyperbolic(hyperbolic_model='mass_shell') —
without needing jetnet. Guards the properties the sampler must maintain over a full
integration: stay on the shell (with re-projection preventing drift), stay finite, respect the
mask (padding parked at the apex), and work with references + CFG on.
"""
import pytest
import torch

from util.mass_shell import project_to_shell
from util.minkowski_utils import normsq4
from tests.lorentz_test_utils import build_model, sample_inputs

# Well-conditioned shell (mass ~ momentum scale) so an *untrained* random model is stable over
# a full integration — these tests target the sampler's *logic* (on-shell, masking,
# determinism, refs/CFG). The stiff near-light-like regime (small m) is a trained-model /
# clamp concern, covered separately by test_exp_map_clamps_stiff_regime in test_mass_shell.py.
M = 1.0


def _integrate(model, y0, cond, mask, steps, ref_vectors=None, use_cfg=False):
    """Mirror the generate_samples mass-shell loop over `steps` uniform time steps."""
    times = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    y = y0
    on_shell_err = []
    for i in range(steps):
        y = model.step_hyperbolic(
            y_t=y, jet_conditions=cond, mask=mask,
            t_start=times[i], t_end=times[i + 1],
            hyperbolic_model="mass_shell", regulator_mass=M,
            use_cfg=use_cfg, guidance_weight=2.0, ref_vectors=ref_vectors,
        )
        assert torch.isfinite(y).all()
        real = mask > 0
        err = (normsq4(y)[real] - M * M).abs().max().item()
        on_shell_err.append(err)
    return y, max(on_shell_err)


def test_full_integration_stays_on_shell():
    """Over a full 64-step integration, real particles stay on H_m to tight tolerance
    (re-projection keeps numerical drift from accumulating)."""
    model = build_model(seed=0)
    x, _, cond, mask, _ = sample_inputs(seed=3)
    y0 = project_to_shell(x * mask.unsqueeze(-1), M)
    y, max_err = _integrate(model, y0, cond, mask, steps=64)
    assert max_err < 1e-6, f"drifted off shell: max |<y,y> - m^2| = {max_err:.2e}"


def test_padding_parks_at_apex():
    """Masked (padding) rows sit at the apex (m, 0, 0, 0): zero spatial momentum, energy m."""
    model = build_model(seed=0)
    x, _, cond, mask, _ = sample_inputs(seed=3)
    y0 = project_to_shell(x * mask.unsqueeze(-1), M)
    y, _ = _integrate(model, y0, cond, mask, steps=8)
    pad = mask == 0
    if pad.any():
        assert torch.allclose(y[..., 1:4][pad], torch.zeros_like(y[..., 1:4][pad]), atol=1e-8)
        assert torch.allclose(y[..., 0][pad], torch.full_like(y[..., 0][pad], M), atol=1e-6)


def test_masking_real_output_independent_of_padding():
    """Real-particle trajectories are unaffected by how many padding rows trail them."""
    model = build_model(seed=0)
    x, _, cond, mask, _ = sample_inputs(batch=1, n_real=5, max_particles=8, seed=4)
    n_real = int(mask[0].sum().item())

    y_full = project_to_shell(x * mask.unsqueeze(-1), M)
    y_full, _ = _integrate(model, y_full, cond, mask, steps=16)

    xt, maskt = x[:, :n_real], mask[:, :n_real]
    y_trim = project_to_shell(xt * maskt.unsqueeze(-1), M)
    y_trim, _ = _integrate(model, y_trim, cond.clone(), maskt, steps=16)

    assert torch.allclose(y_full[:, :n_real], y_trim, atol=1e-8)


def test_references_and_cfg_paths_stay_on_shell():
    """With references + classifier-free guidance on, the sampler still stays on the shell."""
    model = build_model(use_reference_vectors=True, use_node_scalars=True, seed=0)
    x, _, cond, mask, refs = sample_inputs(seed=5)
    y0 = project_to_shell(x * mask.unsqueeze(-1), M)
    y, max_err = _integrate(model, y0, cond, mask, steps=32, ref_vectors=refs, use_cfg=True)
    assert max_err < 1e-6
    assert torch.isfinite(y).all()


def test_step_is_deterministic():
    model = build_model(seed=0)
    x, _, cond, mask, _ = sample_inputs(seed=3)
    y0 = project_to_shell(x * mask.unsqueeze(-1), M)
    t0 = torch.tensor(0.0, dtype=torch.float64)
    t1 = torch.tensor(0.1, dtype=torch.float64)
    a = model.step_hyperbolic(y0, cond, mask, t0, t1, hyperbolic_model="mass_shell", regulator_mass=M)
    b = model.step_hyperbolic(y0, cond, mask, t0, t1, hyperbolic_model="mass_shell", regulator_mass=M)
    assert torch.equal(a, b)


def test_adaptive_step_matches_euler_when_no_subdivision_needed():
    model = build_model(seed=0)
    x, _, cond, mask, _ = sample_inputs(seed=3)
    y0 = project_to_shell(x * mask.unsqueeze(-1), M)
    t0 = torch.tensor(0.0, dtype=torch.float64)
    t1 = torch.tensor(0.01, dtype=torch.float64)
    ordinary = model.step_hyperbolic(
        y0, cond, mask, t0, t1, hyperbolic_model="mass_shell", regulator_mass=M)
    adaptive = model.step_hyperbolic(
        y0, cond, mask, t0, t1, hyperbolic_model="mass_shell", regulator_mass=M,
        max_step_rapidity=1e6, max_substeps=4)
    assert torch.equal(ordinary, adaptive)


def test_adaptive_step_subdivides_large_field_and_stays_finite(monkeypatch):
    model = build_model(seed=0)
    x, _, cond, mask, _ = sample_inputs(seed=3)
    y0 = project_to_shell(x * mask.unsqueeze(-1), M)

    original = model._mass_shell_velocity
    calls = 0

    def amplified(*args, **kwargs):
        nonlocal calls
        calls += 1
        return 2000.0 * original(*args, **kwargs)

    monkeypatch.setattr(model, "_mass_shell_velocity", amplified)
    out = model.step_hyperbolic(
        y0, cond, mask, torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.01, dtype=torch.float64), hyperbolic_model="mass_shell",
        regulator_mass=M, max_step_rapidity=0.05, max_substeps=64)
    assert calls > 1
    assert torch.isfinite(out).all()
    real = mask > 0
    assert (normsq4(out)[real] - M * M).abs().max() < 1e-6


def test_adaptive_step_fails_instead_of_publishing_unbounded_substeps(monkeypatch):
    model = build_model(seed=0)
    x, _, cond, mask, _ = sample_inputs(seed=3)
    y0 = project_to_shell(x * mask.unsqueeze(-1), M)
    original = model._mass_shell_velocity
    monkeypatch.setattr(model, "_mass_shell_velocity",
                        lambda *args, **kwargs: 100.0 * original(*args, **kwargs))
    with pytest.raises(FloatingPointError, match="exceeded 1 substeps"):
        model.step_hyperbolic(
            y0, cond, mask, torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0.1, dtype=torch.float64), hyperbolic_model="mass_shell",
            regulator_mass=M, max_step_rapidity=0.01, max_substeps=1)
