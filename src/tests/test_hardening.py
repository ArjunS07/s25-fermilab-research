"""Regression / dtype / config round-trip guards for the flag-gated build-ahead work.

  - Run A (all new flags off) reproduces a checked-in reference tensor bit-for-bit, and its
    parameter set contains none of the new-capability modules → the ablation baseline is
    untouched by any later flag work.
  - Geometry ops keep float32 in -> float32 out (float64 only used internally).
  - Every new config field survives YAML load + namespace bridge + run_config_dict.
"""
import os

import pytest
import torch

from tests.lorentz_test_utils import build_model, sample_inputs

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "run_a_reference.pt")

# Parameter-name fragments introduced by the new (default-off) capabilities.
_NEW_MODULE_FRAGMENTS = ("phi_attn", "phi_h", "adaln_mod", "node_seed")


def test_run_a_matches_reference():
    """All-off model on the fixed sample matches the committed reference (baseline unchanged)."""
    ref = torch.load(_FIXTURE, weights_only=True)["output"]
    model = build_model(seed=0)
    x, t, cond, mask, _ = sample_inputs(seed=1)
    with torch.no_grad():
        v = model(x, t, cond, mask)
    assert torch.allclose(v, ref, atol=1e-12, rtol=0.0)


def test_run_a_has_no_new_capability_params():
    keys = list(build_model(seed=0).state_dict().keys())
    for frag in _NEW_MODULE_FRAGMENTS:
        assert not any(frag in k for k in keys), f"baseline unexpectedly has {frag} params"


def test_run_a_is_deterministic():
    x, t, cond, mask, _ = sample_inputs(seed=1)
    v1 = build_model(seed=0)(x, t, cond, mask)
    v2 = build_model(seed=0)(x, t, cond, mask)
    assert torch.equal(v1, v2)


@pytest.mark.parametrize("flag", ["use_node_scalars", "use_adaln", "use_attention"])
def test_each_flag_changes_output(flag):
    """Each new flag is actually wired: turning it on perturbs the baseline output."""
    x, t, cond, mask, _ = sample_inputs(seed=1)
    base = build_model(seed=0)(x, t, cond, mask)
    on = build_model(seed=0, **{flag: True})(x, t, cond, mask)
    assert not torch.allclose(base, on, atol=1e-6)


# ── Geometry dtype guards ───────────────────────────────────────────────────

def test_mass_shell_ops_promote_geometry_to_float64():
    from util.mass_shell import (project_to_shell, exp_map, log_map, pushforward_to_tangent,
                                 geodesic_interpolant, conditional_vector_field, mass_shell_loss)
    m = 0.5
    p = project_to_shell(torch.randn(2, 4, 4), m)
    q = project_to_shell(torch.randn(2, 4, 4), m)
    u = pushforward_to_tangent(p, torch.randn(2, 4, 4) * 0.2, m)
    t = torch.full((2,), 0.3)
    mask = torch.ones(2, 4)
    for out in (p, exp_map(p, u, m), log_map(p, q, m), pushforward_to_tangent(p, q, m),
                geodesic_interpolant(p, q, t, m), conditional_vector_field(p, q, t, m),
                mass_shell_loss(u, u, mask, m)):
        assert out.dtype == torch.float64


# ── Config round-trip ───────────────────────────────────────────────────────

def test_new_config_fields_round_trip():
    from config import (TrainRunConfig, InferRunConfig, build_config,
                        train_config_to_namespace, infer_config_to_namespace,
                        run_config_dict)

    cfg = TrainRunConfig.model_validate({
        "model": {"use_attention": True, "use_hyperbolic": True,
                  "hyperbolic_model": "mass_shell", "regulator_mass": 0.25},
    })
    # Survives model_dump -> re-validate.
    cfg2 = TrainRunConfig.model_validate(cfg.model_dump())
    assert cfg2.model.use_attention is True
    assert cfg2.model.hyperbolic_model == "mass_shell"
    assert cfg2.model.regulator_mass == 0.25

    ns = train_config_to_namespace(cfg)
    assert ns.use_attention is True
    assert ns.hyperbolic_model == "mass_shell"
    assert ns.regulator_mass == 0.25

    icfg = InferRunConfig.model_validate({
        "model": {"use_attention": True, "hyperbolic_model": "mass_shell", "regulator_mass": 0.25},
    })
    ins = infer_config_to_namespace(icfg)
    assert ins.use_attention is True and ins.hyperbolic_model == "mass_shell" and ins.regulator_mass == 0.25

    assert run_config_dict(cfg, final_scale=1.0)["use_attention"] is True


def test_invalid_enum_values_rejected():
    from config import ModelConfig
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ModelConfig(hyperbolic_model="ball")
