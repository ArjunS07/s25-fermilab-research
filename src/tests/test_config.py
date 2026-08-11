import tempfile
import os

import pytest
import yaml
from pydantic import ValidationError

from config import (
    TrainRunConfig,
    InferRunConfig,
    apply_dotlist_overrides,
    build_config,
    load_yaml_config,
    parse_config_cli,
)


# ── Defaults ────────────────────────────────────────────────────────────────

def test_train_defaults():
    cfg = TrainRunConfig()
    assert cfg.paths.output_path == "/mnt/data/output"
    assert cfg.data.jet_types == ["g", "q", "t"]
    assert cfg.data.num_particles == 150
    assert cfg.model.n_hidden == 128
    assert cfg.model.n_layers == 3
    assert cfg.model.architecture == "mass_shell_gnn"
    assert cfg.model.flow_geometry == "mass_shell"
    assert cfg.model.reference_mode == "plain_readout"
    assert cfg.model.scalar_init_mode == "normsq"
    assert cfg.model.particle_readout_mode == "ambient"
    assert cfg.model.geometry_mode == "evolving_auxiliary"
    assert cfg.model.field_degree_normalization == "none"
    assert cfg.model.use_reference_vectors is True
    assert cfg.model.include_mass_condition is True
    assert cfg.training.n_train_samples == 1_000_000
    assert cfg.training.batch_size == 16
    assert cfg.training.target_batch_size == 256
    assert cfg.training.cfg_null_dropout_rate == 0.2
    assert cfg.training.num_epochs == 100
    assert cfg.training.max_optimizer_steps is None
    assert cfg.training.stability_probe_steps == []
    assert cfg.training.time_sampling == "power_law"
    assert cfg.inference.n_samples == 50_000
    assert cfg.inference.integration_steps == 16
    assert cfg.training.use_cosine_lr is True
    assert cfg.training.lr_warmup_epochs == 10
    assert cfg.training.use_curriculum is True
    assert cfg.training.use_time_sampling is True
    assert cfg.training.prior_dist == "isotropic_com"
    assert cfg.training.coupling == "online_geodesic_icp"
    assert cfg.training.use_ema is False
    assert cfg.training.lr == 6e-4


def test_infer_defaults():
    cfg = InferRunConfig()
    assert cfg.paths.out_dir is None
    assert cfg.model.n_hidden == 128
    assert cfg.model.n_layers == 3
    assert cfg.data.num_particles == 30
    assert cfg.data.jet_types == ["g", "q", "t"]
    assert cfg.inference.integration_steps == 16
    assert cfg.inference.batch_size == 256
    assert cfg.inference.use_cfg is False
    assert cfg.inference.sampler == "euler"
    assert cfg.model.use_reference_vectors is True
    assert cfg.model.include_mass_condition is True
    assert cfg.inference.vf_mode == "none"
    assert cfg.inference.use_ema_weights is False


# ── Literal / contract validation ───────────────────────────────────────────

def test_invalid_prior_dist_choice_rejected():
    with pytest.raises(ValidationError):
        TrainRunConfig(training={"prior_dist": "nonexistent"})


def test_invalid_sampler_choice_rejected():
    with pytest.raises(ValidationError):
        InferRunConfig(inference={"sampler": "rk4"})


@pytest.mark.parametrize("field,value", [
    ("mass_shell_max_step_rapidity", 0),
    ("mass_shell_max_step_rapidity", -0.5),
    ("mass_shell_max_substeps", 0),
    ("integration_end_time", 0),
    ("integration_end_time", 1.1),
])
def test_invalid_adaptive_mass_shell_limits_rejected(field, value):
    with pytest.raises(ValidationError):
        InferRunConfig(inference={field: value})


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        TrainRunConfig(training={"typo_field": 1})


def test_removed_legacy_field_rejected():
    """Legacy/variant model fields no longer exist and must hard-fail (extra=forbid)."""
    for dead in ("backbone", "geometric_state", "use_global_pooling", "use_hyperbolic",
                 "use_attention", "vector_channels"):
        with pytest.raises(ValidationError):
            TrainRunConfig(model={dead: 0})


@pytest.mark.parametrize("field", ["use_reference_vectors", "include_mass_condition"])
def test_mass_shell_contract_flags_must_be_true(field):
    with pytest.raises(ValidationError):
        TrainRunConfig(model={field: False})


@pytest.mark.parametrize("geometry", ["euclidean", "mass_shell"])
@pytest.mark.parametrize("reference_mode,use_references", [
    ("none", False), ("plain_readout", True),
])
def test_lorentznet_geometry_reference_matrix_validates(geometry, reference_mode, use_references):
    cfg = TrainRunConfig(model={
        "architecture": "lorentznet",
        "flow_geometry": geometry,
        "reference_mode": reference_mode,
        "use_reference_vectors": use_references,
    })
    assert cfg.model.flow_geometry == geometry
    assert cfg.model.reference_mode == reference_mode


def test_lorentznet_reference_flag_must_match_mode():
    with pytest.raises(ValidationError):
        TrainRunConfig(model={
            "architecture": "lorentznet", "reference_mode": "none",
            "use_reference_vectors": True,
        })
    with pytest.raises(ValidationError):
        TrainRunConfig(model={
            "architecture": "lorentznet", "reference_mode": "plain_readout",
            "use_reference_vectors": False,
        })


def test_shell_geometric_lorentznet_modes_require_mass_shell_geometry():
    common = {
        "architecture": "lorentznet",
        "flow_geometry": "mass_shell",
        "reference_mode": "normalized_tangent_readout",
        "particle_readout_mode": "normalized_logmap",
        "geometry_mode": "fixed_physical_geodesic",
        "field_degree_normalization": "sqrt",
        "use_reference_vectors": True,
    }
    cfg = TrainRunConfig(model=common)
    assert cfg.model.geometry_mode == "fixed_physical_geodesic"
    with pytest.raises(ValidationError):
        TrainRunConfig(model={**common, "flow_geometry": "euclidean"})
    with pytest.raises(ValidationError):
        TrainRunConfig(model={
            "architecture": "lorentznet",
            "reference_mode": "plain_readout",
            "particle_readout_mode": "ambient",
            "field_degree_normalization": "sqrt",
            "use_reference_vectors": True,
        })


def test_reference_contraction_scalar_init_requires_references_without_readout():
    cfg = TrainRunConfig(model={
        "architecture": "lorentznet",
        "flow_geometry": "mass_shell",
        "reference_mode": "none",
        "scalar_init_mode": "reference_contractions",
        "use_reference_vectors": True,
    })
    assert cfg.model.reference_mode == "none"
    assert cfg.model.scalar_init_mode == "reference_contractions"
    assert cfg.model.use_reference_vectors is True
    with pytest.raises(ValidationError):
        TrainRunConfig(model={
            "architecture": "lorentznet",
            "reference_mode": "none",
            "scalar_init_mode": "reference_contractions",
            "use_reference_vectors": False,
        })


def test_reference_contraction_scalar_init_rejected_for_mass_shell_gnn():
    with pytest.raises(ValidationError):
        TrainRunConfig(model={"scalar_init_mode": "reference_contractions"})


@pytest.mark.parametrize("filename,reference_mode", [
    ("g30-lorentznet-e-rfm-refscalar-icp-none.yaml", "none"),
    ("g30-lorentznet-f-rfm-refscalar-icp-refs.yaml", "plain_readout"),
])
def test_reference_scalar_icp_experiment_configs_validate(filename, reference_mode):
    path = os.path.join(os.path.dirname(__file__), "..", "configs", filename)
    cfg = build_config(TrainRunConfig, path)
    assert cfg.model.flow_geometry == "mass_shell"
    assert cfg.model.scalar_init_mode == "reference_contractions"
    assert cfg.model.reference_mode == reference_mode
    assert cfg.model.use_reference_vectors is True
    assert cfg.training.coupling == "online_geodesic_icp"
    assert cfg.training.prior_dist == "axis_aligned_equal"
    assert cfg.inference.prior_dist == "axis_aligned_equal"


@pytest.mark.parametrize(
    "filename,reference_mode,geometry_mode",
    [
        ("g30-lorentznet-g-rfm-logmap-lognormal.yaml", "plain_readout", "evolving_auxiliary"),
        ("g30-lorentznet-h-rfm-logmap-tangentrefs.yaml", "normalized_tangent_readout", "evolving_auxiliary"),
        ("g30-lorentznet-i-rfm-logmap-fixedgeom.yaml", "plain_readout", "fixed_physical_geodesic"),
        ("g30-lorentznet-j-rfm-logmap-fixedgeom-tangentrefs.yaml", "normalized_tangent_readout", "fixed_physical_geodesic"),
    ],
)
def test_logmap_factorial_configs_validate(filename, reference_mode, geometry_mode):
    path = os.path.join(os.path.dirname(__file__), "..", "configs", filename)
    cfg = build_config(TrainRunConfig, path)
    assert cfg.model.flow_geometry == "mass_shell"
    assert cfg.model.scalar_init_mode == "reference_contractions"
    assert cfg.model.reference_mode == reference_mode
    assert cfg.model.particle_readout_mode == "normalized_logmap"
    assert cfg.model.geometry_mode == geometry_mode
    assert cfg.model.field_degree_normalization == "sqrt"
    assert cfg.training.prior_dist == "axis_aligned_lognormal"
    assert cfg.inference.prior_dist == "axis_aligned_lognormal"
    assert cfg.training.coupling == "online_geodesic_icp"


@pytest.mark.parametrize("field", ["n_hidden", "n_layers"])
def test_model_dimensions_must_be_positive(field):
    with pytest.raises(ValidationError):
        TrainRunConfig(model={field: 0})


def test_lr_cadence_and_rng_seeds_validate():
    cfg = TrainRunConfig(training={
        "lr_step_unit": "epoch", "model_seed": 1, "data_order_seed": 2,
        "time_seed": 3, "dropout_seed": 4, "prior_seed": 5,
    })
    assert cfg.training.lr_step_unit == "epoch"
    assert (cfg.training.model_seed, cfg.training.data_order_seed, cfg.training.time_seed,
            cfg.training.dropout_seed, cfg.training.prior_seed) == (1, 2, 3, 4, 5)


def test_optimizer_warmup_steps_are_typed_and_validated():
    cfg = TrainRunConfig(training={
        "lr_step_unit": "optimizer_step", "lr_warmup_steps": 200,
        "max_optimizer_steps": 500,
    })
    assert cfg.training.lr_warmup_steps == 200
    with pytest.raises(ValidationError):
        TrainRunConfig(training={"lr_step_unit": "epoch", "lr_warmup_steps": 200})
    with pytest.raises(ValidationError):
        TrainRunConfig(training={
            "lr_step_unit": "optimizer_step", "lr_warmup_steps": 500,
            "max_optimizer_steps": 500,
        })


def test_qualification_steps_are_sorted_bounded_and_nonnegative():
    cfg = TrainRunConfig(training={
        "max_optimizer_steps": 2000,
        "stability_probe_steps": [0, 100, 500, 2000],
    })
    assert cfg.training.stability_probe_steps[-1] == 2000
    for invalid in ([100, 0], [0, 0], [-1], [2001]):
        with pytest.raises(ValidationError):
            TrainRunConfig(training={
                "max_optimizer_steps": 2000,
                "stability_probe_steps": invalid,
            })


def test_target_batch_size_must_be_exact_accumulation_multiple():
    cfg = TrainRunConfig(training={"batch_size": 50, "target_batch_size": 250})
    assert cfg.training.target_batch_size // cfg.training.batch_size == 5
    for target in (49, 251):
        with pytest.raises(ValidationError):
            TrainRunConfig(training={"batch_size": 50, "target_batch_size": target})


def test_invalid_fraction_warning_must_not_exceed_hard_gate():
    cfg = TrainRunConfig(inference={
        "warn_invalid_fraction": 0.0001,
        "max_invalid_fraction": 0.001,
    })
    assert cfg.inference.warn_invalid_fraction == 0.0001
    with pytest.raises(ValidationError):
        TrainRunConfig(inference={
            "warn_invalid_fraction": 0.01,
            "max_invalid_fraction": 0.001,
        })


# ── Dotlist overrides ────────────────────────────────────────────────────────

def test_apply_dotlist_overrides_nested_scalar():
    base = {"training": {"num_epochs": 100}}
    result = apply_dotlist_overrides(base, ["training.num_epochs=300"])
    assert result["training"]["num_epochs"] == 300
    assert base["training"]["num_epochs"] == 100  # base untouched


def test_apply_dotlist_overrides_bool_and_list():
    base = {}
    result = apply_dotlist_overrides(
        base, ["training.use_ema=true", "data.jet_types=[g,q,t]"]
    )
    assert result["training"]["use_ema"] is True
    assert result["data"]["jet_types"] == ["g", "q", "t"]


def test_apply_dotlist_overrides_creates_missing_path():
    result = apply_dotlist_overrides({}, ["training.batch_size=32"])
    assert result == {"training": {"batch_size": 32}}


def test_apply_dotlist_overrides_invalid_format_raises():
    with pytest.raises(ValueError):
        apply_dotlist_overrides({}, ["no_equals_sign"])


# ── YAML round-trip ──────────────────────────────────────────────────────────

def test_yaml_round_trip():
    cfg = TrainRunConfig(training={"num_epochs": 42, "use_curriculum": False})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg.model_dump(), f)
        loaded_dict = load_yaml_config(path)
        cfg2 = TrainRunConfig.model_validate(loaded_dict)
    assert cfg2.training.num_epochs == 42
    assert cfg2.training.use_curriculum is False
    assert cfg2 == cfg


def test_build_config_from_yaml_with_overrides():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run.yaml")
        with open(path, "w") as f:
            yaml.dump({"training": {"num_epochs": 50}}, f)
        cfg = build_config(TrainRunConfig, path, ["training.num_epochs=99"])
    assert cfg.training.num_epochs == 99


def test_build_config_no_path_no_overrides_is_defaults():
    cfg = build_config(TrainRunConfig, None, [])
    assert cfg == TrainRunConfig()


# ── CLI parsing ──────────────────────────────────────────────────────────────

def test_parse_config_cli_basic():
    config_path, overrides = parse_config_cli(
        ["--config", "configs/foo.yaml", "--set", "a.b=1", "--set", "c.d=2"]
    )
    assert config_path == "configs/foo.yaml"
    assert overrides == ["a.b=1", "c.d=2"]


def test_parse_config_cli_no_args():
    config_path, overrides = parse_config_cli([])
    assert config_path is None
    assert overrides == []
