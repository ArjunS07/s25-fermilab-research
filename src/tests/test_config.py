import tempfile
import os

import pytest
import yaml
from pydantic import ValidationError

from config import (
    TrainRunConfig,
    InferRunConfig,
    CacheRunConfig,
    apply_dotlist_overrides,
    build_config,
    load_yaml_config,
    parse_config_cli,
    train_config_to_namespace,
    infer_config_to_namespace,
    cache_config_to_namespace,
    run_config_dict,
)


# ── Default parity with legacy argparse defaults ────────────────────────────

def test_train_defaults_match_legacy_argparse():
    cfg = TrainRunConfig()
    assert cfg.paths.output_path == "/mnt/data/output"
    assert cfg.data.jet_types == ["g", "q", "t"]
    assert cfg.data.num_particles == 150
    assert cfg.model.n_hidden == 128
    assert cfg.model.n_layers == 3
    assert cfg.model.use_residual is False
    assert cfg.training.n_train_samples == 1_000_000
    assert cfg.training.batch_size == 16
    assert cfg.training.target_batch_size == 256
    assert cfg.training.cfg_null_dropout_rate == 0.2
    assert cfg.training.num_epochs == 100
    assert cfg.training.epoch_frac == 1.0
    assert cfg.training.max_optimizer_steps is None
    assert cfg.training.stability_probe_steps == []
    assert cfg.training.stability_probe_save_checkpoints is False
    assert cfg.training.qualification_min_loss_improvement is None
    assert cfg.training.sigma_min == 1e-4
    assert cfg.training.train_space == "cartesian"
    assert cfg.training.time_sampling == "power_law"
    assert cfg.inference.n_samples == 50_000
    assert cfg.inference.n_viz_samples == 1000
    assert cfg.inference.integration_steps == 16
    assert cfg.inference.integration_end_time == 0.99999
    assert cfg.training.use_cosine_lr is True
    assert cfg.training.lr_t0 == 0
    assert cfg.training.lr_warmup_epochs == 10
    assert cfg.model.use_hyperbolic is False
    assert cfg.training.use_curriculum is True
    assert cfg.training.use_time_sampling is True
    assert cfg.model.use_reference_vectors is False
    assert cfg.model.use_node_scalars is False
    assert cfg.training.prior_dist == "isotropic_com"
    assert cfg.training.use_icp is False
    assert cfg.training.eta_min_factor == 0.3
    assert cfg.training.use_ema is False
    assert cfg.training.ema_decay == 0.999
    assert cfg.model.use_adaln is False
    assert cfg.training.curriculum_alpha_start == 2.0
    assert cfg.training.n_curriculum_buckets == 10
    assert cfg.paths.icp_cache_path is None
    assert cfg.paths.cache_dir == "/mnt/data/caches"
    assert cfg.paths.resume_weights is None
    assert cfg.training.distributed is False
    assert cfg.training.lr == 6e-4
    assert cfg.training.weight_decay == 1e-6


def test_infer_defaults_match_legacy_argparse():
    cfg = InferRunConfig()
    assert cfg.paths.out_dir is None
    assert cfg.model.n_hidden == 128
    assert cfg.model.n_layers == 3
    assert cfg.model.use_residual is False
    assert cfg.data.num_particles == 30
    assert cfg.data.jet_types == ["g", "q", "t"]
    assert cfg.inference.n_samples == 50_000
    assert cfg.inference.n_viz_samples == 1000
    assert cfg.inference.integration_steps == 16
    assert cfg.inference.integration_end_time == 0.99999
    assert cfg.inference.batch_size == 256
    assert cfg.inference.cfg_guidance_weight == 2.0
    assert cfg.inference.use_cfg is False
    assert cfg.inference.seed == 42
    assert cfg.model.use_hyperbolic is False
    assert cfg.inference.sampler == "euler"
    assert cfg.inference.mass_shell_max_step_rapidity is None
    assert cfg.inference.mass_shell_max_substeps == 64
    assert cfg.inference.warn_invalid_fraction is None
    assert cfg.inference.max_invalid_fraction is None
    assert cfg.model.use_reference_vectors is False
    assert cfg.model.use_node_scalars is False
    assert cfg.model.use_adaln is False
    assert cfg.inference.vf_mode == "none"
    assert cfg.inference.skip_samples is False
    assert cfg.inference.skip_metrics is False
    assert cfg.inference.stability_probe_samples == 64
    assert cfg.inference.stability_probe_integration_steps == 8
    assert cfg.model.velocity_readout_init == "small_normal"


def test_cache_defaults_match_legacy_argparse():
    cfg = CacheRunConfig()
    assert cfg.paths.output_path == "/mnt/data/output"
    assert cfg.data.jet_types == ["g", "q", "t"]
    assert cfg.data.num_particles == 150
    assert cfg.paths.cache_dir == "/mnt/data/caches"
    assert cfg.cache.n_samples is None
    assert cfg.cache.n_workers == max(1, (os.cpu_count() or 2) // 2)
    assert cfg.cache.icp_max_iter == 1000
    assert cfg.cache.skip_if_exists is True


# ── Literal validation ───────────────────────────────────────────────────────

def test_invalid_train_space_choice_rejected():
    with pytest.raises(ValidationError):
        TrainRunConfig(training={"train_space": "spherical"})


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


def test_tangent_attention_requires_typed_mass_shell_contract():
    with pytest.raises(ValidationError):
        TrainRunConfig(model={"backbone": "tangent_attention"})
    cfg = TrainRunConfig(model={
        "backbone": "tangent_attention", "use_reference_vectors": True,
        "include_mass_condition": True, "use_hyperbolic": True,
        "hyperbolic_model": "mass_shell", "n_hidden": 128,
        "num_attention_heads": 4,
    })
    assert cfg.model.backbone == "tangent_attention"


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
        base, ["model.use_residual=true", "data.jet_types=[g,q,t]"]
    )
    assert result["model"]["use_residual"] is True
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


# ── Namespace bridges ────────────────────────────────────────────────────────

def test_train_config_to_namespace_attribute_names():
    cfg = TrainRunConfig()
    ns = train_config_to_namespace(cfg)
    assert ns.num_particles == 150
    assert ns.n_hidden == 128
    assert ns.batch_size == 16
    assert ns.use_cosine_lr is True
    assert ns.prior_dist == "isotropic_com"
    assert ns.distributed is False
    assert ns.lr == 6e-4


def test_infer_config_to_namespace_attribute_names():
    cfg = InferRunConfig()
    ns = infer_config_to_namespace(cfg)
    assert ns.num_particles == 30
    assert ns.batch_size == 256
    assert ns.sampler == "euler"
    assert ns.vf_mode == "none"


def test_cache_config_to_namespace_attribute_names():
    cfg = CacheRunConfig()
    ns = cache_config_to_namespace(cfg)
    assert ns.num_particles == 150
    assert ns.n_workers == max(1, (os.cpu_count() or 2) // 2)
    assert ns.skip_if_exists is True


# ── Checkpoint config dict (backward-compat format) ─────────────────────────

def test_run_config_dict_matches_legacy_keys():
    cfg = TrainRunConfig(data={"num_particles": 30, "jet_types": ["g"]})
    d = run_config_dict(cfg, final_scale=1.2345)
    assert set(d.keys()) == {
        "num_particles", "n_layers", "n_hidden", "use_residual", "include_pt",
        "use_reference_vectors", "use_node_scalars", "node_scalar_seed",
        "use_adaln", "use_attention", "jet_types", "final_scale",
        "backbone", "include_mass_condition", "num_attention_heads",
        "vector_channels", "regulator_mass", "velocity_readout_init",
    }
    assert d["num_particles"] == 30
    assert d["jet_types"] == ["g"]
    assert d["include_pt"] is True
    assert d["final_scale"] == 1.2345
