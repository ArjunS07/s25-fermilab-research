"""
config.py — Typed run configuration for train.py / infer.py.

Configs are pydantic models loaded from YAML (`--config path/to/run.yaml`) with
optional dotlist overrides (`--set training.num_epochs=300`). Every field default
below matches the current argparse default for the corresponding flag, so an
empty/omitted YAML reproduces today's behavior exactly.
"""

import argparse
import copy
import os
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jet_types: list[str] = Field(default_factory=lambda: ["g", "q", "t"])
    num_particles: int = 150


class InferDataConfig(DataConfig):
    """infer.py's standalone CLI historically defaulted num_particles to 30."""
    num_particles: int = 30


class ModelConfig(BaseModel):
    """Reference type-A architecture or the matched LorentzNet FM ablation."""
    model_config = ConfigDict(extra="forbid")

    n_hidden: int = Field(default=128, ge=1)
    n_layers: int = Field(default=3, ge=1)
    architecture: Literal["mass_shell_gnn", "lorentznet"] = "mass_shell_gnn"
    flow_geometry: Literal["euclidean", "mass_shell"] = "mass_shell"
    reference_mode: Literal[
        "none", "plain_readout", "normalized_tangent_readout"
    ] = "plain_readout"
    scalar_init_mode: Literal["normsq", "reference_contractions"] = "normsq"
    particle_readout_mode: Literal["ambient", "normalized_logmap"] = "ambient"
    geometry_mode: Literal["evolving_auxiliary", "fixed_physical_geodesic"] = (
        "evolving_auxiliary"
    )
    field_degree_normalization: Literal["none", "sqrt"] = "none"
    inject_condition_time_each_block: bool = False
    # Shell mass in normalised units (momenta are O(1) after final_scale). The mass-shell
    # regulator: smaller = more massless/relativistic but numerically stiffer. Runs use 0.1.
    regulator_mass: float = 0.5
    # Reference vectors are a structural requirement of the mass-shell GNN.
    use_reference_vectors: bool = True

    @model_validator(mode="after")
    def _require_reference_contract(self):
        if self.architecture == "mass_shell_gnn":
            if self.flow_geometry != "mass_shell":
                raise ValueError("mass_shell_gnn only supports flow_geometry='mass_shell'")
            if not self.use_reference_vectors:
                raise ValueError(
                    "model.use_reference_vectors must be true (mass-shell GNN contract)"
                )
            if self.scalar_init_mode != "normsq":
                raise ValueError(
                    "model.scalar_init_mode is only configurable for LorentzNet"
                )
            if self.particle_readout_mode != "ambient":
                raise ValueError(
                    "model.particle_readout_mode is only configurable for LorentzNet"
                )
            if self.geometry_mode != "evolving_auxiliary":
                raise ValueError("model.geometry_mode is only configurable for LorentzNet")
            if self.field_degree_normalization != "none":
                raise ValueError(
                    "model.field_degree_normalization is only configurable for LorentzNet"
                )
            if self.inject_condition_time_each_block:
                raise ValueError(
                    "model.inject_condition_time_each_block is only configurable for LorentzNet"
                )
        else:
            expects_references = (
                self.reference_mode != "none"
                or self.scalar_init_mode == "reference_contractions"
            )
            if self.use_reference_vectors != expects_references:
                raise ValueError(
                    "LorentzNet use_reference_vectors must be true when either "
                    "plain_readout or reference_contractions needs references, and "
                    "false otherwise"
                )
            needs_shell_geometry = (
                self.reference_mode == "normalized_tangent_readout"
                or self.particle_readout_mode == "normalized_logmap"
                or self.geometry_mode == "fixed_physical_geodesic"
            )
            if needs_shell_geometry and self.flow_geometry != "mass_shell":
                raise ValueError(
                    "normalized tangent/log-map readouts and fixed physical geometry "
                    "require flow_geometry='mass_shell'"
                )
            if (self.field_degree_normalization == "sqrt"
                    and self.particle_readout_mode != "normalized_logmap"):
                raise ValueError(
                    "sqrt field-degree normalization requires normalized_logmap readout"
                )
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_train_samples: int = 1_000_000
    batch_size: int = 16
    target_batch_size: int = 256
    cfg_null_dropout_rate: float = 0.2
    num_epochs: int = 100
    epoch_frac: float = 1.0
    max_optimizer_steps: Optional[int] = Field(default=None, ge=1)
    stability_probe_steps: list[int] = Field(default_factory=list)
    stability_probe_save_checkpoints: bool = False
    qualification_min_loss_improvement: float | None = Field(default=None, ge=0, le=1)
    time_sampling: Literal["uniform", "power_law", "lognorm"] = "power_law"

    lr: float = 6e-4
    weight_decay: float = 1e-6
    use_cosine_lr: bool = True
    lr_warmup_steps: int = Field(default=0, ge=0)
    eta_min_factor: float = 0.3

    use_time_sampling: bool = True

    use_ema: bool = False
    ema_decay: float = 0.999

    prior_dist: Literal[
        "isotropic_com", "isotropic_lognorm", "jet_ref_frame", "axis_aligned",
        "axis_aligned_per_jet", "axis_aligned_equal", "axis_aligned_lognormal"
    ] = "isotropic_com"

    # Fresh-noise coupling applied online each step (no frozen cache → no path
    # memorization; see discussions/22). "online_geodesic_icp" recomputes the squared-
    # geodesic Hungarian assignment per batch on freshly drawn prior noise (minibatch
    # OT-CFM); "none" pairs fresh noise in identity order (for future ICP-vs-none ablation).
    coupling: Literal["online_geodesic_icp", "none"] = "online_geodesic_icp"

    distributed: bool = False
    model_seed: int = 42
    data_order_seed: int = 1042
    time_seed: int = 2042
    dropout_seed: int = 3042
    prior_seed: int = 4042

    @model_validator(mode="after")
    def validate_qualification_steps(self):
        if self.target_batch_size < self.batch_size:
            raise ValueError("training.target_batch_size must be at least training.batch_size")
        if self.target_batch_size % self.batch_size:
            raise ValueError(
                "training.target_batch_size must be divisible by training.batch_size; "
                "otherwise gradient accumulation silently uses a smaller effective batch"
            )
        if any(step < 0 for step in self.stability_probe_steps):
            raise ValueError("training.stability_probe_steps must be non-negative")
        if self.stability_probe_steps != sorted(set(self.stability_probe_steps)):
            raise ValueError("training.stability_probe_steps must be sorted and unique")
        if (self.max_optimizer_steps is not None and self.stability_probe_steps
                and self.stability_probe_steps[-1] > self.max_optimizer_steps):
            raise ValueError("stability probe step exceeds training.max_optimizer_steps")
        if (self.max_optimizer_steps is not None
                and self.lr_warmup_steps >= self.max_optimizer_steps):
            raise ValueError(
                "training.lr_warmup_steps must be smaller than max_optimizer_steps"
            )
        return self


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_samples: int = 50_000
    # When set, generate exactly this many samples for each configured jet type
    # rather than drawing types randomly.  This is the publication-evaluation
    # mode: a GQT run with value 50_000 produces exactly 150_000 samples.
    samples_per_jet_type: int | None = Field(default=None, ge=1)
    integration_steps: int = 16
    integration_end_time: float = Field(default=0.99999, gt=0, le=1)
    cfg_guidance_weight: float = 2.0
    use_cfg: bool = False
    use_ema_weights: bool = False
    seed: int = 42
    batch_size: int = 256
    # Scientific qualification threshold. Exceeding it marks qualification failed.
    max_invalid_fraction: float | None = Field(default=None, ge=0, le=1)
    # Dedicated smoke/qualification jobs may opt into a nonzero exit. Ordinary training and
    # diagnostic jobs retain artifacts/metrics and report qualification in summary.json.
    fail_on_qualification_error: bool = False
    # Softer reporting threshold. Exceeding it is recorded as a qualification warning
    # but does not suppress physics metrics.
    warn_invalid_fraction: float | None = Field(default=None, ge=0, le=1)
    stability_probe_samples: int = Field(default=64, ge=1)
    stability_probe_integration_steps: int = Field(default=8, ge=1)
    skip_samples: bool = False
    skip_metrics: bool = False
    stratify_metrics_by_class: bool = False
    prior_dist: Literal[
        "isotropic_com", "isotropic_lognorm", "jet_ref_frame", "axis_aligned",
        "axis_aligned_per_jet", "axis_aligned_equal", "axis_aligned_lognormal"
    ] = "isotropic_com"

    @model_validator(mode="after")
    def validate_invalid_fraction_thresholds(self):
        if (self.warn_invalid_fraction is not None
                and self.max_invalid_fraction is not None
                and self.warn_invalid_fraction > self.max_invalid_fraction):
            raise ValueError(
                "inference.warn_invalid_fraction must not exceed max_invalid_fraction"
            )
        return self


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str = "/mnt/data/output"
    resume_weights: Optional[str] = None
    checkpoint_path: Optional[str] = None
    out_dir: Optional[str] = None
    replay_bundle_path: Optional[str] = None
    replay_samples_path: Optional[str] = None
    replay_prior_samples_path: Optional[str] = None
    cache_dir: str = "/mnt/data/caches"


class TrainRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

class InferRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: InferDataConfig = Field(default_factory=InferDataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    paths: PathConfig = Field(default_factory=PathConfig)


# ── YAML loading + dotlist overrides ────────────────────────────────────────

def load_yaml_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def _coerce_override_value(raw: str) -> Any:
    """Parse a --set value as YAML so bool/int/float/list literals just work
    (e.g. `true`, `300`, `[g,q,t]`); falls back to the raw string otherwise."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_dotlist_overrides(base: dict, overrides: list[str]) -> dict:
    result = copy.deepcopy(base)
    for item in overrides:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid --set override (expected key=value): {item!r}")
        value = _coerce_override_value(raw_value)
        parts = key.strip().split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def build_config(config_cls, config_path: Optional[str], overrides: Optional[list[str]] = None):
    base = load_yaml_config(config_path)
    merged = apply_dotlist_overrides(base, overrides or [])
    return config_cls.model_validate(merged)


def parse_config_cli(argv: Optional[list[str]] = None) -> tuple[Optional[str], list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    ns, _ = parser.parse_known_args(argv)
    return ns.config, ns.overrides


# ── Generation controls (shared by train-time viz/eval and standalone infer) ──

def generation_controls_from_config(cfg, prior_dist):
    """Generation kwargs for generate_samples(), read directly from a run config.

    prior_dist differs by caller (training uses cfg.training.prior_dist; inference uses
    cfg.inference.prior_dist), so it is passed explicitly.
    """
    return {
        "use_cfg": cfg.inference.use_cfg,
        "cfg_guidance_weight": cfg.inference.cfg_guidance_weight,
        "regulator_mass": cfg.model.regulator_mass,
        "use_reference_vectors": cfg.model.use_reference_vectors,
        "use_hyperbolic": cfg.model.flow_geometry == "mass_shell",
        "integration_end_time": cfg.inference.integration_end_time,
        "prior_dist": prior_dist,
    }
