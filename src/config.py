"""
config.py — Typed run configuration for train.py / infer.py / cache_icp.py.

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
    stage1_model_version: Literal["legacy", "v2", "v3"] = "legacy"


class InferDataConfig(DataConfig):
    """infer.py's standalone CLI historically defaulted num_particles to 30."""
    num_particles: int = 30


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_hidden: int = 128
    n_layers: int = 3
    use_residual: bool = False
    use_reference_vectors: bool = False
    use_node_scalars: bool = False
    # Source for the initial per-node scalar h_i (only used when use_node_scalars=True).
    # "physics" — [ψ(m²), ψ(⟨x,e_t⟩)=E, ψ(⟨x,x_jet⟩)]; grid default (with cols 2–3 zeroed
    # when refs off). "zero" — h_i seeded as zeros (blank channel, purely learned via
    # message passing). "conditions" — jet_conditions (7-dim) broadcast per particle.
    node_scalar_seed: Literal["physics", "zero", "conditions"] = "physics"
    use_adaln: bool = False
    use_attention: bool = False
    use_hyperbolic: bool = False
    # When use_hyperbolic, which Riemannian geometry: "poincare" (radial-tanh ball, the
    # original) or "mass_shell" (hyperboloid at <p,p>=regulator_mass^2, Phase 4).
    hyperbolic_model: Literal["poincare", "mass_shell"] = "poincare"
    # Shell mass in normalised units (momenta are O(1) after final_scale). This is the primary
    # Phase-4 ablation knob; smaller = more massless/relativistic but numerically stiffer
    # (near-light-like, high curvature). 0.5 is a conditioned starting point.
    regulator_mass: float = 0.5
    # v2 keeps the ODE coordinates fixed inside the network and refines invariant scalar
    # plus tangent-vector channels with typed reference cross-attention.
    backbone: Literal["legacy", "tangent_attention"] = "legacy"
    include_mass_condition: bool = False
    num_attention_heads: int = Field(default=8, ge=1)
    vector_channels: int = Field(default=16, ge=1)
    velocity_readout_init: Literal["small_normal", "zero"] = "small_normal"

    @model_validator(mode="after")
    def validate_tangent_attention_contract(self):
        if self.backbone == "tangent_attention":
            if not self.use_reference_vectors:
                raise ValueError("tangent_attention requires model.use_reference_vectors=true")
            if not self.include_mass_condition:
                raise ValueError("tangent_attention requires model.include_mass_condition=true")
            if not self.use_hyperbolic or self.hyperbolic_model != "mass_shell":
                raise ValueError("tangent_attention currently requires mass-shell geometry")
            if self.n_hidden % self.num_attention_heads:
                raise ValueError("model.n_hidden must be divisible by model.num_attention_heads")
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
    sigma_min: float = 1e-4
    train_space: Literal["cartesian", "polar"] = "cartesian"
    time_sampling: Literal["uniform", "power_law", "lognorm"] = "power_law"

    lr: float = 6e-4
    weight_decay: float = 1e-6
    use_cosine_lr: bool = True
    lr_schedule: Literal["legacy_restarts", "monotonic_cosine"] = "legacy_restarts"
    lr_t0: int = 0
    lr_warmup_epochs: int = 10
    eta_min_factor: float = 0.3

    use_curriculum: bool = True
    use_time_sampling: bool = True
    curriculum_alpha_start: float = 2.0
    n_curriculum_buckets: int = 10

    use_ema: bool = False
    ema_decay: float = 0.999

    prior_dist: Literal[
        "isotropic_com", "isotropic_lognorm", "jet_ref_frame", "axis_aligned",
        "axis_aligned_per_jet"
    ] = "isotropic_com"

    # ICP is opt-in. A cache present on a shared PVC must never silently change a baseline.
    use_icp: bool = False
    icp_assignment_cost: Literal["euclidean", "geodesic", "squared_geodesic"] = "euclidean"

    distributed: bool = False

    @model_validator(mode="after")
    def validate_qualification_steps(self):
        if any(step < 0 for step in self.stability_probe_steps):
            raise ValueError("training.stability_probe_steps must be non-negative")
        if self.stability_probe_steps != sorted(set(self.stability_probe_steps)):
            raise ValueError("training.stability_probe_steps must be sorted and unique")
        if (self.max_optimizer_steps is not None and self.stability_probe_steps
                and self.stability_probe_steps[-1] > self.max_optimizer_steps):
            raise ValueError("stability probe step exceeds training.max_optimizer_steps")
        return self


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_samples: int = 50_000
    n_viz_samples: int = 1000
    integration_steps: int = 16
    integration_end_time: float = Field(default=0.99999, gt=0, le=1)
    cfg_guidance_weight: float = 2.0
    use_cfg: bool = False
    use_ema_weights: bool = False
    seed: int = 42
    batch_size: int = 256
    sampler: Literal["euler", "heun"] = "euler"
    # Optional Lorentz-invariant stability control for mass-shell Euler integration.
    # Each nominal interval is subdivided until ||v||_g * dt / m is at most this value.
    mass_shell_max_step_rapidity: float | None = Field(default=None, gt=0)
    mass_shell_max_substeps: int = Field(default=64, ge=1)
    # Optional experiment gate. Training exits non-zero after writing diagnostics
    # when the generated non-finite/exploding trajectory fraction exceeds this value.
    max_invalid_fraction: float | None = Field(default=None, ge=0, le=1)
    # Softer reporting threshold. Exceeding it is recorded as a qualification warning
    # but does not suppress physics metrics.
    warn_invalid_fraction: float | None = Field(default=None, ge=0, le=1)
    stability_probe_samples: int = Field(default=64, ge=1)
    stability_probe_integration_steps: int = Field(default=8, ge=1)
    vf_mode: Literal["cfg", "nocfg", "both", "none"] = "none"
    skip_samples: bool = False
    skip_metrics: bool = False
    prior_dist: Literal[
        "isotropic_com", "isotropic_lognorm", "jet_ref_frame", "axis_aligned",
        "axis_aligned_per_jet"
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
    icp_cache_path: Optional[str] = None
    cache_dir: str = "/mnt/data/caches"


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_workers: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 2) // 2))
    icp_max_iter: int = 1000
    skip_if_exists: bool = True
    n_samples: Optional[int] = None
    # Geometry of the ICP assignment cost. "euclidean" = alternating permutation+Kabsch ICP
    # (the default). "mass_shell" = permutation-only Hungarian on geodesic distance over the
    # mass shell (Phase 4); no rotation, since Euclidean Kabsch is not valid on the shell.
    geometry: Literal["euclidean", "mass_shell"] = "euclidean"
    assignment_cost: Literal["euclidean", "geodesic", "squared_geodesic"] | None = None
    regulator_mass: float = 0.5
    prior_dist: Literal["isotropic_com", "isotropic_lognorm", "jet_ref_frame", "axis_aligned", "axis_aligned_per_jet"] = "isotropic_com"
    seed: int = 42

    @model_validator(mode="after")
    def validate_assignment_cost(self):
        if self.assignment_cost is None:
            self.assignment_cost = ("euclidean" if self.geometry == "euclidean"
                                    else "geodesic")
        allowed = ({"euclidean"} if self.geometry == "euclidean"
                   else {"geodesic", "squared_geodesic"})
        if self.assignment_cost not in allowed:
            raise ValueError(
                f"cache.assignment_cost={self.assignment_cost!r} is incompatible with "
                f"cache.geometry={self.geometry!r}"
            )
        return self


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


class CacheRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


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


# ── Namespace bridges (temporary — removed once train/infer/cache_icp read
#    cfg.section.field directly instead of args.field) ──────────────────────

def train_config_to_namespace(cfg: TrainRunConfig) -> argparse.Namespace:
    return argparse.Namespace(
        output_path=cfg.paths.output_path,
        jet_types=cfg.data.jet_types,
        num_particles=cfg.data.num_particles,
        n_hidden=cfg.model.n_hidden,
        n_layers=cfg.model.n_layers,
        use_residual=cfg.model.use_residual,
        n_train_samples=cfg.training.n_train_samples,
        batch_size=cfg.training.batch_size,
        target_batch_size=cfg.training.target_batch_size,
        cfg_null_dropout_rate=cfg.training.cfg_null_dropout_rate,
        num_epochs=cfg.training.num_epochs,
        epoch_frac=cfg.training.epoch_frac,
        max_optimizer_steps=cfg.training.max_optimizer_steps,
        stability_probe_steps=cfg.training.stability_probe_steps,
        stability_probe_save_checkpoints=cfg.training.stability_probe_save_checkpoints,
        qualification_min_loss_improvement=cfg.training.qualification_min_loss_improvement,
        sigma_min=cfg.training.sigma_min,
        train_space=cfg.training.train_space,
        time_sampling=cfg.training.time_sampling,
        n_samples=cfg.inference.n_samples,
        n_viz_samples=cfg.inference.n_viz_samples,
        integration_steps=cfg.inference.integration_steps,
        integration_end_time=cfg.inference.integration_end_time,
        vf_mode=cfg.inference.vf_mode,
        use_cfg=cfg.inference.use_cfg,
        use_ema_weights=cfg.inference.use_ema_weights,
        cfg_guidance_weight=cfg.inference.cfg_guidance_weight,
        inference_seed=cfg.inference.seed,
        sampler=cfg.inference.sampler,
        mass_shell_max_step_rapidity=cfg.inference.mass_shell_max_step_rapidity,
        mass_shell_max_substeps=cfg.inference.mass_shell_max_substeps,
        max_invalid_fraction=cfg.inference.max_invalid_fraction,
        warn_invalid_fraction=cfg.inference.warn_invalid_fraction,
        stability_probe_samples=cfg.inference.stability_probe_samples,
        stability_probe_integration_steps=cfg.inference.stability_probe_integration_steps,
        skip_metrics=cfg.inference.skip_metrics,
        use_cosine_lr=cfg.training.use_cosine_lr,
        lr_schedule=cfg.training.lr_schedule,
        lr_t0=cfg.training.lr_t0,
        lr_warmup_epochs=cfg.training.lr_warmup_epochs,
        use_hyperbolic=cfg.model.use_hyperbolic,
        hyperbolic_model=cfg.model.hyperbolic_model,
        regulator_mass=cfg.model.regulator_mass,
        backbone=cfg.model.backbone,
        include_mass_condition=cfg.model.include_mass_condition,
        num_attention_heads=cfg.model.num_attention_heads,
        vector_channels=cfg.model.vector_channels,
        velocity_readout_init=cfg.model.velocity_readout_init,
        use_curriculum=cfg.training.use_curriculum,
        use_time_sampling=cfg.training.use_time_sampling,
        use_reference_vectors=cfg.model.use_reference_vectors,
        use_node_scalars=cfg.model.use_node_scalars,
        node_scalar_seed=cfg.model.node_scalar_seed,
        use_attention=cfg.model.use_attention,
        prior_dist=cfg.training.prior_dist,
        use_icp=cfg.training.use_icp,
        icp_assignment_cost=cfg.training.icp_assignment_cost,
        eta_min_factor=cfg.training.eta_min_factor,
        use_ema=cfg.training.use_ema,
        ema_decay=cfg.training.ema_decay,
        use_adaln=cfg.model.use_adaln,
        curriculum_alpha_start=cfg.training.curriculum_alpha_start,
        n_curriculum_buckets=cfg.training.n_curriculum_buckets,
        icp_cache_path=cfg.paths.icp_cache_path,
        cache_dir=cfg.paths.cache_dir,
        resume_weights=cfg.paths.resume_weights,
        distributed=cfg.training.distributed,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )


def infer_config_to_namespace(cfg: InferRunConfig) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_path=cfg.paths.checkpoint_path,
        output_path=cfg.paths.output_path,
        out_dir=cfg.paths.out_dir,
        replay_bundle_path=cfg.paths.replay_bundle_path,
        n_hidden=cfg.model.n_hidden,
        n_layers=cfg.model.n_layers,
        use_residual=cfg.model.use_residual,
        num_particles=cfg.data.num_particles,
        jet_types=cfg.data.jet_types,
        n_samples=cfg.inference.n_samples,
        n_viz_samples=cfg.inference.n_viz_samples,
        integration_steps=cfg.inference.integration_steps,
        integration_end_time=cfg.inference.integration_end_time,
        batch_size=cfg.inference.batch_size,
        cfg_guidance_weight=cfg.inference.cfg_guidance_weight,
        use_cfg=cfg.inference.use_cfg,
        use_ema_weights=cfg.inference.use_ema_weights,
        seed=cfg.inference.seed,
        use_hyperbolic=cfg.model.use_hyperbolic,
        hyperbolic_model=cfg.model.hyperbolic_model,
        regulator_mass=cfg.model.regulator_mass,
        backbone=cfg.model.backbone,
        include_mass_condition=cfg.model.include_mass_condition,
        num_attention_heads=cfg.model.num_attention_heads,
        vector_channels=cfg.model.vector_channels,
        velocity_readout_init=cfg.model.velocity_readout_init,
        sampler=cfg.inference.sampler,
        mass_shell_max_step_rapidity=cfg.inference.mass_shell_max_step_rapidity,
        mass_shell_max_substeps=cfg.inference.mass_shell_max_substeps,
        use_reference_vectors=cfg.model.use_reference_vectors,
        use_node_scalars=cfg.model.use_node_scalars,
        node_scalar_seed=cfg.model.node_scalar_seed,
        use_adaln=cfg.model.use_adaln,
        use_attention=cfg.model.use_attention,
        vf_mode=cfg.inference.vf_mode,
        skip_samples=cfg.inference.skip_samples,
        skip_metrics=cfg.inference.skip_metrics,
        prior_dist=cfg.inference.prior_dist,
    )


def cache_config_to_namespace(cfg: CacheRunConfig) -> argparse.Namespace:
    return argparse.Namespace(
        output_path=cfg.paths.output_path,
        jet_types=cfg.data.jet_types,
        num_particles=cfg.data.num_particles,
        cache_dir=cfg.paths.cache_dir,
        icp_cache_path=cfg.paths.icp_cache_path,
        n_samples=cfg.cache.n_samples,
        n_workers=cfg.cache.n_workers,
        icp_max_iter=cfg.cache.icp_max_iter,
        skip_if_exists=cfg.cache.skip_if_exists,
        geometry=cfg.cache.geometry,
        assignment_cost=cfg.cache.assignment_cost,
        regulator_mass=cfg.cache.regulator_mass,
        prior_dist=cfg.cache.prior_dist,
        seed=cfg.cache.seed,
    )


def generation_controls_from_namespace(args: argparse.Namespace) -> dict:
    """Return generation controls shared by train-time and standalone evaluation."""
    controls = {
        "use_cfg": args.use_cfg,
        "cfg_guidance_weight": args.cfg_guidance_weight,
        "use_hyperbolic": args.use_hyperbolic,
        "hyperbolic_model": args.hyperbolic_model,
        "regulator_mass": args.regulator_mass,
        "use_reference_vectors": args.use_reference_vectors,
        "sampler": args.sampler,
        "mass_shell_max_step_rapidity": args.mass_shell_max_step_rapidity,
        "mass_shell_max_substeps": args.mass_shell_max_substeps,
        "integration_end_time": args.integration_end_time,
        "prior_dist": args.prior_dist,
    }
    # Older callers/tests construct a minimal Namespace. Only v2-aware namespaces carry
    # these controls, preserving the exact legacy kwargs contract.
    if hasattr(args, "backbone"):
        controls["backbone"] = args.backbone
    if hasattr(args, "include_mass_condition"):
        controls["include_mass_condition"] = args.include_mass_condition
    return controls


def run_config_dict(cfg: TrainRunConfig, final_scale: float) -> dict:
    """Checkpoint-embeddable dict matching the pre-config `run_config` format
    (architecture-only keys), for backward compatibility with older loaders."""
    return {
        "num_particles": cfg.data.num_particles,
        "n_layers": cfg.model.n_layers,
        "n_hidden": cfg.model.n_hidden,
        "use_residual": cfg.model.use_residual,
        "include_pt": True,
        "use_reference_vectors": cfg.model.use_reference_vectors,
        "use_node_scalars": cfg.model.use_node_scalars,
        "node_scalar_seed": cfg.model.node_scalar_seed,
        "use_adaln": cfg.model.use_adaln,
        "use_attention": cfg.model.use_attention,
        "backbone": cfg.model.backbone,
        "include_mass_condition": cfg.model.include_mass_condition,
        "num_attention_heads": cfg.model.num_attention_heads,
        "vector_channels": cfg.model.vector_channels,
        "velocity_readout_init": cfg.model.velocity_readout_init,
        "regulator_mass": cfg.model.regulator_mass,
        "jet_types": cfg.data.jet_types,
        "final_scale": float(final_scale),
    }
