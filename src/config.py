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
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jet_types: list[str] = Field(default_factory=lambda: ["g", "q", "t"])
    num_particles: int = 150


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


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_train_samples: int = 1_000_000
    batch_size: int = 16
    target_batch_size: int = 256
    cfg_null_dropout_rate: float = 0.2
    num_epochs: int = 100
    epoch_frac: float = 1.0
    sigma_min: float = 1e-4
    train_space: Literal["cartesian", "polar"] = "cartesian"
    time_sampling: Literal["uniform", "power_law", "lognorm"] = "power_law"

    lr: float = 6e-4
    weight_decay: float = 1e-6
    use_cosine_lr: bool = True
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
        "isotropic_com", "isotropic_lognorm", "jet_ref_frame", "axis_aligned"
    ] = "isotropic_com"

    distributed: bool = False


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_samples: int = 50_000
    n_viz_samples: int = 1000
    integration_steps: int = 16
    cfg_guidance_weight: float = 2.0
    batch_size: int = 256
    sampler: Literal["euler", "heun"] = "euler"
    vf_mode: Literal["cfg", "nocfg", "both", "none"] = "both"
    skip_samples: bool = False
    skip_metrics: bool = False


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str = "/mnt/data/output"
    resume_weights: Optional[str] = None
    checkpoint_path: Optional[str] = None
    out_dir: Optional[str] = None
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
    regulator_mass: float = 0.5


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
        sigma_min=cfg.training.sigma_min,
        train_space=cfg.training.train_space,
        time_sampling=cfg.training.time_sampling,
        n_samples=cfg.inference.n_samples,
        n_viz_samples=cfg.inference.n_viz_samples,
        integration_steps=cfg.inference.integration_steps,
        use_cosine_lr=cfg.training.use_cosine_lr,
        lr_t0=cfg.training.lr_t0,
        lr_warmup_epochs=cfg.training.lr_warmup_epochs,
        use_hyperbolic=cfg.model.use_hyperbolic,
        hyperbolic_model=cfg.model.hyperbolic_model,
        regulator_mass=cfg.model.regulator_mass,
        use_curriculum=cfg.training.use_curriculum,
        use_time_sampling=cfg.training.use_time_sampling,
        use_reference_vectors=cfg.model.use_reference_vectors,
        use_node_scalars=cfg.model.use_node_scalars,
        use_attention=cfg.model.use_attention,
        prior_dist=cfg.training.prior_dist,
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
        n_hidden=cfg.model.n_hidden,
        n_layers=cfg.model.n_layers,
        use_residual=cfg.model.use_residual,
        num_particles=cfg.data.num_particles,
        jet_types=cfg.data.jet_types,
        n_samples=cfg.inference.n_samples,
        n_viz_samples=cfg.inference.n_viz_samples,
        integration_steps=cfg.inference.integration_steps,
        batch_size=cfg.inference.batch_size,
        cfg_guidance_weight=cfg.inference.cfg_guidance_weight,
        use_hyperbolic=cfg.model.use_hyperbolic,
        hyperbolic_model=cfg.model.hyperbolic_model,
        regulator_mass=cfg.model.regulator_mass,
        sampler=cfg.inference.sampler,
        use_reference_vectors=cfg.model.use_reference_vectors,
        use_node_scalars=cfg.model.use_node_scalars,
        use_adaln=cfg.model.use_adaln,
        use_attention=cfg.model.use_attention,
        vf_mode=cfg.inference.vf_mode,
        skip_samples=cfg.inference.skip_samples,
        skip_metrics=cfg.inference.skip_metrics,
    )


def cache_config_to_namespace(cfg: CacheRunConfig) -> argparse.Namespace:
    return argparse.Namespace(
        output_path=cfg.paths.output_path,
        jet_types=cfg.data.jet_types,
        num_particles=cfg.data.num_particles,
        cache_dir=cfg.paths.cache_dir,
        n_samples=cfg.cache.n_samples,
        n_workers=cfg.cache.n_workers,
        icp_max_iter=cfg.cache.icp_max_iter,
        skip_if_exists=cfg.cache.skip_if_exists,
        geometry=cfg.cache.geometry,
        regulator_mass=cfg.cache.regulator_mass,
    )


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
        "use_adaln": cfg.model.use_adaln,
        "use_attention": cfg.model.use_attention,
        "jet_types": cfg.data.jet_types,
        "final_scale": float(final_scale),
    }
