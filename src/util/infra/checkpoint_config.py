"""Checkpoint architecture reconstruction + checkpoint-dict assembly.

Shared by training (``build_checkpoint``/``build_run_config``) and inference
(``resolve_architecture``).
"""

# Architecture keys reconstructed from a checkpoint's embedded full_config. Only the
# fields the mass-shell GNN actually consumes; extra keys in older checkpoints are ignored.
ARCH_KEYS = (
    "n_hidden", "n_layers", "regulator_mass",
    "use_reference_vectors",
    "architecture", "flow_geometry", "reference_mode", "scalar_init_mode",
    "particle_readout_mode", "geometry_mode", "field_degree_normalization",
    "inject_condition_time_each_block",
)


def resolve_architecture(model_cfg, checkpoint):
    """Overlay a checkpoint's embedded architecture onto ``model_cfg`` (a ModelConfig).

    Returns ``(updated_model_cfg, mismatches)``. If the checkpoint has no self-describing
    ``full_config`` (older raw state dicts), returns ``model_cfg`` unchanged.
    """
    if not isinstance(checkpoint, dict) or not checkpoint.get("full_config"):
        return model_cfg, {}
    saved = checkpoint["full_config"].get("model", {})
    updates = {k: saved[k] for k in ARCH_KEYS if k in saved}
    mismatches = {k: (getattr(model_cfg, k, None), v)
                  for k, v in updates.items() if getattr(model_cfg, k, None) != v}
    return (model_cfg.model_copy(update=updates) if updates else model_cfg), mismatches


def build_run_config(cfg, final_scale):
    """Architecture-only dict embedded in checkpoints as ``config`` (compat with older loaders)."""
    return {
        "num_particles": cfg.data.num_particles,
        "n_layers": cfg.model.n_layers,
        "n_hidden": cfg.model.n_hidden,
        "include_pt": True,
        "use_reference_vectors": cfg.model.use_reference_vectors,
        "regulator_mass": cfg.model.regulator_mass,
        "backbone": cfg.model.architecture,
        "architecture": cfg.model.architecture,
        "flow_geometry": cfg.model.flow_geometry,
        "reference_mode": cfg.model.reference_mode,
        "scalar_init_mode": cfg.model.scalar_init_mode,
        "particle_readout_mode": cfg.model.particle_readout_mode,
        "geometry_mode": cfg.model.geometry_mode,
        "field_degree_normalization": cfg.model.field_degree_normalization,
        "inject_condition_time_each_block": cfg.model.inject_condition_time_each_block,
        "jet_types": cfg.data.jet_types,
        "final_scale": float(final_scale),
    }


def build_checkpoint(*, model_state, epoch, global_optimizer_step, losses, run_config,
                     full_config, optimizer_state=None, scheduler_state=None,
                     ema_state=None, rng_state=None, extra=None):
    """Assemble a checkpoint dict. Optional pieces are included only when provided, so the
    same helper serves the stability-probe, latest, and final checkpoints."""
    ckpt = {
        "model_state_dict": model_state,
        "epoch": epoch,
        "global_optimizer_step": global_optimizer_step,
        "losses": losses,
        "config": run_config,
        "full_config": full_config,
    }
    if optimizer_state is not None:
        ckpt["optimizer_state_dict"] = optimizer_state
    if scheduler_state is not None:
        ckpt["scheduler_state_dict"] = scheduler_state
    if ema_state is not None:
        ckpt["ema_state_dict"] = ema_state
    if rng_state is not None:
        ckpt["rng_state"] = rng_state
    if extra:
        ckpt.update(extra)
    return ckpt
