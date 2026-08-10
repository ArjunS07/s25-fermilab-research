"""Dependency-free ambient-space ODE helpers for Euclidean flow matching."""

import torch


def _euclidean_velocity(model, state, conditions, masks, t, references,
                        use_cfg=False, guidance_weight=2.0):
    model_dtype = next(model.parameters()).dtype
    batch_t = torch.as_tensor(t, device=state.device, dtype=model_dtype).expand(state.shape[0])
    refs = references.to(model_dtype) if references is not None else None
    velocity = model(
        state.to(model_dtype), batch_t, conditions.to(model_dtype), masks, refs
    )
    if use_cfg:
        unconditional = model(
            state.to(model_dtype), batch_t,
            model.make_null_cond(conditions).to(model_dtype), masks, refs,
        )
        velocity = velocity + guidance_weight * (velocity - unconditional)
    return velocity * masks.unsqueeze(-1).to(velocity.dtype)


def euclidean_ode_step(model, state, conditions, masks, t_start, t_end, *,
                       references=None, sampler="euler", use_cfg=False,
                       guidance_weight=2.0):
    """One ordinary ambient-space Euler or Heun step, with no physical projection."""
    dt = (t_end - t_start).to(state.dtype)
    v0 = _euclidean_velocity(
        model, state, conditions, masks, t_start, references, use_cfg, guidance_weight
    ).to(state.dtype)
    if sampler == "euler":
        out = state + dt * v0
    elif sampler == "heun":
        predictor = state + dt * v0
        v1 = _euclidean_velocity(
            model, predictor, conditions, masks, t_end, references,
            use_cfg, guidance_weight,
        ).to(state.dtype)
        out = state + dt * (v0 + v1) / 2
    else:
        raise ValueError(f"unknown Euclidean sampler: {sampler!r}")
    return out * masks.unsqueeze(-1).to(out.dtype)


__all__ = ["euclidean_ode_step"]
