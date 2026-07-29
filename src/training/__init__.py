"""Geometry-specific flow-matching objectives."""

def flow_matching_loss(*, model, raw_model, config, x0, x1, t, mask,
                       conditions, references):
    """Dispatch to one explicit geometry implementation."""
    if config.use_hyperbolic and config.hyperbolic_model == "mass_shell":
        from training.mass_shell import mass_shell_flow_loss
        return mass_shell_flow_loss(
            model=model, raw_model=raw_model, x0=x0, x1=x1, t=t,
            mask=mask, conditions=conditions, references=references,
            regulator_mass=config.regulator_mass,
            tangent_backbone=config.backbone == "tangent_attention",
            return_diagnostics=getattr(config, "particle_coupling", "legacy") != "legacy",
        )
    if config.use_hyperbolic:
        from training.legacy import poincare_flow_loss
        return poincare_flow_loss(
            model=model, x0=x0, x1=x1, t=t, mask=mask,
            conditions=conditions, references=references,
        )
    from training.legacy import euclidean_flow_loss
    return euclidean_flow_loss(
        model=model, x0=x0, x1=x1, t=t, mask=mask,
        conditions=conditions, references=references,
        train_space=config.train_space, sigma_min=config.sigma_min,
    )


__all__ = ["flow_matching_loss"]
