"""Flow-matching objective dispatch."""

def flow_matching_loss(*, model, raw_model, config, x0, x1, t, mask,
                       conditions, references):
    from training.mass_shell import mass_shell_flow_loss
    return mass_shell_flow_loss(
        model=model, raw_model=raw_model, x0=x0, x1=x1, t=t,
        mask=mask, conditions=conditions, references=references,
        regulator_mass=config.regulator_mass,
        tangent_backbone=True,
        return_diagnostics=False,
    )


__all__ = ["flow_matching_loss"]
