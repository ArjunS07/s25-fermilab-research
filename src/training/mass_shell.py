"""Mass-shell Riemannian flow-matching objective."""

import torch

from util.mass_shell import (
    conditional_vector_field,
    geodesic_interpolant,
    mass_shell_loss,
    project_to_shell,
    pushforward_to_tangent,
)


def mass_shell_flow_loss(*, model, raw_model, x0, x1, t, mask,
                         conditions, references, regulator_mass,
                         tangent_backbone):
    """Construct the geodesic conditional path and evaluate tangent MSE."""
    p0 = project_to_shell(x0, regulator_mass).to(x0.device)
    p1 = project_to_shell(x1, regulator_mass).to(x1.device)
    state = geodesic_interpolant(p0, p1, t, regulator_mass)
    mask4 = mask.to(torch.float64).unsqueeze(-1).expand_as(state)
    target = conditional_vector_field(
        state, p1, t.to(torch.float64), regulator_mass
    ) * mask4

    model_dtype = next(raw_model.parameters()).dtype
    model_state = state if tangent_backbone else state.to(model_dtype)
    model_references = references.to(model_dtype) if references is not None else None
    prediction = model.forward(
        x=model_state, t=t.to(model_dtype),
        jet_conditions=conditions.to(model_dtype), mask=mask,
        ref_vectors=model_references,
    )
    tangent_prediction = pushforward_to_tangent(
        state, prediction.to(torch.float64), regulator_mass
    ) * mask4
    return mass_shell_loss(tangent_prediction, target, mask, regulator_mass)
