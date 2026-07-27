"""Historical Euclidean and Poincare flow-matching objectives."""

from jetnet.utils import EtaPhiPtE_to_cartesian, cartesian_to_EtaPhiPtE

from util.distributions import hyperbolic_interpolant
from util.hyperbolic import hyperbolic_loss, pushforward


def poincare_flow_loss(*, model, x0, x1, t, mask, conditions, references):
    mask4 = mask.unsqueeze(-1).expand_as(x0)
    state, ball_state, target = hyperbolic_interpolant(x0, x1, t)
    state = state.to(x0.device)
    ball_state = ball_state.to(x0.device)
    target = target * mask4
    prediction = model.forward(
        x=state, t=t, jet_conditions=conditions, mask=mask,
        ref_vectors=references,
    ) * mask4
    tangent_prediction = pushforward(state, prediction) * mask4
    return hyperbolic_loss(tangent_prediction, target, ball_state, mask=mask)


def euclidean_flow_loss(*, model, x0, x1, t, mask, conditions, references,
                        train_space, sigma_min):
    mask4 = mask.unsqueeze(-1).expand_as(x0)
    t_view = t.view(-1, 1, 1)
    if train_space == "polar":
        x0_coordinates = cartesian_to_EtaPhiPtE(x0)
        x1_coordinates = cartesian_to_EtaPhiPtE(x1)
        state = ((1 - (1 - sigma_min) * t_view) * x0_coordinates
                 + t_view * x1_coordinates)
        state = EtaPhiPtE_to_cartesian(state)
    elif train_space == "cartesian":
        state = (1 - (1 - sigma_min) * t_view) * x0 + t_view * x1
    else:
        raise ValueError(f"unknown Euclidean training space {train_space!r}")
    state = state.to(x0.device)
    target = (x1 - (1 - sigma_min) * x0) * mask4
    prediction = model.forward(
        x=state, t=t, jet_conditions=conditions, mask=mask,
        ref_vectors=references,
    ) * mask4
    n_real = mask.sum().clamp(min=1)
    return (target - prediction).square().sum() / (n_real * x0.shape[-1])
