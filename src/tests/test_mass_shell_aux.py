import pytest
import torch

from tests.lorentz_test_utils import apply_transform, random_proper_transform
from util.mass_shell import project_to_shell, pushforward_to_tangent
from util.mass_shell_aux import (
    auxiliary_warmup,
    gram_transport_loss,
    total_momentum_transport_loss,
)


def sample_fields(m=0.1, seed=4):
    generator = torch.Generator().manual_seed(seed)
    spatial = torch.randn(3, 5, 3, generator=generator, dtype=torch.float64)
    state = project_to_shell(
        torch.cat([torch.zeros(3, 5, 1, dtype=torch.float64), spatial], -1), m
    )
    raw_a = torch.randn(3, 5, 4, generator=generator, dtype=torch.float64)
    raw_b = torch.randn(3, 5, 4, generator=generator, dtype=torch.float64)
    prediction = pushforward_to_tangent(state, raw_a, m)
    target = pushforward_to_tangent(state, raw_b, m)
    mask = torch.tensor([
        [1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0],
    ], dtype=torch.float64)
    prediction = prediction * mask.unsqueeze(-1)
    target = target * mask.unsqueeze(-1)
    lab_time = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).expand(3, -1)
    return state, prediction, target, mask, lab_time


@pytest.mark.parametrize("m", [0.1, 0.03])
def test_auxiliary_losses_zero_at_target_and_have_finite_gradients(m):
    state, _, target, mask, lab_time = sample_fields(m=m)
    prediction = target.detach().clone().requires_grad_(True)
    loss = (
        gram_transport_loss(state, prediction, target, mask)
        + total_momentum_transport_loss(prediction, target, mask, lab_time)
    )
    loss.backward()
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert torch.isfinite(prediction.grad).all()


def test_auxiliary_losses_are_jointly_lorentz_invariant():
    state, prediction, target, mask, lab_time = sample_fields()
    transform = random_proper_transform(seed=17)
    gram = gram_transport_loss(state, prediction, target, mask)
    gram_l = gram_transport_loss(
        apply_transform(state, transform),
        apply_transform(prediction, transform),
        apply_transform(target, transform),
        mask,
    )
    total = total_momentum_transport_loss(
        prediction, target, mask, lab_time
    )
    total_l = total_momentum_transport_loss(
        apply_transform(prediction, transform),
        apply_transform(target, transform),
        mask,
        apply_transform(lab_time, transform),
    )
    assert torch.allclose(gram, gram_l, atol=1e-7, rtol=1e-6)
    assert torch.allclose(total, total_l, atol=1e-7, rtol=1e-6)


def test_auxiliary_losses_are_permutation_invariant_and_ignore_padding():
    state, prediction, target, mask, lab_time = sample_fields()
    permutation = torch.tensor([2, 0, 4, 1, 3])
    base_g = gram_transport_loss(state, prediction, target, mask)
    perm_g = gram_transport_loss(
        state[:, permutation], prediction[:, permutation],
        target[:, permutation], mask[:, permutation],
    )
    base_t = total_momentum_transport_loss(prediction, target, mask, lab_time)
    perm_t = total_momentum_transport_loss(
        prediction[:, permutation], target[:, permutation],
        mask[:, permutation], lab_time,
    )
    assert torch.allclose(base_g, perm_g)
    assert torch.allclose(base_t, perm_t)

    changed = prediction.clone()
    changed[mask == 0] = 1e9
    assert torch.allclose(
        base_g, gram_transport_loss(state, changed, target, mask)
    )
    assert torch.allclose(
        base_t, total_momentum_transport_loss(changed, target, mask, lab_time)
    )


def test_auxiliary_warmup_is_continuation_local():
    assert auxiliary_warmup(100_000, 100_000, 2_000) == 0.0
    assert auxiliary_warmup(101_000, 100_000, 2_000) == 0.5
    assert auxiliary_warmup(102_000, 100_000, 2_000) == 1.0
