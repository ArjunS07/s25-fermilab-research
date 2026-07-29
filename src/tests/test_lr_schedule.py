import pytest
import torch

from util.lr_schedule import (
    build_epoch_scheduler,
    build_step_scheduler,
    learning_rate_multiplier,
)


def test_warmup_changes_smoothly_per_optimizer_step():
    values = [
        learning_rate_multiplier(
            step, total_steps=100_000, warmup_steps=5_000,
            eta_min_factor=0.03, schedule="monotonic_cosine",
        )
        for step in range(5_002)
    ]
    assert values[0] == 1e-6
    assert max(b - a for a, b in zip(values, values[1:])) < 0.00021
    assert values[4_999] < values[5_000]
    assert values[5_000] == 1.0


def test_scheduler_steps_after_each_optimizer_update_and_roundtrips():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=6e-4)
    scheduler = build_step_scheduler(
        optimizer, total_steps=100, steps_per_epoch=10, warmup_epochs=2,
        eta_min_factor=0.03, schedule="monotonic_cosine", restart_epoch=0,
    )
    initial = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    assert initial == pytest.approx(6e-10)
    assert initial < optimizer.param_groups[0]["lr"] < 6e-4

    state = scheduler.state_dict()
    restored = build_step_scheduler(
        optimizer, total_steps=100, steps_per_epoch=10, warmup_epochs=2,
        eta_min_factor=0.03, schedule="monotonic_cosine", restart_epoch=0,
    )
    restored.load_state_dict(state)
    assert restored.last_epoch == scheduler.last_epoch


def test_explicit_optimizer_warmup_steps_override_dataset_epoch_size():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = build_step_scheduler(
        optimizer, total_steps=500, steps_per_epoch=40, warmup_epochs=10,
        warmup_steps=200, eta_min_factor=0.03,
        schedule="monotonic_cosine", restart_epoch=0,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-10)
    for _ in range(200):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_epoch_scheduler_reproduces_historical_piecewise_warmup():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=6e-4)
    scheduler = build_epoch_scheduler(
        optimizer, total_epochs=202, warmup_epochs=10,
        eta_min_factor=0.03, schedule="monotonic_cosine", restart_epoch=0,
    )
    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed == pytest.approx([6e-10, 6.000054e-5, 0.00012000048,
                                      0.00018000042, 0.00024000036])
