import random

import numpy as np
import torch

from util.rng import (capture_rng_state, keyed_seed, keyed_torch_rng,
                      restore_rng_state)


def test_keyed_streams_do_not_depend_on_model_initialization_consumption():
    def realized(init_width):
        torch.manual_seed(42)
        torch.randn(init_width, init_width)
        order_gen = torch.Generator().manual_seed(keyed_seed(1042, 3, 1, 0))
        order = torch.randperm(100, generator=order_gen)
        with keyed_torch_rng(2042, 3, 7, 0, "cpu"):
            times = torch.rand(16)
        with keyed_torch_rng(3042, 3, 7, 0, "cpu"):
            dropout = torch.rand(16)
        return order, times, dropout
    small = realized(16)
    large = realized(256)
    assert all(torch.equal(a, b) for a, b in zip(small, large))


def test_rng_checkpoint_roundtrip_covers_python_numpy_and_torch():
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(4))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_keyed_partial_epoch_resume_matches_uninterrupted_updates():
    def run(model, optimizer, start_batch, stop_batch):
        for batch in range(start_batch, stop_batch):
            with keyed_torch_rng(2042, 5, batch, 0, "cpu"):
                x, target = torch.randn(8, 3), torch.randn(8, 2)
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(x), target).backward()
            optimizer.step()

    torch.manual_seed(9)
    full = torch.nn.Linear(3, 2)
    split = torch.nn.Linear(3, 2); split.load_state_dict(full.state_dict())
    full_opt = torch.optim.AdamW(full.parameters(), lr=1e-3)
    split_opt = torch.optim.AdamW(split.parameters(), lr=1e-3)
    run(full, full_opt, 0, 10)
    run(split, split_opt, 0, 4)
    checkpoint_model = {k: v.clone() for k, v in split.state_dict().items()}
    checkpoint_opt = split_opt.state_dict()
    resumed = torch.nn.Linear(3, 2); resumed.load_state_dict(checkpoint_model)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_opt.load_state_dict(checkpoint_opt)
    run(resumed, resumed_opt, 4, 10)
    for key, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
