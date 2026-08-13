import pytest
import torch

from util.data.jet_attributes import generate_masks


def test_generate_masks_are_vectorized_trailing_float_masks():
    counts = torch.tensor([0, 1, 3, 5])
    masks = generate_masks(counts, max_particles_per_jet=5, device="cpu")

    assert masks.dtype == torch.float32
    assert masks.device.type == "cpu"
    assert torch.equal(masks, torch.tensor([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.float32))


def test_generate_masks_rejects_counts_above_capacity():
    with pytest.raises(AssertionError):
        generate_masks(torch.tensor([6]), max_particles_per_jet=5, device="cpu")
