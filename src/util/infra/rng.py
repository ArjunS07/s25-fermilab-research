"""Checkpoint helpers for reproducible stochastic training streams."""
import random
import numpy as np
import torch
from contextlib import contextmanager

def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def _cpu_byte_tensor(value):
    """Normalize RNG state loaded with a CUDA ``map_location``.

    ``torch.load(..., map_location=device)`` moves every tensor in a checkpoint,
    including CPU RNG-state tensors, onto the training GPU.  PyTorch's RNG restore
    APIs require CPU ByteTensors even when restoring CUDA generator state.
    """
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_tensor(state["torch"]))
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_cpu_byte_tensor(item) for item in state["cuda"]])

def keyed_seed(base_seed: int, epoch: int, minibatch: int = 0, rank: int = 0) -> int:
    """Stable non-overlapping key for a named stochastic stream."""
    return int(base_seed + 1_000_003 * epoch + 10_007 * minibatch + 97 * rank)

@contextmanager
def keyed_torch_rng(base_seed: int, epoch: int, minibatch: int, rank: int, device):
    """Run random tensor creation without consuming any other Torch RNG stream."""
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if torch.device(device).type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices):
        seed = keyed_seed(base_seed, epoch, minibatch, rank)
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed(seed)
        yield
