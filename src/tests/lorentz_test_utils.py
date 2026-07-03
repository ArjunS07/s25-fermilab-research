"""Shared helpers for CPU equivariance / reference-vector tests.

All 4-vectors are ordered (E, px, py, pz) with metric signature (+, -, -, -), matching
``util/minkowski_utils.py``. Lorentz transforms act as ``x'_mu = Lambda_mu_nu x_nu``, so a
batch ``x (..., 4)`` transforms as ``x @ Lambda.T``.
"""
import math

import torch

from models.LEFT_JeN import LEFTJeN


def rotation_4x4(axis: str, angle: float, dtype=torch.float64) -> torch.Tensor:
    """Spatial rotation embedded in 4x4 (time component untouched)."""
    c, s = math.cos(angle), math.sin(angle)
    R = torch.eye(4, dtype=dtype)
    if axis == "x":       # rotate (py, pz)
        R[2, 2], R[2, 3], R[3, 2], R[3, 3] = c, -s, s, c
    elif axis == "y":     # rotate (pz, px)
        R[3, 3], R[3, 1], R[1, 3], R[1, 1] = c, -s, s, c
    elif axis == "z":     # rotate (px, py)
        R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    else:
        raise ValueError(axis)
    return R


def boost_4x4(axis: str, rapidity: float, dtype=torch.float64) -> torch.Tensor:
    """Lorentz boost along a spatial axis with the given rapidity."""
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    L = torch.eye(4, dtype=dtype)
    idx = {"x": 1, "y": 2, "z": 3}[axis]
    L[0, 0] = ch
    L[0, idx] = sh
    L[idx, 0] = sh
    L[idx, idx] = ch
    return L


def random_proper_transform(seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """A generic proper orthochronous transform: rotations composed with a modest boost.

    Momenta/rapidities are kept small so the network's ±1e3 invariant clamps never activate
    (they would only ever break equivariance, never fix it)."""
    g = torch.Generator().manual_seed(seed)
    angles = (torch.rand(3, generator=g) * 2 - 1) * math.pi
    L = rotation_4x4("x", float(angles[0]), dtype)
    L = rotation_4x4("y", float(angles[1]), dtype) @ L
    L = rotation_4x4("z", float(angles[2]), dtype) @ L
    xi = float((torch.rand(1, generator=g) * 2 - 1) * 0.3)
    L = boost_4x4("x", xi, dtype) @ L
    return L


def apply_transform(vecs: torch.Tensor, Lambda: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 Lorentz transform to every 4-vector in ``vecs (..., 4)``."""
    return vecs @ Lambda.transpose(-1, -2)


def build_model(use_reference_vectors=False, use_node_scalars=False, seed=0):
    """Small double-precision model in eval mode (dropout off) for deterministic checks."""
    torch.manual_seed(seed)
    model = LEFTJeN(
        max_num_jet_types=5,
        max_particles=8,
        embed_dim=16,
        num_layers=2,
        message_dim=16,
        hidden_dim=16,
        include_pt=True,
        use_reference_vectors=use_reference_vectors,
        use_node_scalars=use_node_scalars,
        node_scalar_dim=16,
    )
    return model.double().eval()


def sample_inputs(batch=2, n_real=6, max_particles=8, jet_axis="z", seed=1, dtype=torch.float64):
    """Build a small physical batch.

    Returns x, t, jet_conditions, mask, ref_vectors. The jet-momentum reference is aligned
    with ``jet_axis`` (used by the residual-SO(2) test). One batch element has a padded row
    to exercise masking.
    """
    g = torch.Generator().manual_seed(seed)
    max_p = max_particles

    # Massless-ish particles: E = |p|, small momenta.
    p3 = (torch.rand(batch, max_p, 3, generator=g, dtype=dtype) * 2 - 1) * 0.5
    E = p3.norm(dim=-1, keepdim=True)
    x = torch.cat([E, p3], dim=-1)

    mask = torch.zeros(batch, max_p, dtype=dtype)
    mask[:, :n_real] = 1.0
    mask[0, n_real - 1] = 0.0  # extra padded row in batch element 0
    x = x * mask.unsqueeze(-1)

    t = torch.rand(batch, generator=g, dtype=dtype)

    onehot = torch.zeros(batch, 5, dtype=dtype)
    onehot[:, 0] = 1.0  # gluon
    n_particles = mask.sum(dim=1, keepdim=True)
    pt = torch.rand(batch, 1, generator=g, dtype=dtype)
    jet_conditions = torch.cat([onehot, n_particles, pt], dim=-1)

    # References: e_t = (1,0,0,0); jet 4-momentum aligned to jet_axis.
    e_t = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype).expand(batch, 1, 4)
    axis_idx = {"x": 1, "y": 2, "z": 3}[jet_axis]
    jet_p = torch.zeros(batch, 1, 4, dtype=dtype)
    jet_p[..., 0] = 3.0            # jet energy
    jet_p[..., axis_idx] = 2.5     # momentum along the axis
    ref_vectors = torch.cat([e_t, jet_p], dim=1)  # (B, 2, 4)

    return x, t, jet_conditions, mask, ref_vectors
