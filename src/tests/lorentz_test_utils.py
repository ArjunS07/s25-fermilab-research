"""Shared helpers for CPU equivariance / reference-vector tests.

All 4-vectors are ordered (E, px, py, pz) with metric signature (+, -, -, -), matching
``util/minkowski_utils.py``. Lorentz transforms act as ``x'_mu = Lambda_mu_nu x_nu``, so a
batch ``x (..., 4)`` transforms as ``x @ Lambda.T``.
"""
import math

import torch

from models.mass_shell_gnn import LEFTJeN
from util.geometry.mass_shell import project_to_shell


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


MASS = 1.0  # shared regulator mass for test models/inputs (well-conditioned shell)


def build_model(use_reference_vectors=True, seed=0, hidden_dim=16, num_layers=2,
                regulator_mass=MASS, **_ignored):
    """Small double-precision mass-shell GNN in eval mode for deterministic checks.

    The mass-shell GNN is the single architecture; it always uses typed references and the
    mass condition. Legacy kwargs (use_node_scalars/use_adaln/use_attention) are accepted
    and ignored so historical call sites keep working.
    """
    torch.manual_seed(seed)
    model = LEFTJeN(
        max_num_jet_types=5,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        include_pt=True,
        include_mass_condition=True,
        use_reference_vectors=use_reference_vectors,
        regulator_mass=regulator_mass,
    )
    return model.double().eval()


def sample_inputs(batch=2, n_real=6, max_particles=8, jet_axis="z", seed=1, dtype=torch.float64,
                  mass=MASS):
    """Build a small on-shell physical batch for the mass-shell GNN.

    Returns x (on the mass shell), t, jet_conditions (8-dim, incl. mass), mask, ref_vectors.
    The jet-momentum reference is aligned with ``jet_axis`` (residual-SO(2) test). One batch
    element has a padded row (parked at the shell apex so no zero-vector geometry appears).
    """
    g = torch.Generator().manual_seed(seed)
    max_p = max_particles

    x = project_to_shell(
        (torch.rand(batch, max_p, 4, generator=g, dtype=dtype) * 2 - 1) * 0.5, mass)

    mask = torch.zeros(batch, max_p, dtype=dtype)
    mask[:, :n_real] = 1.0
    mask[0, n_real - 1] = 0.0  # extra padded row in batch element 0
    x[0, n_real - 1] = torch.tensor([mass, 0.0, 0.0, 0.0], dtype=dtype)  # park at apex

    t = torch.rand(batch, generator=g, dtype=dtype)

    onehot = torch.zeros(batch, 5, dtype=dtype)
    onehot[:, 0] = 1.0  # gluon
    n_particles = mask.sum(dim=1, keepdim=True)
    pt = torch.rand(batch, 1, generator=g, dtype=dtype)
    jet_mass = torch.rand(batch, 1, generator=g, dtype=dtype)
    jet_conditions = torch.cat([onehot, n_particles, pt, jet_mass], dim=-1)  # 8 dims

    # References: e_t = (1,0,0,0); jet 4-momentum aligned to jet_axis.
    e_t = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype).expand(batch, 1, 4)
    axis_idx = {"x": 1, "y": 2, "z": 3}[jet_axis]
    jet_p = torch.zeros(batch, 1, 4, dtype=dtype)
    jet_p[..., 0] = 3.0            # jet energy
    jet_p[..., axis_idx] = 2.5     # momentum along the axis
    ref_vectors = torch.cat([e_t, jet_p], dim=1)  # (B, 2, 4)

    return x, t, jet_conditions, mask, ref_vectors
