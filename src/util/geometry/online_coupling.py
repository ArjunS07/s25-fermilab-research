"""Online per-batch geodesic ICP coupling for fresh-noise flow matching.

Standard conditional flow matching draws a fresh prior sample ``x0`` every step and
pairs it with the target ``x1`` under some coupling. This module computes the
mass-shell geodesic optimal-transport coupling (minibatch OT-CFM) *online*, so
training never reuses a frozen per-example noise/pairing and cannot memorise a
finite bundle of paths. It replaces the precomputed ICP cache: the assignment math
is identical to the old cache (a single squared-geodesic Hungarian over the real
particles), only now applied to freshly drawn noise on every batch.

Convention (matching the retired cache): ``paired_x0 = x0[perm]``; the target ``x1``
is never mutated; only real (unmasked) particles enter the assignment, so padding
can never be paired to a real particle.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from util.geometry.mass_shell import (
    geodesic_cost_matrix,
    geodesic_distance,
    project_to_shell,
)


def _batched_geodesic_cost_matrices(
    x0: torch.Tensor, x1: torch.Tensor, regulator_mass: float
) -> torch.Tensor:
    """Return all per-jet pairwise costs without changing per-pair arithmetic."""
    p0 = project_to_shell(x0.to(torch.float64), regulator_mass)
    p1 = project_to_shell(x1.to(torch.float64), regulator_mass)
    return geodesic_distance(p0.unsqueeze(2), p1.unsqueeze(1), regulator_mass)


def geodesic_permutation(x0_real: torch.Tensor, x1_real: torch.Tensor,
                         regulator_mass: float,
                         assignment_cost: str = "squared_geodesic") -> np.ndarray:
    """Real-row prior permutation minimising mass-shell geodesic transport for one jet.

    Returns ``perm`` of shape ``(n_real,)`` int32 with the convention
    ``paired_x0 = x0_real[perm]`` — i.e. position ``k`` holds the prior particle
    Hungarian-paired with target ``x1_real[k]``.
    """
    # Compute the assignment geometry in float64 for parity with the retired cache,
    # regardless of the (typically float32) training dtype of the inputs.
    cost = geodesic_cost_matrix(x0_real.double(), x1_real.double(), regulator_mass)
    if assignment_cost == "squared_geodesic":
        cost = cost.square()
    elif assignment_cost != "geodesic":
        raise ValueError(f"unsupported mass-shell assignment cost: {assignment_cost!r}")
    cost_np = cost.detach().cpu().numpy().astype(np.float64)
    # linear_sum_assignment returns (row=x0 index, col=x1 index). We need position k to
    # hold the x0 item paired with x1[k], i.e. the inverse of col_ind — hence argsort.
    _, col_ind = linear_sum_assignment(cost_np)
    return np.argsort(col_ind).astype(np.int32)


def online_geodesic_coupling(x0: torch.Tensor, x1: torch.Tensor, mask: torch.Tensor,
                             regulator_mass: float,
                             assignment_cost: str = "squared_geodesic") -> torch.Tensor:
    """Align a batch of fresh prior clouds to their targets by per-jet geodesic ICP.

    x0, x1 : ``(B, P, 4)`` prior / target 4-vectors in normalised space.
    mask   : ``(B, P)`` float, ``1`` = real particle, ``0`` = trailing padding.

    Returns ``paired_x0`` ``(B, P, 4)``: the real rows of each jet permuted to their
    Hungarian partner in ``x1``; padding rows are left untouched. ``x1`` is never
    mutated. Real particles are trailing-padded (padding last), so the first
    ``n_real`` rows are the real ones.
    """
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {tuple(x0.shape)} vs {tuple(x1.shape)}")
    # The CPU implementation deliberately stays per-jet. A batched float64 PyTorch cost
    # calculation recruits the whole host thread pool; under the 2.4-core NRP CFS quota that
    # exhausted the quota early in almost every period and starved the GPU. On CUDA, build the
    # dense costs on-device and transfer only the resulting small cost tensor for SciPy's
    # Hungarian solve.
    n_real = mask.detach().to(device="cpu", dtype=torch.long).sum(dim=1).tolist()
    if max(n_real, default=0) <= 1:
        return x0.clone()

    if x0.device.type == "cpu":
        paired = x0.clone()
        for b, n in enumerate(n_real):
            if n > 1:
                perm = geodesic_permutation(
                    x0[b, :n], x1[b, :n], regulator_mass, assignment_cost
                )
                index = torch.as_tensor(perm, device=x0.device, dtype=torch.long)
                paired[b, :n] = x0[b, :n][index]
        return paired

    costs = _batched_geodesic_cost_matrices(x0.detach(), x1.detach(), regulator_mass)
    if assignment_cost == "squared_geodesic":
        costs = costs.square()
    elif assignment_cost != "geodesic":
        raise ValueError(f"unsupported mass-shell assignment cost: {assignment_cost!r}")
    costs_np = costs.cpu().numpy()

    # Build padded identity permutations on the host, fill the real prefixes with their
    # Hungarian solutions, then apply every jet permutation in one device gather.
    permutations = torch.arange(x0.shape[1], dtype=torch.long).expand(x0.shape[0], -1).clone()
    for b, n in enumerate(n_real):
        if n > 1:
            _, col_ind = linear_sum_assignment(costs_np[b, :n, :n])
            permutations[b, :n] = torch.from_numpy(np.argsort(col_ind).astype(np.int64))
    gather_index = permutations.to(x0.device).unsqueeze(-1).expand_as(x0)
    return torch.gather(x0, dim=1, index=gather_index)
