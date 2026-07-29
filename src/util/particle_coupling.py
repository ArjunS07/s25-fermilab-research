"""Deterministic within-jet particle couplings for matched flow experiments.

Every function returns ``perm`` with the convention ``paired_x0 = x0[perm]``;
target rows are never mutated. Callers pass real rows only, so padding cannot enter
an assignment. Stable pT ordering uses original row index as the tie-break.
"""

from __future__ import annotations

import numpy as np
import torch


EXPLICIT_COUPLINGS = (
    "exact_geodesic_icp",
    "canonical_pt",
    "random_frozen",
)
RANDOM_ALGORITHM = "numpy-pcg64-seedsequence-v1"


def _pt(x: torch.Tensor) -> np.ndarray:
    values = torch.sqrt(x[:, 1].square() + x[:, 2].square())
    return values.detach().cpu().numpy().astype(np.float64, copy=False)


def stable_descending_pt_order(x: torch.Tensor) -> np.ndarray:
    """Descending-pT order with original row index as an explicit tie-break."""
    indices = np.arange(len(x), dtype=np.int64)
    return np.lexsort((indices, -_pt(x))).astype(np.int32)


def canonical_pt_permutation(x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
    """Pair equal pT ranks while leaving the target tensor in its original order."""
    if len(x0) != len(x1):
        raise ValueError("prior and target must contain the same number of real particles")
    prior_order = stable_descending_pt_order(x0)
    target_order = stable_descending_pt_order(x1)
    perm = np.empty(len(x0), dtype=np.int32)
    perm[target_order] = prior_order
    return perm


def random_frozen_permutation(n_real: int, coupling_seed: int,
                              dataset_index: int) -> np.ndarray:
    """Frozen uniform permutation keyed only by seed and ordered dataset index."""
    if n_real < 0:
        raise ValueError("n_real must be non-negative")
    seed = np.random.SeedSequence([int(coupling_seed), int(dataset_index)])
    return np.random.Generator(np.random.PCG64(seed)).permutation(n_real).astype(np.int32)


def coupling_permutation(mode: str, x0: torch.Tensor, x1: torch.Tensor, *,
                         regulator_mass: float, assignment_cost: str,
                         coupling_seed: int, dataset_index: int) -> np.ndarray:
    """Return the real-row prior permutation for one explicit ablation arm."""
    if mode == "exact_geodesic_icp":
        # Local import avoids a module cycle at cache_icp import time.
        from cache_icp import _permute_geodesic
        perm, _ = _permute_geodesic(x0, x1, regulator_mass, assignment_cost)
        return perm
    if mode == "canonical_pt":
        return canonical_pt_permutation(x0, x1)
    if mode == "random_frozen":
        return random_frozen_permutation(len(x0), coupling_seed, dataset_index)
    raise ValueError(f"unsupported explicit particle coupling: {mode!r}")
