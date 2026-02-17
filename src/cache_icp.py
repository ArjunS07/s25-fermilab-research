"""
cache_icp.py — Pre-compute ICP-aligned prior distributions for training.

For each jet in the training set we generate an isotropic-CoM prior sample x_0
and find the *permutation* of its particles that minimises the total squared
Cartesian distance to the corresponding target cloud x_1 (Hungarian / linear
assignment). Only the permutation is applied — no rotation or translation —
so the physics of the prior is preserved.

The resulting cache is a tensor of shape (N, max_particles, 4) stored in the
normalized space (after dividing by final_scale, same convention as train.py).
Pass --icp_cache_path to train.py to load it instead of generating x_0 fresh.

Usage:
    python cache_icp.py --output_path /mnt/data/output --num_particles 30 \
                        --n_workers 16
"""

import argparse
import logging
import os
import pickle
from multiprocessing import Pool

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

from data import data_args, get_data_path
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.distributions import gen_initial_distribution

MAX_N_PARTICLES = 150
RANDOM_SEED = 42


# ── Worker (must be top-level for multiprocessing pickling) ──────────────────

def _icp_permute_worker(task):
    """
    Find the permutation of x_0 that minimises total squared Euclidean
    distance to x_1 for the real (unmasked) particles of one jet.

    task : (idx, x_0_full, x_1_full, n_real)
        idx       – global index in the cache array (returned to reconstruct order)
        x_0_full  – (max_particles, 4) float32 numpy array
        x_1_full  – (max_particles, 4) float32 numpy array
        n_real    – number of real particles (rest is zero-padding)

    Returns (idx, x_0_permuted_full)
    """
    idx, x_0_full, x_1_full, n_real = task

    if n_real == 0:
        return idx, x_0_full.copy()

    x_0_real = x_0_full[:n_real]   # (n_real, 4)
    x_1_real = x_1_full[:n_real]   # (n_real, 4)

    # cost[i, j] = ||x_1[i] - x_0[j]||^2
    cost = cdist(x_1_real, x_0_real, metric='sqeuclidean')

    # Normalise for numerical stability (mirrors align_clouds.py convention)
    max_c = cost.max()
    if max_c > 0:
        cost = cost * (1000.0 / max_c)

    # col_ind[i] gives the x_0 particle that best matches x_1[i]
    _, col_ind = linear_sum_assignment(cost)

    x_0_permuted = np.zeros_like(x_0_full)
    x_0_permuted[:n_real] = x_0_real[col_ind]
    return idx, x_0_permuted


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_scale(X_transformed):
    """Reproduce train.py's scale computation (unmasked particles only)."""
    mask_flat = X_transformed[:, :, 4].flatten().numpy().astype(bool)
    scales = [
        np.std(np.array(X_transformed[:, :, c].flatten())[mask_flat])
        for c in range(4)
    ]
    return float(np.mean(scales))


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    parser = argparse.ArgumentParser(description="Pre-compute ICP-aligned prior cache")
    parser.add_argument("--output_path", type=str, default="/mnt/data/output",
                        help="Root output path (same as used for data.py and train.py)")
    parser.add_argument("--num_particles", type=int, default=data_args["num_particles"],
                        help="Max particles per jet (must match the train.py run)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Cap the number of jets to process (default: all training jets)")
    parser.add_argument("--n_workers", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2),
                        help="Number of parallel worker processes")
    parser.add_argument("--cache_filename", type=str, default="icp_cache.pkl",
                        help="Output filename inside output_path")
    args = parser.parse_args()

    # ── Load and transform training data (same pipeline as train.py) ─────────
    data_path = get_data_path(args.output_path)
    logging.info("Loading training data …")
    with open(f"{data_path}/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    X_transformed = transform_rel_particle_coordinates_to_cartesian(X_train).to("cpu")
    if args.num_particles < MAX_N_PARTICLES:
        X_transformed = X_transformed[:, :args.num_particles, :]

    n_total = len(X_transformed)
    if args.n_samples is not None:
        n_total = min(n_total, args.n_samples)
    X_transformed = X_transformed[:n_total]

    # ── Compute scale (identical to train.py) ─────────────────────────────────
    final_scale = _compute_scale(X_transformed)
    logging.info(f"final_scale = {final_scale:.6f}")

    # Scale x_1 into normalised space
    X_scaled = X_transformed.clone()
    X_scaled[:, :, :4] = X_transformed[:, :, :4] / final_scale

    x_1_np = X_scaled[:, :, :4].numpy().astype(np.float32)          # (N, P, 4)
    n_real_np = X_scaled[:, :, 4].sum(dim=1).long().numpy()          # (N,)

    # ── Generate prior x_0 for all jets at once ───────────────────────────────
    logging.info(f"Generating {n_total} prior clouds …")
    # Generate in chunks to keep memory under control
    chunk = 10_000
    x_0_parts = []
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        x_0_parts.append(
            gen_initial_distribution(
                batch_size=end - start,
                num_particles=args.num_particles,
            ).numpy().astype(np.float32)
        )
    x_0_np = np.concatenate(x_0_parts, axis=0)   # (N, P, 4)

    # ── Run ICP permutation in parallel ──────────────────────────────────────
    tasks = [
        (i, x_0_np[i], x_1_np[i], int(n_real_np[i]))
        for i in range(n_total)
    ]

    cache = np.zeros((n_total, args.num_particles, 4), dtype=np.float32)

    logging.info(f"Running Hungarian assignment on {n_total} jets "
                 f"with {args.n_workers} workers …")
    with Pool(processes=args.n_workers) as pool:
        for idx, x_0_permuted in tqdm(
            pool.imap_unordered(_icp_permute_worker, tasks, chunksize=64),
            total=n_total,
            desc="ICP",
        ):
            cache[idx] = x_0_permuted

    # ── Save ──────────────────────────────────────────────────────────────────
    cache_path = os.path.join(args.output_path, args.cache_filename)
    payload = {
        "x_0_cache": cache,
        "final_scale": final_scale,
        "num_particles": args.num_particles,
        "n_samples": n_total,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Saved ICP cache → {cache_path}  shape={cache.shape}")
