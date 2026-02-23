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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


from data import data_args, get_data_path
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.distributions import gen_initial_distribution

MAX_N_PARTICLES = 150
RANDOM_SEED = 42


def _align_point_clouds_till_converge(x_0_orig, x1, max_iter=200):
    x_0 = x_0_orig.clone()
    i = 0
    dist = np.linalg.norm(x_0 - x1, axis=1).sum()
    dist_delta = np.inf

    best_dist = dist
    best_x_0 = x_0.clone()

    while i < max_iter and dist_delta > 1e-8:
        cost = cdist(x_0, x1, metric='euclidean')
        # Better for stability
        cost = cost * (1_000. / cost.max())
        _, col_ind = linear_sum_assignment(cost)

        x_0 = x_0[col_ind]
        # Align cartesian 3-momenta
        rot, _, _ = R.align_vectors(x1[:, 1:4], x_0[:, 1:4], return_sensitivity=True)
        x_0[:, 1:4] = x_0[:, 1:4] @ rot.as_matrix().T

        dist_new = np.linalg.norm(x_0 - x1, axis=1).sum()
        dist_delta = np.abs(dist_new - dist)
        dist = dist_new
        i += 1

        if dist_new < best_dist:
            best_dist = dist_new
            best_x_0 = x_0.clone()

    return best_x_0

# must be top-level for multiprocessing 
def _icp_permute_worker(task):
    """
    Align x_0 to x_1 for the real (unmasked) particles of one jet using the
    alternating permutation + rotation ICP algorithm (Algorithm 3,
    https://arxiv.org/abs/2312.07168):

        while not converged:
            Π  = argmin_Π  ||Π(Rz)^T - y^T||   (Hungarian assignment)
            R  = argmin_R  ||R(Πz)^T - y^T||   (Kabsch / align_vectors)

    Rotation is applied to the 3-momentum components (px, py, pz) only.

    task : (idx, x_0_full, x_1_full, n_real, max_iter)
        idx       – global index in the cache array (returned to reconstruct order)
        x_0_full  – (max_particles, 4) float32 numpy array  (prior, normalised)
        x_1_full  – (max_particles, 4) float32 numpy array  (target, normalised)
        n_real    – number of real particles (rest is zero-padding)
        max_iter  – maximum ICP iterations

    Returns (idx, x_0_aligned_full)
    """
    idx, x_0_full, x_1_full, n_real, max_iter = task

    if n_real == 0:
        return idx, x_0_full.copy()

    x_0_real = x_0_full[:n_real]   # (n_real, 4)
    x_1_real = x_1_full[:n_real]   # (n_real, 4)

    # align_point_clouds_till_converge expects tensors (calls .clone() internally).
    # Use float64 so scipy's Kabsch solver works without dtype promotion issues.
    x_0_t = torch.from_numpy(x_0_real.astype(np.float64))
    x_1_t = torch.from_numpy(x_1_real.astype(np.float64))

    x_0_aligned = _align_point_clouds_till_converge(x_0_t, x_1_t, max_iter=max_iter)

    x_0_out = np.zeros_like(x_0_full)
    x_0_out[:n_real] = x_0_aligned.numpy().astype(np.float32)
    return idx, x_0_out


def _compute_scale(X_transformed):
    """Reproduce train.py's scale computation (unmasked particles only)."""
    mask_flat = X_transformed[:, :, 4].flatten().numpy().astype(bool)
    scales = [
        np.std(np.array(X_transformed[:, :, c].flatten())[mask_flat])
        for c in range(4)
    ]
    return float(np.mean(scales))



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
    parser.add_argument("--icp_max_iter", type=int, default=1000,
                        help="Maximum ICP iterations per jet (default 1000 )")
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

    tasks = [
        (i, x_0_np[i], x_1_np[i], int(n_real_np[i]), args.icp_max_iter)
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
