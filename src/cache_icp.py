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
    python cache_icp.py --config configs/cache-icp-30.yaml \
                        --set paths.output_path=/mnt/data/output --set cache.n_workers=16
"""

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


from config import CacheRunConfig, build_config, parse_config_cli, cache_config_to_namespace

# NB: data / coordinates / distributions pull in jetnet, so they are imported lazily inside
# __main__ (below). This keeps the assignment worker functions importable — and unit-testable —
# without a jetnet install.


def canonical_cache_path(cache_dir: str, jet_types: list, num_particles: int) -> str:
    """Return the canonical path for an ICP cache given its config."""
    key = "_".join(sorted(jet_types)) + f"_p{num_particles}"
    return os.path.join(cache_dir, key, "icp_cache.pkl")

MAX_N_PARTICLES = 150
RANDOM_SEED = 42


def _align_point_clouds_till_converge(x_0_orig, x1, max_iter=200):
    x_0 = x_0_orig.clone()
    n_real = len(x_0)
    i = 0
    dist = np.linalg.norm(x_0 - x1, axis=1).sum()
    dist_delta = np.inf

    perm_cumulative = np.arange(n_real, dtype=np.int32)
    rot_cumulative = np.eye(3, dtype=np.float64)
    best_dist = dist
    best_x_0 = x_0.clone()
    best_perm = perm_cumulative.copy()
    best_rot = rot_cumulative.copy()

    while i < max_iter and dist_delta > 1e-8:
        cost = cdist(x_0, x1, metric='euclidean')
        # Better for stability
        cost = cost * (1_000. / cost.max())
        _, col_ind = linear_sum_assignment(cost)

        perm_cumulative = perm_cumulative[col_ind]
        x_0 = x_0[col_ind]
        # Align cartesian 3-momenta
        rot, _, _ = R.align_vectors(x1[:, 1:4], x_0[:, 1:4], return_sensitivity=True)
        rot_mat = rot.as_matrix()
        rot_cumulative = rot_mat @ rot_cumulative
        x_0[:, 1:4] = x_0[:, 1:4] @ rot_mat.T

        dist_new = np.linalg.norm(x_0 - x1, axis=1).sum()
        dist_delta = np.abs(dist_new - dist)
        dist = dist_new
        i += 1

        if dist_new < best_dist:
            best_dist = dist_new
            best_x_0 = x_0.clone()
            best_perm = perm_cumulative.copy()
            best_rot = rot_cumulative.copy()

    return best_x_0, best_perm, best_rot

def _permute_geodesic(x_0_real, x_1_real, m):
    """Permutation-only assignment minimising total geodesic distance on the mass shell.

    Both clouds are lifted onto H_m (regulator mass m), the pairwise geodesic-distance matrix
    is built, and a single Hungarian assignment gives the net permutation. No rotation step —
    Euclidean Kabsch is not a valid isometry of the shell (plan Phase 4).

    x_0_real, x_1_real : (n_real, 4) float64 tensors (normalised space).
    Returns (perm, rot) where perm is (n_real,) int32 and rot is the 3x3 identity (permutation
    only), so the cache format is unchanged.
    """
    from util.mass_shell import geodesic_cost_matrix

    n_real = x_0_real.shape[0]
    # Pairwise geodesic cost: cost[i, j] = d(prior_i, target_j). Only real particles enter,
    # so no real particle can ever be assigned to apex-parked padding (masking guard).
    cost = geodesic_cost_matrix(x_0_real, x_1_real, m)             # (n, n)
    cost_np = cost.numpy().astype(np.float64)
    _, col_ind = linear_sum_assignment(cost_np)
    perm = np.arange(n_real, dtype=np.int32)[col_ind]
    return perm, np.eye(3, dtype=np.float32)


# must be top-level for multiprocessing
def _icp_permute_worker(task):
    """
    Find the net permutation and net rotation that align x_0 to x_1 for the
    real (unmasked) particles of one jet using the alternating permutation +
    rotation ICP algorithm (Algorithm 3, https://arxiv.org/abs/2312.07168):

        while not converged:
            Π  = argmin_Π  ||Π(Rz)^T - y^T||   (Hungarian assignment)
            R  = argmin_R  ||R(Πz)^T - y^T||   (Kabsch / align_vectors)

    Both the net permutation and the cumulative rotation matrix are cached.
    At training time a fresh x_0 is drawn from the prior, the stored permutation
    is applied, then the stored rotation is applied to the 3-momenta.

    For geometry == "mass_shell" the alternating step is replaced by a single permutation-only
    Hungarian assignment on geodesic distance over the mass shell (rot stays identity).

    task : (idx, x_0_full, x_1_full, n_real, max_iter, geometry, regulator_mass)
        idx            – global index in the cache array (returned to reconstruct order)
        x_0_full       – (max_particles, 4) float32 numpy array  (prior, normalised)
        x_1_full       – (max_particles, 4) float32 numpy array  (target, normalised)
        n_real         – number of real particles (rest is zero-padding)
        max_iter       – maximum ICP iterations (euclidean geometry only)
        geometry       – "euclidean" or "mass_shell"
        regulator_mass – shell mass m (mass_shell geometry only)

    Returns (idx, perm_full, rot) where:
        perm_full  – (max_particles,) int32: net permutation (identity for padding)
        rot        – (3, 3) float32: cumulative rotation matrix for 3-momenta
    """
    idx, x_0_full, x_1_full, n_real, max_iter, geometry, regulator_mass = task

    max_particles = x_0_full.shape[0]
    perm_full = np.arange(max_particles, dtype=np.int32)
    rot = np.eye(3, dtype=np.float32)

    if n_real == 0:
        return idx, perm_full, rot

    # Only the real particles enter the cost — padding never participates in the assignment.
    x_0_real = x_0_full[:n_real]   # (n_real, 4)
    x_1_real = x_1_full[:n_real]   # (n_real, 4)

    # Use float64 so the geometry / scipy Kabsch solver work without dtype promotion issues.
    x_0_t = torch.from_numpy(x_0_real.astype(np.float64))
    x_1_t = torch.from_numpy(x_1_real.astype(np.float64))

    if geometry == "mass_shell":
        best_perm, best_rot = _permute_geodesic(x_0_t, x_1_t, regulator_mass)
    else:
        _x_0_aligned, best_perm, best_rot = _align_point_clouds_till_converge(x_0_t, x_1_t, max_iter=max_iter)

    perm_full[:n_real] = best_perm
    rot = best_rot.astype(np.float32)
    return idx, perm_full, rot


def _compute_scale(X_transformed):
    """Reproduce train.py's scale computation (unmasked particles only)."""
    mask_flat = X_transformed[:, :, 4].flatten().numpy().astype(bool)
    scales = [
        np.std(np.array(X_transformed[:, :, c].flatten())[mask_flat])
        for c in range(4)
    ]
    return float(np.mean(scales))



if __name__ == "__main__":
    # Heavy, jetnet-pulling deps imported here so the module (and its worker functions) stay
    # importable for unit tests without a jetnet install.
    from data import get_data_path
    from util.coordinates import transform_rel_particle_coordinates_to_cartesian
    from util.distributions import gen_initial_distribution

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    config_path, overrides = parse_config_cli()
    cfg = build_config(CacheRunConfig, config_path, overrides)
    args = cache_config_to_namespace(cfg)

    # ── Compute canonical cache path ──────────────────────────────────────────
    cache_path = canonical_cache_path(args.cache_dir, args.jet_types, args.num_particles)
    logging.info(f"Cache target: {cache_path}")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as _f:
            _existing = pickle.load(_f)
        if "perm_cache" in _existing:
            if args.skip_if_exists:
                logging.info("Cache already exists — skipping computation (--no-skip_if_exists to force).")
                raise SystemExit(0)
        else:
            logging.warning(
                "Existing cache at %s is old format (x_0_cache). Deleting and recomputing.", cache_path
            )
            os.remove(cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

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

    logging.info(f"ICP geometry: {args.geometry}"
                 + (f" (regulator mass m={args.regulator_mass})" if args.geometry == "mass_shell" else ""))
    tasks = [
        (i, x_0_np[i], x_1_np[i], int(n_real_np[i]), args.icp_max_iter,
         args.geometry, args.regulator_mass)
        for i in range(n_total)
    ]

    perm_cache = np.zeros((n_total, args.num_particles), dtype=np.int32)
    rot_cache = np.zeros((n_total, 3, 3), dtype=np.float32)

    logging.info(f"Running ICP on {n_total} jets with {args.n_workers} workers …")
    with Pool(processes=args.n_workers) as pool:
        for idx, perm_full, rot in tqdm(
            pool.imap_unordered(_icp_permute_worker, tasks, chunksize=64),
            total=n_total,
            desc="ICP",
        ):
            perm_cache[idx] = perm_full
            rot_cache[idx] = rot

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "perm_cache": perm_cache,
        "rot_cache": rot_cache,
        "final_scale": final_scale,
        "num_particles": args.num_particles,
        "n_samples": n_total,
        "geometry": args.geometry,
        "regulator_mass": args.regulator_mass,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Saved ICP cache → {cache_path}  perm={perm_cache.shape}  rot={rot_cache.shape}")
