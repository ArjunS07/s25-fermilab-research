import random

import numpy as np
import torch
from pydantic import ValidationError

from cache_icp import (_icp_permute_worker, _mass_shell_transport_costs,
                       _permute_geodesic)
from util.particle_coupling import (
    canonical_pt_permutation,
    coupling_permutation,
    random_frozen_permutation,
    stable_descending_pt_order,
)
from config import CacheRunConfig, TrainRunConfig, train_config_to_namespace


def _cloud(pt_values, *, offset=0.0, max_particles=None):
    n = len(pt_values)
    p = max_particles or n
    x = np.zeros((p, 4), dtype=np.float32)
    for i, pt in enumerate(pt_values):
        phi = 0.31 * (i + 1 + offset)
        px, py = pt * np.cos(phi), pt * np.sin(phi)
        pz = 0.07 * (i + offset)
        x[i] = [np.sqrt(px * px + py * py + pz * pz), px, py, pz]
    return x


def test_canonical_pairing_is_stable_and_deterministic_with_ties():
    x0 = torch.from_numpy(_cloud([2.0, 5.0, 5.0, 1.0], offset=1))
    x1 = torch.from_numpy(_cloud([1.0, 3.0, 3.0, 7.0], offset=2))
    x0[1:3, 1:3] = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    x1[1:3, 1:3] = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    assert stable_descending_pt_order(x0).tolist() == [1, 2, 0, 3]
    first = canonical_pt_permutation(x0, x1)
    second = canonical_pt_permutation(x0.clone(), x1.clone())
    assert first.tolist() == [3, 2, 0, 1]
    assert np.array_equal(first, second)


def test_every_arm_excludes_padding_and_preserves_exact_prior_and_target_sets():
    n_real, p = 4, 7
    x0 = _cloud([1.0, 2.0, 3.0, 4.0], offset=1, max_particles=p)
    x1 = _cloud([4.5, 3.5, 2.5, 1.5], offset=2, max_particles=p)
    target_before = x1.copy()
    for mode in ("exact_geodesic_icp", "canonical_pt", "random_frozen"):
        _, perm, rot = _icp_permute_worker((
            11, x0.copy(), x1, n_real, 10, "mass_shell", 0.1,
            "squared_geodesic", mode, 5042,
        ))
        assert np.array_equal(perm[n_real:], np.arange(n_real, p))
        assert np.array_equal(rot, np.eye(3, dtype=np.float32))
        assert sorted(perm[:n_real].tolist()) == list(range(n_real))
        assert np.array_equal(x1, target_before)
        paired = x0[perm[:n_real]]
        assert {tuple(row) for row in paired} == {tuple(row) for row in x0[:n_real]}


def test_random_pairing_is_keyed_and_independent_of_all_global_rng_streams():
    expected = random_frozen_permutation(30, 5042, 123)
    random.seed(9); np.random.seed(10); torch.manual_seed(11)
    _ = random.random(), np.random.rand(100), torch.rand(100)
    actual = random_frozen_permutation(30, 5042, 123)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(random_frozen_permutation(30, 5042, 124), expected)
    assert not np.array_equal(random_frozen_permutation(30, 5043, 123), expected)


def test_random_pairing_uninterrupted_equals_resumed_dataset_sequence():
    uninterrupted = [random_frozen_permutation(12, 5042, i) for i in range(20)]
    before_checkpoint = [random_frozen_permutation(12, 5042, i) for i in range(7)]
    # A resumed process may have arbitrary model/time/dropout RNG state.
    random.seed(999); np.random.seed(998); torch.manual_seed(997)
    after_resume = [random_frozen_permutation(12, 5042, i) for i in range(7, 20)]
    for a, b in zip(uninterrupted, before_checkpoint + after_resume):
        assert np.array_equal(a, b)


def test_canonical_pairing_is_equivariant_to_joint_row_permutation():
    x0 = torch.from_numpy(_cloud([1.0, 7.0, 3.0, 2.0], offset=1))
    x1 = torch.from_numpy(_cloud([8.0, 2.0, 5.0, 1.0], offset=2))
    perm = canonical_pt_permutation(x0, x1)
    row_perm = torch.tensor([2, 0, 3, 1])
    x0p, x1p = x0[row_perm], x1[row_perm]
    paired = x0[torch.from_numpy(perm)]
    paired_p = x0p[torch.from_numpy(canonical_pt_permutation(x0p, x1p))]
    assert torch.equal(paired_p, paired[row_perm])


def test_explicit_exact_arm_matches_legacy_squared_geodesic_result():
    x0 = torch.from_numpy(_cloud([1.0, 4.0, 2.0, 3.0], offset=1)).double()
    x1 = torch.from_numpy(_cloud([3.5, 1.5, 4.5, 2.5], offset=2)).double()
    legacy, legacy_rot = _permute_geodesic(x0, x1, 0.1, "squared_geodesic")
    explicit = coupling_permutation(
        "exact_geodesic_icp", x0, x1, regulator_mass=0.1,
        assignment_cost="squared_geodesic", coupling_seed=5042, dataset_index=0,
    )
    assert np.array_equal(explicit, legacy)
    assert np.array_equal(legacy_rot, np.eye(3, dtype=np.float32))


def test_transport_cost_diagnostic_uses_real_rows_only():
    p = 6
    x0 = _cloud([1.0, 4.0, 2.0], offset=1, max_particles=p)[None]
    x1 = _cloud([3.5, 1.5, 4.5], offset=2, max_particles=p)[None]
    sums, means = _mass_shell_transport_costs(x0, x1, np.array([3]), 0.1)
    x0[:, 3:] = 1e20
    x1[:, 3:] = -1e20
    changed_sums, changed_means = _mass_shell_transport_costs(x0, x1, np.array([3]), 0.1)
    assert np.array_equal(sums, changed_sums)
    assert np.array_equal(means, changed_means)
    assert np.allclose(sums, means * 3)


def test_every_coupling_reconstructs_from_checkpoint_full_config():
    for mode in ("exact_geodesic_icp", "canonical_pt", "random_frozen"):
        assignment = "squared_geodesic"
        cfg = TrainRunConfig.model_validate({
            "training": {
                "use_icp": True,
                "icp_assignment_cost": assignment,
                "particle_coupling": mode,
                "coupling_seed": 8765,
            }
        })
        checkpoint = {"full_config": cfg.model_dump()}
        restored = TrainRunConfig.model_validate(checkpoint["full_config"])
        args = train_config_to_namespace(restored)
        assert args.particle_coupling == mode
        assert args.coupling_seed == 8765


def test_explicit_coupling_config_rejects_stale_or_incompatible_fields():
    with np.testing.assert_raises(ValidationError):
        TrainRunConfig.model_validate({"training": {"particle_couplng": "canonical_pt"}})
    with np.testing.assert_raises(ValidationError):
        TrainRunConfig.model_validate({"training": {"particle_coupling": "canonical_pt"}})
    with np.testing.assert_raises(ValidationError):
        CacheRunConfig.model_validate({
            "cache": {"geometry": "euclidean", "particle_coupling": "random_frozen"}
        })
