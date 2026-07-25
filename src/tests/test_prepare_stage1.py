import json
import pickle

import pytest
import torch

from prepare_stage1 import cache_key, resolve_stage1, stage1_spec, validate_bundle


class _DatasetLike:
    def __init__(self):
        self.particles = torch.zeros(2, 30, 4)
        self.jets = torch.zeros(2, 5)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.particles, self.jets
        return self.particles[index], self.jets[index]


def _legacy_bundle(path, type_ids=(0,), particles=30, with_summary=False):
    data = path / "data"
    data.mkdir(parents=True)
    particle_tensor = torch.zeros(len(type_ids), particles, 4)
    jet_features = torch.zeros(len(type_ids), 5)
    jet_features[:, -1] = torch.tensor(type_ids)
    for split in ("x_train.pkl", "x_test.pkl"):
        with (data / split).open("wb") as handle:
            pickle.dump((particle_tensor, jet_features), handle)
    (path / "jet_attr_model.pth").write_bytes(b"model")
    if with_summary:
        train = path / "train"
        train.mkdir()
        train.joinpath("summary.json").write_text(json.dumps({
            "full_config": {"data": {"jet_types": ["g"], "num_particles": particles}}
        }))


def test_cache_key_is_order_independent_for_same_type_set():
    assert cache_key(stage1_spec(["q", "g"], 30)) == cache_key(stage1_spec(["g", "q"], 30))


def test_v3_cache_spec_is_distinct_and_records_hybrid_flow():
    v2 = stage1_spec(["g"], 30, "v2")
    v3 = stage1_spec(["g"], 30, "v3")
    assert cache_key(v2) != cache_key(v3)
    assert v3["model"]["version"] == "categorical-spline-flow-v3"


def test_imports_validated_legacy_bundle_then_reuses_cache(tmp_path):
    legacy = tmp_path / "legacy"
    _legacy_bundle(legacy)
    cache = tmp_path / "cache"
    first = tmp_path / "first"
    bundle, reused = resolve_stage1(first, cache, ["g"], 30, legacy)
    assert reused
    assert first.joinpath("data").is_symlink()
    assert first.joinpath("jet_attr_model.pth").is_symlink()
    metadata = json.loads(bundle.joinpath("stage1_metadata.json").read_text())
    assert metadata["spec"] == stage1_spec(["g"], 30)

    second = tmp_path / "second"
    same_bundle, reused = resolve_stage1(second, cache, ["g"], 30)
    assert reused and same_bundle == bundle


def test_legacy_summary_avoids_loading_large_pickle(tmp_path):
    legacy = tmp_path / "legacy"
    _legacy_bundle(legacy, with_summary=True)
    legacy.joinpath("data", "x_train.pkl").write_bytes(b"not a pickle")
    validate_bundle(legacy, stage1_spec(["g"], 30), require_metadata=False)


def test_unversioned_dataset_object_is_validated_through_full_slice(tmp_path):
    bundle = tmp_path / "dataset-object"
    data = bundle / "data"
    data.mkdir(parents=True)
    with (data / "x_train.pkl").open("wb") as handle:
        pickle.dump(_DatasetLike(), handle)
    (data / "x_test.pkl").write_bytes(b"present")
    (bundle / "jet_attr_model.pth").write_bytes(b"model")
    validate_bundle(bundle, stage1_spec(["g"], 30), require_metadata=False)


@pytest.mark.parametrize("jet_types,particles", [(["q"], 30), (["g"], 150)])
def test_rejects_incompatible_legacy_bundle(tmp_path, jet_types, particles):
    legacy = tmp_path / "legacy"
    _legacy_bundle(legacy)
    with pytest.raises(ValueError):
        validate_bundle(legacy, stage1_spec(jet_types, particles), require_metadata=False)
