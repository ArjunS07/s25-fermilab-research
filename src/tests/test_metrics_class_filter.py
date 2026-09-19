import pytest
import torch

from util.data.jet_attributes import select_configured_jets


def _mixed_test_set():
    particles = torch.arange(4 * 3 * 4, dtype=torch.float32).reshape(4, 3, 4)
    features = torch.zeros(4, 5)
    features[:, 4] = torch.tensor([0, 1, 2, 1])
    return particles, features


def test_select_configured_test_jets_filters_global_labels():
    particles, features = select_configured_jets(_mixed_test_set(), ["q"])

    assert particles.shape[0] == 2
    assert features[:, 4].tolist() == [1.0, 1.0]


def test_select_configured_test_jets_preserves_configured_union():
    particles, features = select_configured_jets(_mixed_test_set(), ["g", "t"])

    assert particles.shape[0] == 2
    assert features[:, 4].tolist() == [0.0, 2.0]


def test_select_configured_test_jets_rejects_absent_class():
    with pytest.raises(ValueError, match="contains no samples"):
        select_configured_jets(_mixed_test_set(), ["w"])
