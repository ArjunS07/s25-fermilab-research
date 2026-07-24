import numpy as np
import pytest

from util.perceptual import frechet_from_features


def test_frechet_identical_features_is_zero():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(100, 8))
    assert frechet_from_features(values, values) == pytest.approx(0.0, abs=1e-9)


def test_frechet_rejects_bad_shapes():
    with pytest.raises(ValueError):
        frechet_from_features(np.zeros((4, 2)), np.zeros((4, 3)))
