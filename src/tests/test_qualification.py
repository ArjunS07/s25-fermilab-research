import numpy as np
import pytest

from util.metrics.qualification import loss_improvement_summary, optimizer_limit_reached


def test_optimizer_limit_is_exact_and_optional():
    assert not optimizer_limit_reached(1999, 2000)
    assert optimizer_limit_reached(2000, 2000)
    assert optimizer_limit_reached(2001, 2000)
    assert not optimizer_limit_reached(999999, None)


def test_loss_improvement_uses_disjoint_endpoint_windows():
    values = np.concatenate([np.full(200, 10.0), np.full(200, 8.0)])
    report = loss_improvement_summary(values)
    assert report["loss_window"] == 200
    assert report["first_loss_median"] == 10.0
    assert report["final_loss_median"] == 8.0
    assert report["loss_improvement_fraction"] == pytest.approx(0.2)
    assert report["losses_finite"]


def test_loss_improvement_flags_nonfinite_history():
    report = loss_improvement_summary([1.0, np.nan])
    assert not report["losses_finite"]
    with pytest.raises(ValueError):
        loss_improvement_summary([])
