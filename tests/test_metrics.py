import numpy as np
import pytest

from freshretail_xgb.metrics import wape, wpe


def test_wape():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 1.0, 5.0])
    assert wape(actual, predicted) == pytest.approx(3.0 / 6.0)


def test_wpe_sign():
    actual = np.array([2.0, 2.0])
    predicted = np.array([1.0, 2.0])
    assert wpe(actual, predicted) == pytest.approx(-0.25)


def test_shape_validation():
    with pytest.raises(ValueError):
        wape([1, 2], [1])


def test_zero_denominator():
    with pytest.raises(ZeroDivisionError):
        wape([0, 0], [0, 1])
