"""Evaluation metrics used in the FreshRetailNet experiments."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Inputs must contain only finite values")
    return actual, predicted


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error."""
    actual, predicted = _arrays(y_true, y_pred)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        raise ZeroDivisionError("WAPE is undefined when total absolute target is zero")
    return float(np.abs(predicted - actual).sum() / denominator)


def wpe(y_true, y_pred) -> float:
    """Weighted percentage error; negative values indicate underprediction."""
    actual, predicted = _arrays(y_true, y_pred)
    denominator = actual.sum()
    if denominator == 0:
        raise ZeroDivisionError("WPE is undefined when total target is zero")
    return float((predicted - actual).sum() / denominator)
