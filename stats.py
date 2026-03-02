"""Small numerical helpers for pytest demos."""

from __future__ import annotations

import numpy as np


def mean_std(values: np.ndarray | list[float]) -> tuple[float, float]:
    """Return mean and population standard deviation for the input values."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("mean_std expects at least one value")
    return float(array.mean()), float(array.std(ddof=0))
