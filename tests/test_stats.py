import numpy as np

from stats import mean_std


def test_mean_std_simple():
    values = np.array([1.0, 2.0, 3.0])
    mean, std = mean_std(values)
    assert mean == 2.0
    assert np.isclose(std, np.sqrt(2.0 / 3.0))


def test_mean_std_empty():
    try:
        mean_std([])
    except ValueError as exc:
        assert "at least one value" in str(exc)
    else:
        raise AssertionError("expected ValueError on empty input")
