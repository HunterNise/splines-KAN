import sys
import os
# add the parent directory of this test file to the Python path so we can import functions from tasks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tasks'))

import pytest
import numpy as np
import torch

from functions import (
    # knot/interval conversions
    knots_to_intervals_naive,
    knots_to_intervals_torch,
    intervals_to_knots_naive,
    intervals_to_knots_torch,
    # parametrization
    uniform_params_naive,
    uniform_params_torch,
    chord_length_params_naive,
    chord_length_params_torch,
    centripetal_params_naive,
    centripetal_params_torch,
    make_grid,
    # b-spline evaluation
    bspline_basis_eval,
    bspline_basis_matrix,
    bspline_eval_naive,
    bspline_eval_torch,
    find_knot_span_naive,
    find_knot_span,
    de_boor,
    solve_control_points,
)


# ==================================================
# Shared fixtures
# ==================================================

@pytest.fixture
def uniform_knots():
    """Uniform knot vector [0, 1/3, 2/3, 1]."""
    return [0.0, 1/3, 2/3, 1.0]

@pytest.fixture
def clamped_cubic_knots():
    """Clamped cubic knot vector for 4 control points: [0,0,0,0,1,1,1,1]."""
    return [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

@pytest.fixture
def simple_controls_2d():
    """Four 2D control points forming a simple polygon."""
    return np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0]])


# ==================================================
# Knot / Interval Conversions
# ==================================================

class TestKnotsToIntervals:
    def test_naive_basic(self):
        knots = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = knots_to_intervals_naive(knots)
        assert np.allclose(result, [0.25, 0.25, 0.25, 0.25])

    def test_naive_output_length(self):
        knots = np.linspace(0.0, 1.0, 10)
        result = knots_to_intervals_naive(knots)
        assert len(result) == 9

    def test_naive_two_knots(self):
        knots = [0.0, 1.0]
        result = knots_to_intervals_naive(knots)
        assert np.isclose(result, [1.0])

    def test_naive_non_uniform(self):
        knots = [0.0, 0.1, 0.6, 1.0]
        result = knots_to_intervals_naive(knots)
        assert np.allclose(result, [0.1, 0.5, 0.4])

    def test_naive_repeated_knots(self):
        # clamped quadratic knot vector: zero-width intervals at both endpoints and interior
        knots = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
        result = knots_to_intervals_naive(knots)
        assert np.allclose(result, [0.0, 0.5, 0.0, 0.5, 0.0])
    

    def test_torch_basic(self):
        knots = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        result = knots_to_intervals_torch(knots)
        assert torch.allclose(result, torch.tensor([0.25, 0.25, 0.25, 0.25]))

    def test_torch_output_length(self):
        knots = torch.linspace(0.0, 1.0, 10)
        result = knots_to_intervals_torch(knots)
        assert result.shape[0] == 9

    def test_torch_preserves_dtype(self):
        knots = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = knots_to_intervals_torch(knots)
        assert result.dtype == torch.float64

    def test_torch_two_knots(self):
        knots = torch.tensor([0.0, 1.0])
        result = knots_to_intervals_torch(knots)
        assert torch.isclose(result, torch.tensor([1.0]))

    def test_torch_non_uniform(self):
        knots = torch.tensor([0.0, 0.1, 0.6, 1.0])
        result = knots_to_intervals_torch(knots)
        assert torch.allclose(result, torch.tensor([0.1, 0.5, 0.4]))

    def test_torch_repeated_knots(self):
        knots = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
        result = knots_to_intervals_torch(knots)
        assert torch.allclose(result, torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0]))


class TestIntervalsToKnots:
    def test_naive_basic(self):
        intervals = [0.25, 0.25, 0.25, 0.25]
        result = intervals_to_knots_naive(intervals)
        assert np.allclose(result, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_naive_output_length(self):
        intervals = [0.2, 0.2, 0.2, 0.2, 0.2]
        result = intervals_to_knots_naive(intervals)
        assert len(result) == 6

    def test_naive_one_interval(self):
        intervals = [1.0]
        result = intervals_to_knots_naive(intervals)
        assert np.all(result == [0.0, 1.0])

    def test_naive_endpoints(self):
        intervals = [0.1, 0.5, 0.4]
        result = intervals_to_knots_naive(intervals)
        assert result[0] == 0.0
        assert np.isclose(result[-1], 1.0)      # approximately 1 due to floating point
    
    def test_naive_non_decreasing(self):
        intervals = [0.1, 0.5, 0.4]
        result = intervals_to_knots_naive(intervals)
        assert np.all(np.diff(result) >= 0)

    def test_naive_zero_width_intervals(self):
        # intervals with zeros produce repeated knots
        intervals = [0.0, 0.5, 0.0, 0.5, 0.0]
        result = intervals_to_knots_naive(intervals)
        assert np.allclose(result, [0.0, 0.0, 0.5, 0.5, 1.0, 1.0])


    def test_torch_basic(self):
        intervals = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = intervals_to_knots_torch(intervals)
        assert torch.allclose(result, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))

    def test_torch_output_length(self):
        intervals = torch.ones(5) * 0.2
        result = intervals_to_knots_torch(intervals)
        assert result.shape[0] == 6
    
    def test_torch_preserves_dtype(self):
        intervals = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = intervals_to_knots_torch(intervals)
        assert result.dtype == torch.float64

    def test_torch_one_interval(self):
        intervals = torch.tensor([1.0])
        result = intervals_to_knots_torch(intervals)
        assert torch.all(result == torch.tensor([0.0, 1.0]))

    def test_torch_endpoints(self):
        intervals = torch.tensor([0.1, 0.5, 0.4])
        result = intervals_to_knots_torch(intervals)
        assert result[0] == 0.0
        assert torch.isclose(result[-1], torch.tensor(1.0))      # approximately 1 due to floating point
    
    def test_torch_non_decreasing(self):
        intervals = torch.tensor([0.1, 0.5, 0.4])
        result = intervals_to_knots_torch(intervals)
        assert torch.all(result[:-1] <= result[1:])

    def test_torch_zero_width_intervals(self):
        intervals = torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0])
        result = intervals_to_knots_torch(intervals)
        assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0]))


    def test_roundtrip_naive(self):
        knots = [0.0, 0.1, 0.3, 0.7, 1.0]
        result = intervals_to_knots_naive(knots_to_intervals_naive(knots))
        assert np.allclose(result, knots)

    def test_roundtrip_torch(self):
        knots = torch.tensor([0.0, 0.1, 0.3, 0.7, 1.0])
        result = intervals_to_knots_torch(knots_to_intervals_torch(knots))
        assert torch.allclose(result, knots)


# ==================================================
# Parametrization
# ==================================================

class TestUniformParams:
    def test_naive_basic(self):
        pts = [[0, 0], [1, 0], [2, 0]]
        params = uniform_params_naive(pts)
        assert np.allclose(params, [0.0, 0.5, 1.0])
    
    def test_naive_output_length(self):
        pts = [[i, 0] for i in range(7)]
        params = uniform_params_naive(pts)
        assert len(params) == 7

    def test_naive_two_points(self):
        pts = [[0.0, 0.0], [1.0, 1.0]]
        params = uniform_params_naive(pts)
        assert np.allclose(params, [0.0, 1.0])

    def test_naive_identical_points(self):
        pts = [[1, 1], [1, 1], [1, 1]]
        params = uniform_params_naive(pts)
        assert np.allclose(params, [0.0, 0.5, 1.0])
    
    def test_naive_endpoints(self):
        pts = [[0, 0], [2, 1], [3, 3], [2, 5], [0, 6], [-2, 5], [-3, 3], [-2, 1], [0, 0]]
        params = uniform_params_naive(pts)
        assert params[0] == 0.0 and params[-1] == 1.0
    
    def test_naive_within_bounds(self):
        pts = [[0, 0], [2, 1], [3, 3], [2, 5], [0, 6], [-2, 5], [-3, 3], [-2, 1], [0, 0]]
        params = uniform_params_naive(pts)
        assert np.all((params >= 0) & (params <= 1))

    def test_naive_non_decreasing(self):
        pts = [[0, 0], [2, 1], [3, 3], [2, 5], [0, 6], [-2, 5], [-3, 3], [-2, 1], [0, 0]]
        params = uniform_params_naive(pts)
        assert np.all(params[:-1] <= params[1:])

    def test_naive_spacing(self):
        pts = [[1, 2], [3, 4], [5, 6], [7, 8]]
        params = uniform_params_naive(pts)
        diffs = np.diff(params)
        assert np.allclose(diffs, diffs[0])   # all equal


    def test_torch_basic(self):
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        params = uniform_params_torch(pts)
        assert torch.allclose(params, torch.tensor([0.0, 0.5, 1.0]))
    
    def test_torch_output_length(self):
        pts = torch.tensor([[i, 0] for i in range(7)])
        params = uniform_params_torch(pts)
        assert params.shape[0] == 7

    def test_torch_preserves_dtype(self):
        pts = torch.zeros(4, 2, dtype=torch.float64)
        params = uniform_params_torch(pts)
        assert params.dtype == torch.float64

    def test_torch_two_points(self):
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        params = uniform_params_torch(pts)
        assert torch.allclose(params, torch.tensor([0.0, 1.0]))
    
    def test_torch_identical_points(self):
        pts = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        params = uniform_params_torch(pts)
        assert torch.allclose(params, torch.tensor([0.0, 0.5, 1.0]))
    
    def test_torch_endpoints(self):
        pts = torch.tensor([[2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0], [-2.0, 5.0], [-3.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
        params = uniform_params_torch(pts)
        assert params[0] == 0.0 and params[-1] == 1.0
    
    def test_torch_within_bounds(self):
        pts = torch.tensor([[2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0], [-2.0, 5.0], [-3.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
        params = uniform_params_torch(pts)
        assert torch.all((params >= 0) & (params <= 1))

    def test_torch_non_decreasing(self):
        pts = torch.tensor([[2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0], [-2.0, 5.0], [-3.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
        params = uniform_params_torch(pts)
        assert torch.all(params[:-1] <= params[1:])
    
    def test_torch_spacing(self):
        pts = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        params = uniform_params_torch(pts)
        diffs = params[1:] - params[:-1]
        assert torch.allclose(diffs, diffs[0])   # all equal


    def test_naive_matches_torch(self):
        pts = [[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0]]
        params_naive = uniform_params_naive(pts)
        params_torch = uniform_params_torch(
            torch.tensor(pts, dtype=torch.float64)).numpy()
        assert np.allclose(params_naive, params_torch)


class TestChordLengthParams:
    def test_naive_basic(self):
        # points at distances 1 and 2, so params should be [0, 1/3, 1]
        pts = [[0, 0], [1, 0], [3, 0]]
        params = chord_length_params_naive(pts)
        assert np.allclose(params, [0.0, 1/3, 1.0])

    def test_naive_output_length(self):
        pts = [[i, 0] for i in range(7)]
        params = chord_length_params_naive(pts)
        assert len(params) == 7

    def test_naive_two_points(self):
        pts = [[0.0, 0.0], [1.0, 1.0]]
        params = chord_length_params_naive(pts)
        assert np.allclose(params, [0.0, 1.0])

    def test_naive_identical_points(self):
        pts = [[1, 1], [1, 1], [1, 1]]
        params = chord_length_params_naive(pts)
        assert np.allclose(params, [0.0, 0.5, 1.0])

    def test_naive_endpoints(self):
        pts = [[0, 0], [2, 1], [3, 3], [2, 5], [0, 6], [-2, 5], [-3, 3], [-2, 1], [0, 0]]
        params = chord_length_params_naive(pts)
        assert params[0] == 0.0 and params[-1] == 1.0

    def test_naive_within_bounds(self):
        pts = [[0, 0], [2, 1], [3, 3], [2, 5], [0, 6], [-2, 5], [-3, 3], [-2, 1], [0, 0]]
        params = chord_length_params_naive(pts)
        assert np.all((params >= 0) & (params <= 1))
    
    def test_naive_non_decreasing(self):
        pts = [[0, 0], [2, 1], [3, 3], [2, 5], [0, 6], [-2, 5], [-3, 3], [-2, 1], [0, 0]]
        params = chord_length_params_naive(pts)
        assert np.all(params[:-1] <= params[1:])


    def test_torch_basic(self):
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        params = chord_length_params_torch(pts)
        assert torch.allclose(params, torch.tensor([0.0, 1/3, 1.0]))

    def test_torch_output_length(self):
        pts = torch.tensor([[i, 0] for i in range(7)])
        params = chord_length_params_torch(pts)
        assert params.shape[0] == 7
    
    def test_torch_preserves_dtype(self):
        pts = torch.zeros(4, 2, dtype=torch.float64)
        params = chord_length_params_torch(pts)
        assert params.dtype == torch.float64

    def test_torch_two_points(self):
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        params = chord_length_params_torch(pts)
        assert torch.allclose(params, torch.tensor([0.0, 1.0]))

    def test_torch_identical_points(self):
        pts = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        params = chord_length_params_torch(pts)
        assert torch.allclose(params, torch.tensor([0.0, 0.5, 1.0]))

    def test_torch_endpoints(self):
        pts = torch.tensor([[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0], [-2.0, 5.0], [-3.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
        params = chord_length_params_torch(pts)
        assert params[0] == 0.0 and params[-1] == 1.0

    def test_torch_within_bounds(self):
        pts = torch.tensor([[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0], [-2.0, 5.0], [-3.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
        params = chord_length_params_torch(pts)
        assert torch.all((params >= 0) & (params <= 1))

    def test_torch_non_decreasing(self):
        pts = torch.tensor([[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0], [-2.0, 5.0], [-3.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
        params = chord_length_params_torch(pts)
        assert torch.all(params[:-1] <= params[1:])


    def test_naive_matches_torch(self):
        pts = [[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0]]
        params_naive = chord_length_params_naive(pts)
        params_torch = chord_length_params_torch(
            torch.tensor(pts, dtype=torch.float64)).numpy()
        assert np.allclose(params_naive, params_torch)


class TestCentripetalParams:
    def test_naive_basic(self):
        # distances are 1 and 4; sqrt(1)=1, sqrt(4)=2, so params should be [0, 1/3, 1]
        pts = [[0, 0], [1, 0], [5, 0]]
        params = centripetal_params_naive(pts)
        assert np.allclose(params, [0.0, 1/3, 1.0])
    
    def test_naive_output_length(self):
        pts = [[i, 0] for i in range(7)]
        params = centripetal_params_naive(pts)
        assert len(params) == 7

    def test_naive_two_points(self):
        pts = [[0.0, 0.0], [1.0, 1.0]]
        params = centripetal_params_naive(pts)
        assert np.allclose(params, [0.0, 1.0])
    
    def test_naive_identical_points(self):
        pts = [[1, 1], [1, 1], [1, 1]]
        params = centripetal_params_naive(pts)
        assert np.allclose(params, [0.0, 0.5, 1.0])
    
    def test_naive_endpoints(self):
        pts = [[0, 0], [2, 1], [3, 3], [2, 5], [0, 6], [-2, 5], [-3, 3], [-2, 1], [0, 0]]
        params = centripetal_params_naive(pts)
        assert params[0] == 0.0 and params[-1] == 1.0

    def test_naive_within_bounds(self):
        pts = [[0, 0], [2, 1], [3, 3], [2, 5], [0, 6], [-2, 5], [-3, 3], [-2, 1], [0, 0]]
        params = centripetal_params_naive(pts)
        assert np.all((params >= 0) & (params <= 1))

    def test_naive_non_decreasing(self):
        pts = [[0, 0], [2, 1], [3, 3], [2, 5], [0, 6], [-2, 5], [-3, 3], [-2, 1], [0, 0]]
        params = centripetal_params_naive(pts)
        assert np.all(params[:-1] <= params[1:])
    

    def test_torch_basic(self):
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        params = centripetal_params_torch(pts)
        assert torch.allclose(params, torch.tensor([0.0, 1/3, 1.0]))

    def test_torch_output_length(self):
        pts = torch.tensor([[i, 0] for i in range(7)])
        params = centripetal_params_torch(pts)
        assert params.shape[0] == 7

    def test_torch_preserves_dtype(self):
        pts = torch.zeros(4, 2, dtype=torch.float64)
        params = centripetal_params_torch(pts)
        assert params.dtype == torch.float64

    def test_torch_two_points(self):
        pts = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        params = centripetal_params_torch(pts)
        assert torch.allclose(params, torch.tensor([0.0, 1.0]))
    
    def test_torch_identical_points(self):
        pts = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        params = centripetal_params_torch(pts)
        assert torch.allclose(params, torch.tensor([0.0, 0.5, 1.0]))

    def test_torch_endpoints(self):
        pts = torch.tensor([[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0], [-2.0, 5.0], [-3.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
        params = centripetal_params_torch(pts)
        assert params[0] == 0.0 and params[-1] == 1.0

    def test_torch_within_bounds(self):
        pts = torch.tensor([[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0], [-2.0, 5.0], [-3.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
        params = centripetal_params_torch(pts)
        assert torch.all((params >= 0) & (params <= 1))

    def test_torch_non_decreasing(self):
        pts = torch.tensor([[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0], [-2.0, 5.0], [-3.0, 3.0], [-2.0, 1.0], [0.0, 0.0]])
        params = centripetal_params_torch(pts)
        assert torch.all(params[:-1] <= params[1:])


    def test_naive_matches_torch(self):
        pts = [[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [2.0, 5.0], [0.0, 6.0]]
        params_naive = centripetal_params_naive(pts)
        params_torch = centripetal_params_torch(
            torch.tensor(pts, dtype=torch.float64)).numpy()
        assert np.allclose(params_naive, params_torch)


class TestMakeGrid:
    def test_uniform(self):
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        grid = make_grid(pts, method="uniform")
        assert torch.allclose(grid, torch.tensor([0.0, 0.5, 1.0]))

    def test_chord_length(self):
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        grid = make_grid(pts, method="chord_length")
        assert torch.allclose(grid, torch.tensor([0.0, 1/3, 1.0]))

    def test_centripetal(self):
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        grid = make_grid(pts, method="centripetal")
        assert torch.allclose(grid, torch.tensor([0.0, 1/3, 1.0]))

    def test_unknown_method_raises(self):
        pts = torch.zeros(3, 2)
        with pytest.raises(ValueError):
            make_grid(pts, method="unknown")


# ==================================================
# B-spline Evaluation
# ==================================================

class TestBsplineBasisEval:
    def test_degree0_inside_interval(self):
        knots = [0.0, 0.5, 1.0]
        # B_0^0 has support [0, 0.5), B_1^0 has support [0.5, 1.0]
        assert bspline_basis_eval(0, 0, 0.25, knots) == 1.0
        assert bspline_basis_eval(1, 0, 0.25, knots) == 0.0

    def test_degree0_at_right_endpoint_of_last_interval(self):
        knots = [0.0, 0.5, 1.0]
        assert bspline_basis_eval(1, 0, 1.0, knots) == 1.0

    def test_degree0_non_negativity(self):
        knots = [0.0, 0.25, 0.5, 0.75, 1.0]
        t_grid = np.linspace(0.0, 1.0, 20)
        for t in t_grid:
            for j in range(4):
                val = bspline_basis_eval(j, 0, t, knots)
                assert val >= 0.0, f"Non-negativity failed at t={t}, j={j}: value={val}"

    def test_degree0_partition_of_unity(self):
        knots = [0.0, 0.25, 0.5, 0.75, 1.0]
        t_grid = np.linspace(0.0, 1.0, 20)
        for t in t_grid:
            total = sum(bspline_basis_eval(j, 0, t, knots) for j in range(4))
            assert np.isclose(total, 1.0), f"Partition of unity failed at t={t}"

    def test_degree0_local_support(self):
        knots = [0.0, 0.25, 0.5, 0.75, 1.0]
        # At t=0.1, only B_0^0 should be nonzero
        assert bspline_basis_eval(0, 0, 0.1, knots) == 1.0
        assert bspline_basis_eval(1, 0, 0.1, knots) == 0.0
        assert bspline_basis_eval(2, 0, 0.1, knots) == 0.0
        assert bspline_basis_eval(3, 0, 0.1, knots) == 0.0
        # At t=0.3, only B_1^0 should be nonzero
        assert bspline_basis_eval(0, 0, 0.3, knots) == 0.0
        assert bspline_basis_eval(1, 0, 0.3, knots) == 1.0
        assert bspline_basis_eval(2, 0, 0.3, knots) == 0.0
        assert bspline_basis_eval(3, 0, 0.3, knots) == 0.0
        # At t=0.6, only B_2^0 should be nonzero
        assert bspline_basis_eval(0, 0, 0.6, knots) == 0.0
        assert bspline_basis_eval(1, 0, 0.6, knots) == 0.0
        assert bspline_basis_eval(2, 0, 0.6, knots) == 1.0
        assert bspline_basis_eval(3, 0, 0.6, knots) == 0.0
        # At t=0.8, only B_3^0 should be nonzero
        assert bspline_basis_eval(0, 0, 0.8, knots) == 0.0
        assert bspline_basis_eval(1, 0, 0.8, knots) == 0.0
        assert bspline_basis_eval(2, 0, 0.8, knots) == 0.0
        assert bspline_basis_eval(3, 0, 0.8, knots) == 1.0
    
    
    def test_degree1_non_negativity(self):
        knots = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
        t_grid = np.linspace(0.0, 1.0, 20)
        for t in t_grid:
            for j in range(5):
                val = bspline_basis_eval(j, 1, t, knots)
                assert val >= 0.0, f"Non-negativity failed at t={t}, j={j}: value={val}"

    def test_degree1_partition_of_unity(self):
        knots = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
        for t in [0.0, 0.3, 0.6, 1.0]:
            total = sum(bspline_basis_eval(j, 1, t, knots) for j in range(5))
            assert np.isclose(total, 1.0), f"Partition of unity failed at t={t}"

    def test_degree1_local_support(self):
        knots = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
        # At t=0.1, only B_0^1 and B_1^1 should be nonzero
        assert bspline_basis_eval(0, 1, 0.1, knots) > 0.0
        assert bspline_basis_eval(1, 1, 0.1, knots) > 0.0
        assert bspline_basis_eval(2, 1, 0.1, knots) == 0.0
        assert bspline_basis_eval(3, 1, 0.1, knots) == 0.0
        assert bspline_basis_eval(4, 1, 0.1, knots) == 0.0
        # At t=0.3, only B_1^1 and B_2^1 should be nonzero
        assert bspline_basis_eval(0, 1, 0.3, knots) == 0.0
        assert bspline_basis_eval(1, 1, 0.3, knots) > 0.0
        assert bspline_basis_eval(2, 1, 0.3, knots) > 0.0
        assert bspline_basis_eval(3, 1, 0.3, knots) == 0.0
        assert bspline_basis_eval(4, 1, 0.3, knots) == 0.0
        # At t=0.6, only B_2^1 and B_3^1 should be nonzero
        assert bspline_basis_eval(0, 1, 0.6, knots) == 0.0
        assert bspline_basis_eval(1, 1, 0.6, knots) == 0.0
        assert bspline_basis_eval(2, 1, 0.6, knots) > 0.0
        assert bspline_basis_eval(3, 1, 0.6, knots) > 0.0
        assert bspline_basis_eval(4, 1, 0.6, knots) == 0.0
        # At t=0.8, only B_3^1 and B_4^1 should be nonzero
        assert bspline_basis_eval(0, 1, 0.8, knots) == 0.0
        assert bspline_basis_eval(1, 1, 0.8, knots) == 0.0
        assert bspline_basis_eval(2, 1, 0.8, knots) == 0.0
        assert bspline_basis_eval(3, 1, 0.8, knots) > 0.0
        assert bspline_basis_eval(4, 1, 0.8, knots) > 0.0
    

    def test_degree2_non_negativity(self):
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        t_grid = np.linspace(0.0, 1.0, 20)
        for t in t_grid:
            for j in range(4):
                val = bspline_basis_eval(j, 2, t, knots)
                assert val >= 0.0, f"Non-negativity failed at t={t}, j={j}: value={val}"

    def test_degree2_partition_of_unity(self):
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            total = sum(bspline_basis_eval(j, 2, t, knots) for j in range(4))
            assert np.isclose(total, 1.0), f"Partition of unity failed at t={t}"
    

    def test_degree3_non_negativity(self):
        knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        t_grid = np.linspace(0.0, 1.0, 20)
        for t in t_grid:
            for j in range(4):
                val = bspline_basis_eval(j, 3, t, knots)
                assert val >= 0.0, f"Non-negativity failed at t={t}, j={j}: value={val}"
    
    def test_degree3_partition_of_unity(self):
        knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            total = sum(bspline_basis_eval(j, 3, t, knots) for j in range(4))
            assert np.isclose(total, 1.0), f"Partition of unity failed at t={t}"


    def test_repeated_interior_knots_partition_of_unity(self):
        # double interior knot reduces continuity but partition of unity must still hold
        knots = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            total = sum(bspline_basis_eval(j, 2, t, knots) for j in range(5))
            assert np.isclose(total, 1.0), f"Partition of unity failed at t={t}"


class TestBsplineBasisMatrix:

    # test both soft=False and soft=True cases using the same tests, since the properties should hold for both (except local support which is tested separately)
    @pytest.fixture(params=[False, True], ids=["hard", "soft"])
    def soft(self, request):
        return request.param


    def test_shape(self, soft):
        knots = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        t_grid = torch.linspace(0.0, 1.0, 10)
        B = bspline_basis_matrix(t_grid, knots, degree=3, soft=soft)
        assert B.shape == (10, 4)   # K - degree - 1 = 8 - 3 - 1 = 4

    def test_non_negativity(self, soft):
        knots = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        t_grid = torch.linspace(0.0, 1.0, 20)
        B = bspline_basis_matrix(t_grid, knots, degree=2, soft=soft)
        assert torch.all(B >= 0)
    
    def test_partition_of_unity(self, soft):
        knots = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        t_grid = torch.linspace(0.0, 1.0, 20)
        B = bspline_basis_matrix(t_grid, knots, degree=2, soft=soft)
        row_sums = B.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(20), atol=1e-5 if soft else 1e-8)  # allow slightly larger tolerance for soft=True due to small nonzero values outside support

    def test_local_support(self):
        # in knot span [u_i, u_{i+1}), only basis functions B_{i-degree}, ..., B_i can be nonzero]
        knots = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        t_grid = torch.tensor([-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1])
        B = bspline_basis_matrix(t_grid, knots, degree=2, soft=False)
        # At t=-0.1 and t=1.1 (outside the knot span), all basis functions should be zero
        assert (B[0, :] == 0).all()
        assert (B[-1, :] == 0).all()
        # At t=0.0 and t=0.25, only the first 3 basis functions B_0, B_1, B_2 should be nonzero
        assert B[1, -1] == 0
        assert B[2, -1] == 0
        # At t=0.5 and t=0.75, only the last 3 basis functions B_1, B_2, B_3 should be nonzero
        assert B[3, 0] == 0
        assert B[4, 0] == 0
        # At t=1.0, only the last 3 basis functions B_1, B_2, B_3 should be nonzero
        assert B[5, 0] == 0

    def test_soft_local_support(self):
        # with soft=True, all basis functions should be non-negative but not necessarily zero outside their support
        knots = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        t_grid = torch.tensor([-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1])
        B = bspline_basis_matrix(t_grid, knots, degree=2, soft=True)
        # At t=-0.1 and t=1.1 (outside the knot span), all basis functions should be close to zero
        assert (B[0, :] < 1e-5).all()
        assert (B[-1, :] < 1e-5).all()
        # At t=0.0 and t=0.25, only the first 3 basis functions B_0, B_1, B_2 should be significantly larger than the last one
        assert B[1, -1] < 1e-5
        assert B[2, -1] < 1e-5
        # At t=0.5 and t=0.75, only the last 3 basis functions B_1, B_2, B_3 should be significantly larger than the first one
        assert B[3, 0] < 1e-5
        assert B[4, 0] < 1e-5
        # At t=1.0, only the last 3 basis functions B_1, B_2, B_3 should be significantly larger than the first one
        assert B[5, 0] < 1e-5
    
    def test_matches_bspline_basis_eval(self):
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        t_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        B = bspline_basis_matrix(
            torch.tensor(t_grid, dtype=torch.float64), 
            torch.tensor(knots, dtype=torch.float64), degree=2)
        for i, t in enumerate(t_grid):
            for j in range(4):
                expected = bspline_basis_eval(j, 2, t, knots)
                assert np.isclose(B[i, j].item(), expected, atol=1e-6), \
                    f"Mismatch at t={t}, j={j}: matrix={B[i,j].item():.6f}, scalar={expected:.6f}"


    def test_degree0_shape(self):
        # degree 0: K - 0 - 1 = K - 1 columns, one per interval
        knots = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        t_grid = torch.linspace(0.0, 1.0, 10)
        B = bspline_basis_matrix(t_grid, knots, degree=0)
        assert B.shape == (10, 4)

    def test_degree0_partition_of_unity(self):
        knots = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        t_grid = torch.linspace(0.0, 1.0, 20)
        B = bspline_basis_matrix(t_grid, knots, degree=0)
        assert torch.allclose(B.sum(dim=1), torch.ones(20), atol=1e-8)

    def test_degree1_shape(self):
        # degree 1: K - 1 - 1 = K - 2 columns
        knots = torch.tensor([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
        t_grid = torch.linspace(0.0, 1.0, 10)
        B = bspline_basis_matrix(t_grid, knots, degree=1)
        assert B.shape == (10, 5)   # 7 - 1 - 1 = 5

    def test_degree1_partition_of_unity(self):
        knots = torch.tensor([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
        t_grid = torch.linspace(0.0, 1.0, 20)
        B = bspline_basis_matrix(t_grid, knots, degree=1)
        assert torch.allclose(B.sum(dim=1), torch.ones(20), atol=1e-8)

    def test_interior_repeated_knots_partition_of_unity(self):
        # double interior knot reduces continuity but partition of unity must still hold
        knots = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0])
        t_grid = torch.linspace(0.0, 1.0, 20)
        B = bspline_basis_matrix(t_grid, knots, degree=2)
        assert torch.allclose(B.sum(dim=1), torch.ones(20), atol=1e-8)


class TestBsplineEval:
    """Tests comparing bspline_eval_torch, bspline_eval_naive and de_boor for consistency."""

    @pytest.fixture
    def cubic_setup(self):
        knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        controls = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0]])
        return knots, controls


    def test_naive_degree1(self):
        # clamped linear b-spline with 3 control points
        knots = [0.0, 0.0, 0.5, 1.0, 1.0]
        controls = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        pt = bspline_eval_naive(0.25, knots, 1, controls)
        assert np.allclose(pt, [0.5, 0.5])

    def test_naive_at_endpoints(self, cubic_setup):
        knots, controls = cubic_setup
        pt0 = bspline_eval_naive(0.0, knots, 3, controls)
        pt1 = bspline_eval_naive(1.0, knots, 3, controls)
        assert np.allclose(pt0, controls[0])    # clamped: curve starts at first control point
        assert np.allclose(pt1, controls[-1])   # clamped: curve ends at last control point


    def test_de_boor_degree1(self):
        knots = [0.0, 0.0, 0.5, 1.0, 1.0]
        controls = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        pt = de_boor(0.25, knots, 1, controls)
        assert np.allclose(pt, [0.5, 0.5])

    def test_de_boor_at_endpoints(self, cubic_setup):
        knots, controls = cubic_setup
        pt0 = de_boor(0.0, knots, 3, controls)
        pt1 = de_boor(1.0, knots, 3, controls)
        assert np.allclose(pt0, controls[0])
        assert np.allclose(pt1, controls[-1])


    def test_naive_matches_de_boor(self, cubic_setup):
        knots, controls = cubic_setup
        for t in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            pt_naive = bspline_eval_naive(t, knots, 3, controls)
            pt_deboor = de_boor(t, knots, 3, controls)
            assert np.allclose(pt_naive, pt_deboor, atol=1e-6), \
                f"Mismatch at t={t}: naive={pt_naive}, de_boor={pt_deboor}"

    def test_nonclamped_naive_matches_de_boor(self):
        # non-clamped cubic: curve does not interpolate endpoints
        knots = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        controls = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 0.0]])
        # valid domain is [knots[3], knots[5]) = [0.75, 1.25); exclude right endpoint 1.25
        # because find_knot_span maps it to span 5 (outside the last valid span 4)
        for t in [0.75, 0.875, 1.0, 1.125]:
            pt_naive = bspline_eval_naive(t, knots, 3, controls)
            pt_deboor = de_boor(t, knots, 3, controls)
            assert np.allclose(pt_naive, pt_deboor, atol=1e-6), \
                f"Mismatch at t={t}: naive={pt_naive}, de_boor={pt_deboor}"

    def test_torch_eval_matches_naive(self, cubic_setup):
        knots, controls = cubic_setup
        t_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        curve_torch = bspline_eval_torch(
            torch.tensor(t_grid, dtype=torch.float64), 
            torch.tensor(knots, dtype=torch.float64), 3, 
            torch.tensor(controls, dtype=torch.float64)).numpy()
        for i, t in enumerate(t_grid):
            expected = bspline_eval_naive(t, knots, 3, controls)
            assert np.allclose(curve_torch[i], expected, atol=1e-6), \
                f"Torch vs naive mismatch at t={t}"


class TestFindKnotSpan:
    def test_naive_basic(self):
        knots = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert find_knot_span_naive(0.1, knots) == 0
        assert find_knot_span_naive(0.3, knots) == 1
        assert find_knot_span_naive(0.6, knots) == 2

    def test_naive_at_last_knot(self):
        knots = [0.0, 0.5, 1.0]
        assert find_knot_span_naive(1.0, knots) == 1

    def test_naive_at_first_knot(self):
        # t == knots[0] with clamped knots: should return the first non-degenerate span
        knots = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
        assert find_knot_span_naive(0.0, knots) == 2  # span [knots[2], knots[3]) = [0.0, 0.25)

    def test_naive_out_of_bounds_raises(self):
        knots = [0.0, 0.25, 0.5, 0.75, 1.0]
        with pytest.raises(ValueError):
            find_knot_span_naive(-0.1, knots)
        with pytest.raises(ValueError):
            find_knot_span_naive(1.1, knots)


    def test_clamped_at_last_knot(self):
        knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        assert find_knot_span(1.0, knots) == 3

    def test_both_at_first_knot(self):
        # binary search and naive should agree at t == knots[0]
        knots = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
        assert find_knot_span(0.0, knots) == find_knot_span_naive(0.0, knots)

    def test_binary_search_matches_naive(self):
        knots = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
        test_vals = [0.0, 0.1, 0.25, 0.4, 0.5, 0.7, 1.0]
        for t in test_vals:
            assert find_knot_span(t, knots) == find_knot_span_naive(t, knots), \
                f"Mismatch at t={t}"

    def test_repeated_interior_knots(self):
        # knot vector with a double interior knot at 0.5
        knots = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
        for t in [0.49, 0.5, 0.51]:
            assert find_knot_span(t, knots) == find_knot_span_naive(t, knots), \
                f"Mismatch at t={t}"


class TestSolveControlPoints:
    def test_exact_fit_identity_basis(self):
        # if B is identity, control points should equal target
        B = torch.eye(4)
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        C = solve_control_points(B, X, reg=1e-10)
        assert torch.allclose(C, X, atol=1e-4)

    def test_output_shape(self):
        B = torch.ones(10, 4) / 4.0
        X = torch.rand(10, 2)
        C = solve_control_points(B, X)
        assert C.shape == (4, 2)

    def test_reconstruction_quality(self):
        # fit a degree-3 clamped b-spline to points exactly on the curve -> low error
        knots = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        t_grid = torch.linspace(0.0, 1.0, 20)
        B = bspline_basis_matrix(t_grid, knots, degree=3)
        true_C = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0]])
        X = B @ true_C   # points exactly on the curve
        C_fit = solve_control_points(B, X, reg=1e-8)
        assert torch.allclose(C_fit, true_C, atol=1e-3)

    def test_near_singular_regularized(self):
        # poorly conditioned B (all evaluation points clustered near t=0); regularization
        # must prevent a singular solve and return finite control points
        knots = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        t_grid = torch.cat([torch.linspace(0.0, 0.01, 18), torch.linspace(0.99, 1.0, 2)])
        B = bspline_basis_matrix(t_grid, knots, degree=3)
        X = torch.rand(20, 2)
        C = solve_control_points(B, X, reg=1e-2)
        assert C.shape == (4, 2)
        assert torch.all(torch.isfinite(C))
