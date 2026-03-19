"""
Functions implementation.

This module contains implementations of various functions used in B-spline curve fitting tasks, including:
- Knot/Interval conversions: functions to convert between knot vectors and interval vectors.
- Parametrization methods: functions to generate parameter values for given control points
     using uniform, chord length, or centripetal methods.
- B-spline evaluation: functions to evaluate B-spline curves given basis matrices and control points,
     or using the de Boor algorithm for specific parameter values.

NOTE: torch implementations keep dtype/device and are differentiable by autograd
"""


import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import os


# ==================================================


# --- Knot/Interval Conversions ---


def knots_to_intervals_naive(knots):
    """
    Convert knot vector to interval vector.

    Subtract adjacent knot values to get intervals between knots:
        intervals[i] = knots[i+1] - knots[i] for i in [0, m-1]

    Parameters
    ----------
    knots : list of float | np.ndarray (m+1,) float
        A vector of knot values (non-decreasing sequence).
    
    Returns
    -------
    intervals : list of float
        A vector of intervals between knots (positive values).

    Notes
    -----
    - The input `knots` should have at least 2 elements to compute intervals.
    - The output `intervals` will have one less element than `knots`.

    """
    intervals = []
    for i in range(len(knots) - 1):
        intervals.append(knots[i + 1] - knots[i])
    return intervals

def knots_to_intervals_torch(knots: torch.Tensor) -> torch.Tensor:
    """
    Convert knot vector to interval vector (vectorized).

    Subtract adjacent knot values to get intervals between knots:
        intervals[i] = knots[i+1] - knots[i] for i in [0, m-1]

    Parameters
    ----------
    knots : torch.Tensor (m+1,) float
        A vector of knot values (non-decreasing sequence).
    
    Returns
    -------
    intervals : torch.Tensor (m,) float
        A vector of intervals between knots (positive values).

    Notes
    -----
    - The input `knots` should have at least 2 elements to compute intervals.
    - The output `intervals` will have one less element than `knots`.

    """
    return knots[1:] - knots[:-1]   # all but the first minus all but the last

knots_to_intervals = knots_to_intervals_torch

# --------------------------------------------------

def intervals_to_knots_naive(intervals):
    """
    Convert interval vector back to knot vector.

    Prepend a 0 and do a cumulative sum:
        knots[0] = 0
        knots[i+1] = knots[i] + intervals[i] for i in [0, m-1]

    Parameters
    ----------
    intervals : list of float | np.ndarray (m,) float
        A vector of intervals between knots (positive values).
    
    Returns
    -------
    knots : list of float
        A vector of knot values (non-decreasing sequence).

    Notes
    -----
    - The input `intervals` should have at least 1 element to compute knots.
    - The output `knots` will have one more element than `intervals`.
    - The first knot value is 0.
    - The final knot value may not be exact due to floating point precision.
    
    """
    knots = [0]
    for interval in intervals:
        knots.append(knots[-1] + interval)
    return knots

def intervals_to_knots_torch(intervals: torch.Tensor) -> torch.Tensor:
    """
    Convert interval vector back to knot vector (vectorized).

    Prepend a 0 and do a cumulative sum:
        knots[0] = 0
        knots[i+1] = knots[i] + intervals[i] for i in [0, m-1]

    Parameters
    ----------
    intervals : torch.Tensor (m,) float
        A vector of intervals between knots (positive values).
    
    Returns
    -------
    knots : torch.Tensor (m+1,) float
        A vector of knot values (non-decreasing sequence).

    Notes
    -----
    - The input `intervals` should have at least 1 element to compute knots.
    - The output `knots` will have one more element than `intervals`.
    - The first knot value is 0.
    - The final knot value may not be exact due to floating point precision.
    
    """
    zero = torch.zeros(1, dtype=intervals.dtype, device=intervals.device)
    return torch.cat((zero, torch.cumsum(intervals, dim=0)))

intervals_to_knots = intervals_to_knots_torch

# ==================================================


# --- Parametrization ---


def uniform_params_naive(points):
    """
    Generate parametrization vector according to uniform method.

    Simply divide the interval [0, 1] into equal parts:
        params[i] = i / (N - 1) for i in [0, N-1]

    Parameters
    ----------
    points : list of list of float | np.ndarray (N, dim) float
        An ordered list of data points. Each point is an arbitrary dimensional vector.
    
    Returns
    -------
    params : np.ndarray (N,) float
        A vector of parameter values corresponding to each data point.
    
    Notes
    -----
    - The input `points` should have at least 2 points to compute parameters.
    - The output `params` will have the same number of elements as `points`.
    
    """
    return np.linspace(0.0, 1.0, len(points))

def uniform_params_torch(points: torch.Tensor) -> torch.Tensor:
    """
    Generate parametrization vector according to uniform method (Pytorch version).

    Simply divide the interval [0, 1] into equal parts:
        params[i] = i / (N - 1) for i in [0, N-1]

    Parameters
    ----------
    points : torch.Tensor (N, dim) float
        An ordered list of data points. Each point is an arbitrary dimensional vector.
    
    Returns
    -------
    params : torch.Tensor (N,) float
        A vector of parameter values corresponding to each data point.
    
    Notes
    -----
    - The input `points` should have at least 2 points to compute parameters.
    - The output `params` will have the same number of elements as `points`.
    
    """
    # create a linearly spaced vector of length N (number of points) from 0 to 1
    return torch.linspace(0., 1., len(points), 
                          dtype=points.dtype, device=points.device)

uniform_params = uniform_params_torch

# --------------------------------------------------

def chord_length_params_naive(points):
    """
    Generate parametrization vector according to chord length method.

    Compute distances between consecutive points, then normalize cumulative distances to [0, 1]:
        d[j] = ||points[j+1] - points[j]|| for j in [0, N-2]
        params[0] = 0
        params[i] = sum_(j=0)^i d[j] / sum_(j=0)^(N-2) d[j] for i in [1, N-1]

    Parameters
    ----------
    points : list of list of float | np.ndarray (N, dim) float
        An ordered list of data points. Each point is an arbitrary dimensional vector.
    
    Returns
    -------
    params : np.ndarray (N,) float
        A vector of parameter values corresponding to each data point.
    
    Notes
    -----
    - The input `points` should have at least 2 points to compute parameters.
    - The output `params` will have the same number of elements as `points`.
    - If all points are identical, the output will be the same as uniform parameters.

    """
    pts = np.asarray(points)                            # convert to numpy array if not already
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)    # ||p[j+1] - p[j]|| for j in [0, N-2]
    L = d.sum()                                         # total length of the curve (sum of distances)
    if L == 0:
        return uniform_params_naive(points)
    cum = np.concatenate(([0.0], np.cumsum(d)))         # prepend 0 and compute cumulative sum of distances
    return cum / L                                      # normalize to [0, 1]

def chord_length_params_torch(points: torch.Tensor) -> torch.Tensor:
    """
    Generate parametrization vector according to chord length method (Pytorch version).

    Compute distances between consecutive points, then normalize cumulative distances to [0, 1]:
        d[j] = ||points[j+1] - points[j]|| for j in [0, N-2]
        params[0] = 0
        params[i] = sum_(j=0)^i d[j] / sum_(j=0)^(N-2) d[j] for i in [1, N-1]

    Parameters
    ----------
    points : torch.Tensor (N, dim) float
        An ordered list of data points. Each point is an arbitrary dimensional vector.
    
    Returns
    -------
    params : torch.Tensor (N,) float
        A vector of parameter values corresponding to each data point.
    
    Notes
    -----
    - The input `points` should have at least 2 points to compute parameters.
    - The output `params` will have the same number of elements as `points`.
    - If all points are identical, the output will be the same as uniform parameters.

    """
    distances = torch.sqrt(torch.sum((points[1:] - points[:-1]) ** 2, dim=1))   # compute distances between consecutive points
    cumulative_distances = torch.cumsum(distances, dim=0)                       # cumulative sum of distances to get the parameter values before normalization
    total_distance = cumulative_distances[-1]                                   # total length of the curve (last value of cumulative distances)
    if total_distance == 0:
        return uniform_params_torch(points)
    zero = torch.zeros(1, dtype=points.dtype, device=points.device)             # create a zero tensor to prepend to the cumulative distances
    return torch.cat((zero, cumulative_distances / total_distance))             # prepend 0 and normalize to [0, 1]

chord_length_params = chord_length_params_torch

# --------------------------------------------------

def centripetal_params_naive(points):
    """
    Generate parametrization vector according to centripetal method.

    Compute distances between consecutive points, take square root, then normalize cumulative values to [0, 1]:
        rd[j] = sqrt(||points[j+1] - points[j]||) for j in [0, N-2]
        params[0] = 0
        params[i] = sum_(j=0)^i rd[j] / sum_(j=0)^(N-2) rd[j] for i in [1, N-1]

    Parameters
    ----------
    points : list of list of float | np.ndarray (N, dim) float
        An ordered list of data points. Each point is an arbitrary dimensional vector.
    
    Returns
    -------
    params : np.ndarray (N,) float
        A vector of parameter values corresponding to each data point.
    
    Notes
    -----
    - The input `points` should have at least 2 points to compute parameters.
    - The output `params` will have the same number of elements as `points`.
    - If all points are identical, the output will be the same as uniform parameters.

    """
    pts = np.asarray(points)
    rd = np.sqrt(np.linalg.norm(np.diff(pts, axis=0), axis=1))  # same as chord length but with sqrt of distances
    S = rd.sum()
    if S == 0:
        return uniform_params_naive(points)
    cum = np.concatenate(([0.0], np.cumsum(rd)))
    return cum / S

def centripetal_params_torch(points: torch.Tensor) -> torch.Tensor:
    """
    Generate parametrization vector according to centripetal method (Pytorch version).

    Compute distances between consecutive points, take square root, then normalize cumulative values to [0, 1]:
        rd[j] = sqrt(||points[j+1] - points[j]||) for j in [0, N-2]
        params[0] = 0
        params[i] = sum_(j=0)^i rd[j] / sum_(j=0)^(N-2) rd[j] for i in [1, N-1]

    Parameters
    ----------
    points : torch.Tensor (N, dim) float
        An ordered list of data points. Each point is an arbitrary dimensional vector.
    
    Returns
    -------
    params : torch.Tensor (N,) float
        A vector of parameter values corresponding to each data point.
    
    Notes
    -----
    - The input `points` should have at least 2 points to compute parameters.
    - The output `params` will have the same number of elements as `points`.
    - If all points are identical, the output will be the same as uniform parameters.

    """
    distances = torch.sqrt(torch.sum((points[1:] - points[:-1]) ** 2, dim=1))
    cumulative_distances = torch.cumsum(torch.sqrt(distances), dim=0)   # same as chord length but with sqrt of distances
    total_distance = cumulative_distances[-1]
    if total_distance == 0:
        return uniform_params_torch(points)
    zero = torch.zeros(1, dtype=points.dtype, device=points.device)
    return torch.cat((zero, cumulative_distances / total_distance))

centripetal_params = centripetal_params_torch

# --------------------------------------------------

def make_grid(points, method="uniform"):
    """
    Generate a grid of parameter values for a set of points using the specified method.

    Parameters
    ----------
    points : torch.Tensor (N, dim) float
        An ordered list of data points. Each point is an arbitrary dimensional vector.
    method : str, optional
        The method to use for parametrization. 
        Options are "uniform", "chord_length", and "centripetal".
        Default is "uniform".
    
    Returns
    -------
    params : torch.Tensor (N,) float
        A vector of parameter values corresponding to each data point,
         generated according to the specified method.
    
    Notes
    -----
    - The input `points` should have at least 2 points to compute parameters.
    - The output `params` will have the same number of elements as `points`.
    - The generated parameters will be in the range [0, 1].
        The first parameter will be 0 and the last parameter will be 1, with intermediate values determined by the chosen method.
    - If the specified method is not recognized, a ValueError will be raised.
    
    """
    if   method == "uniform":
        return uniform_params(points)
    elif method == "chord_length":
        return chord_length_params(points)
    elif method == "centripetal":
        return centripetal_params(points)
    else:
        raise ValueError(f"Unknown parametrization method: {method}")


# ==================================================


# --- B-spline evaluation ---


def bspline_basis_matrix(t_grid, knots, degree, soft=False, k=1000.0):
    """
    Assemble matrix of B-spline basis functions evaluated at the parameter grid for the given knots and degree.

    Uses the Cox--de Boor recursion formula to compute the basis functions in a vectorized manner,
     starting from degree 0 (piecewise constant) and building up to the desired degree.
    The `soft` option allows for a differentiable approximation of the degree-0 basis functions using sigmoids,
     which can be useful for training with autograd.

    Parameters
    ----------
    t_grid : torch.Tensor (N,) float
        A vector of parameter values where to evaluate the B-spline basis functions.
    knots : torch.Tensor (K,) float
        A vector of knot values (non-decreasing sequence).
    degree : int >= 0
        The degree of the B-spline basis functions.
    soft : bool, optional
        If False, use hard indicator for degree 0 (exact, non-differentiable).
        If True, use soft indicator using `sigmoid(k * (x - left)) * sigmoid(k * (right - x))`.
        Default is False.
    k : float, optional
        Steepness for sigmoid when soft=True (larger k -> closer to hard step).
        Default is 1000.0.

    Returns
    -------
    B : torch.Tensor (N, M) float
        A matrix of B-spline basis functions evaluated at the parameter grid, where M = K - degree - 1.
    
    Notes
    -----
    - The inputs `t_grid` and `knots` should have same dtype and device.
    - The input `t_grid` should have values within the range covered by `knots` for meaningful basis function values.
    - The input `knots` should have at least degree + 2 elements to define the basis functions.

    """
    N = t_grid.shape[0]     # number of evaluation points
    K = knots.shape[0]      # number of knots
    m = K - 1               # number of intervals between knots, which is also the number of degree-0 basis functions
    if m <= 0:
        return torch.zeros((N, 0), dtype=t_grid.dtype, device=t_grid.device)

    # lefts: knots[0..K-2], rights: knots[1..K-1]
    lefts = knots[:-1]          # shape (m,)
    rights = knots[1:]          # shape (m,)
    # expand t_grid to (N,1) for broadcasting against (m,) intervals
    t_expand = t_grid.unsqueeze(1)   # (N,1)

    # degree-0 basis (vectorized)
    # compute indicator functions for each interval [lefts[j], rights[j]) for j=0..m-1

    # hard mask: intervals [left, right) except last interval which includes right endpoint
    mask = (t_expand >= lefts) & (t_expand < rights)    # (N,m) boolean
    
    # find the last non-degenerate interval (where lefts[j] < rights[j])
    last_nondegen_idx = -1
    for jj in range(len(lefts) - 1, -1, -1):    # loop from the end
        if lefts[jj] < rights[jj]:
            last_nondegen_idx = jj
            break
    
    # include right endpoint for the last non-degenerate interval
    if last_nondegen_idx >= 0:
        mask_last = t_grid == rights[last_nondegen_idx] # (N,) boolean
        mask = mask.clone()                             # create a copy to avoid in-place modification which may cause issues with autograd
        mask[:, last_nondegen_idx] = mask[:, last_nondegen_idx] | mask_last     # logical OR between mask (<) and mask_last (==) for the last interval
    
    hard = mask.to(dtype=knots.dtype)      # convert boolean to float

    if not soft:
        Bprev = hard
    else:
        # soft mask: use product of two sigmoids -> smooth approximation of indicator
        # torch.sigmoid : [-inf,+inf] -> [0,1],  0 |-> 1/2,  large positive -> ~1,  large negative -> ~0
        left_term  = torch.sigmoid(k * (t_expand - lefts))      # (N,m)
        right_term = torch.sigmoid(k * (rights - t_expand))     # (N,m)
        soft_vals = (left_term * right_term)
        
        # zero out zero-width intervals — they should contribute nothing,
        # but sigmoid(0)*sigmoid(0)=0.25 would otherwise corrupt the recursion
        non_degenerate = (rights > lefts).unsqueeze(0)          # (1, m) bool
        soft_vals = soft_vals * non_degenerate
        hard      = hard * non_degenerate

        # Straight-Through Estimator (STE):
        #   in the forward pass this equals hard exactly;
        #   in the backward pass autograd sees only soft_vals (hard does not depend on knots, so it has zero gradients)
        Bprev = hard + (soft_vals - soft_vals.detach())

    # recursion for p = 1..degree
    # at recursion p we should produce shape (N, K-p-1)
    for p in range(1, degree + 1):
        M = K - p - 1       # number of basis functions at degree p
        if M <= 0:
            return torch.zeros((N, 0), dtype=t_grid.dtype, device=t_grid.device)
        
        # slices for numerators (with broadcasting):
        # num1_j = t_grid - knots[j]      for j=0..M-1  =>  t_grid[:,None] - knots[0:M]
        # num2_j = knots[j+p+1] - t_grid  for j=0..M-1  =>  knots[p+1 : M+p+1] - t_grid[:,None]
        num1 = t_expand - knots[0 : M]                          # (N, M)
        num2 = knots[p + 1 : p + 1 + M] - t_expand              # (N, M)

        # slices for denominators:
        # denom1_j = knots[j+p]   - knots[j]    for j=0..M-1  =>  knots[p : M+p] - knots[0:M]
        # denom2_j = knots[j+p+1] - knots[j+1]  for j=0..M-1  =>  knots[p+1 : M+p+1] - knots[1:M+1]
        denom1 = knots[p : p + M] - knots[0 : M]                # (M,)
        denom2 = knots[p + 1 : p + 1 + M] - knots[1 : M + 1]    # (M,)

        # expand denom1/2 for broadcasting
        denom1_expand = denom1.unsqueeze(0).expand_as(num1)     # (N, M)
        denom2_expand = denom2.unsqueeze(0).expand_as(num2)     # (N, M)

        # convention: coefficients are zero if the corresponding denominator is zero, which effectively removes that term from the sum
        # safe division: compute coef = num / denom where denom != 0, else 0
        # use masking to avoid dividing by zero (safe for gradients)
        mask1 = denom1_expand != 0
        coef1 = torch.zeros_like(num1)
        coef1[mask1] = num1[mask1] / denom1_expand[mask1]
        
        mask2 = denom2_expand != 0
        coef2 = torch.zeros_like(num2)
        coef2[mask2] = num2[mask2] / denom2_expand[mask2]

        # slices for basis matrix:
        # B_j^{p} = coef1_j * B_j^{p-1} + coef2_j * B_{j+1}^{p-1}  for j=0..M-1
        # Bprev has shape (N, previous_M = M+1)
        left_term  = coef1 * Bprev[:, 0 : M    ]    # (N, M)
        right_term = coef2 * Bprev[:, 1 : M + 1]    # (N, M)
        Bp = left_term + right_term                 # (N, M)
        Bprev = Bp                                  # allocate new tensor each degree
    
    return Bprev   # shape (N, K-degree-1)


def solve_control_points(B, X, reg=1e-6):
    """
    Solve for control points given a B-spline basis matrix and target points using least squares.

    Solve min_C ||B @ C - X||^2 + reg * ||C||^2
     equivalent to the normal equation (B^T B + reg*I) C = B^T X.
    Uses Tikhonov regularization to ensure numerical stability (if ill-conditioned)
     and differentiability (if singular).
    
    Parameters
    ----------
    B : torch.Tensor (N, M) float
        B-spline basis matrix. 
        Each column corresponds to a B-spline basis function evaluated at the parameter grid.
    X : torch.Tensor (N, d) float
        Target points. Each row corresponds to a data point in d-dimensional space.
    reg : float, optional
        Tikhonov regularization parameter.
        Default is 1e-6.

    Returns
    -------
    C : torch.Tensor (M, d) float
        Control points. Each row corresponds to a control point in d-dimensional space.
    
    """
    Bt = B.transpose(0,1)               # (M, N)
    G = Bt @ B                          # (M, M)
    M = G.shape[0]
    Greg = G + torch.eye(M, dtype=B.dtype, device=B.device) * reg
    rhs = Bt @ X                        # (M, d)
    C = torch.linalg.solve(Greg, rhs)   # (M, d)
    return C


def bspline_eval_torch(t_grid, knots, degree, controls):
    basis_matrix = bspline_basis_matrix(t_grid, knots, degree)
    return basis_matrix @ controls

# --------------------------------------------------

def bspline_basis_eval(j, p, t, knots):
    """
    Evaluate a single B-spline basis function at a given parameter using the Cox--de Boor recursion formula.

    The recursion is defined as follows:
        B_j^0(t) = 1 if knots[j] <= t < knots[j+1] else 0, with special handling for the last non-degenerate interval to include the right endpoint.
        B_j^p(t) = ((t - knots[j]) / (knots[j+p] - knots[j])) * B_j^{p-1}(t) + ((knots[j+p+1] - t) / (knots[j+p+1] - knots[j+1])) * B_{j+1}^{p-1}(t) for p > 0,
            with the convention that if a denominator is zero, the corresponding term is defined to be zero (which effectively removes that term from the sum).

    Parameters
    ----------
    j : int
        The index of the basis function (0-based).
    p : int >= 0
        The degree of the B-spline basis function.
    t : float
        The parameter value at which to evaluate the basis function.
    knots : list of float | np.ndarray (m+1,) float
        A vector of knot values (non-decreasing sequence).
    
    Returns
    -------
    value : float
        The value of the B-spline basis function B_j^p(t) at the given parameter t.
    
    Notes
    -----
    - The input `j` should be in the range [0, m-p-1] where m is the number of intervals (number of knots - 1).
        This ensures that the basis function is well-defined for the given degree and knot vector.
    - The input `knots` should have at least p+2 elements to define the basis function.
        This ensures that there exists at least one basis function of degree p for the given knot vector.
    - The output `value` will be non-negative and will be zero outside the support of the basis function.
        The support of B_j^p is [knots[j], knots[j+p+1]) except for the last non-degenerate interval which includes the right endpoint.
    - This implementation is not optimized for performance due to its recursive nature and lack of memoization
        but is straightforward and closely follows the mathematical definition of B-spline basis functions.

    """
    if p == 0:
        # 0-degree basis functions are indicator functions for the intervals defined by the knots
        # support = [left, right) except the last non-degenerate interval which includes the right endpoint

        # find last non-degenerate interval
        last_nondegen_idx = -1
        for jj in range(len(knots) - 2, -1, -1):    # loop from the end
            if knots[jj] < knots[jj + 1]:
                last_nondegen_idx = jj
                break
        
        # include right endpoint of last non-degenerate interval
        if j == last_nondegen_idx and t == knots[j + 1]:
            return 1.0
        else:
            return 1.0 if knots[j] <= t < knots[j + 1] else 0.0
    else:
        coef1 = (t - knots[j]) / (knots[j + p] - knots[j]) if knots[j + p] != knots[j] else 0
        coef2 = (knots[j + p + 1] - t) / (knots[j + p + 1] - knots[j + 1]) if knots[j + p + 1] != knots[j + 1] else 0
        return coef1 * bspline_basis_eval(j, p - 1, t, knots) + coef2 * bspline_basis_eval(j + 1, p - 1, t, knots)


def bspline_eval_naive(t, knots, degree, controls):
    """
    Evaluate the B-spline curve at a given parameter using the control points and the basis functions.

    The B-spline curve is defined as a weighted sum of the control points
     with weights given by the B-spline basis functions:
        S(t) = sum_j B_j^p(t) * control_j
     where B_j^p(t) is the j-th B-spline basis function of degree p evaluated at t,
     and control_j is the j-th control point.
    The basis functions are computed using the Cox--de Boor recursion formula.
    
    Parameters
    ----------
    t : float
        The parameter value at which to evaluate the B-spline curve.
    knots : list of float | np.ndarray (m+1,) float
        A vector of knot values (non-decreasing sequence).
    degree : int >= 0
        The degree of the B-spline basis functions.
    controls : list of list of float | np.ndarray (M, dim) float
        A list of control points, where each control point is an arbitrary dimensional vector.

    Returns
    -------
    point : np.ndarray (dim,) float
        The point on the B-spline curve corresponding to the parameter t.

    Notes
    -----
    - The input `t` should be within the range covered by the `knots` for meaningful evaluation of the curve.
    - The input `knots` should have at least degree + 2 elements to define the basis functions.
    - The input `controls` should have M control points, where M = number of basis functions = number of knots - degree - 1.
    - This version is more memory efficient than computing the entire basis matrix.
      Thus is more suited for plotting.
    
    """
    S = 0
    for j in range(len(controls)):
        B_j_p = bspline_basis_eval(j, degree, t, knots)
        S += B_j_p * controls[j]
    return S


def find_knot_span_naive(t, knots):
    """
    Find the knot span index for a given parameter.

    The knot span index r is defined such that:
        knots[r] <= t < knots[r+1]
    with special handling for the case when t is exactly the last knot value,
     which should return the last non-degenerate span index.
    
    Parameters
    ----------
    t : float
        The parameter value for which to find the knot span.
    knots : list of float | np.ndarray (m+1,) float
        A vector of knot values (non-decreasing sequence).
    
    Returns
    -------
    r : int
        The index of the knot span containing the parameter t.
    
    Notes
    -----
    - The input `t` should be within the range covered by the `knots` for meaningful evaluation of the curve.
    - The input `knots` should have at least 2 elements to define at least one knot span.
    - If `t` is exactly the first knot value, the function will return the first non-degenerate span index.
    - If `t` is exactly the last knot value, the function will return the last non-degenerate span index.

    """
    # handle the case when t is exactly the last knot value (should return the last non-degenerate span index)
    if t == knots[-1]:
        for r in range(len(knots) - 2, -1, -1):
            if knots[r] < knots[r + 1]:    # find the last non-degenerate interval
                return r
        raise ValueError(f"All intervals in the knot vector are degenerate, cannot find a valid knot span for t={t}.")

    # loop from the end to find the last span that satisfies knots[r] <= t
    for r in range(len(knots) - 2, -1, -1):
        if knots[r] <= t < knots[r + 1]:
            return r
    raise ValueError(f"Parameter t={t} is out of bounds of the knot vector.")

def find_knot_span(t, knots):
    """
    Find the knot span index for a given parameter (binary search).

    The knot span index r is defined such that:
        knots[r] <= t < knots[r+1]
    with special handling for the case when t is exactly the last knot value,
     which should return the last non-degenerate span index.
    
    Parameters
    ----------
    t : float
        The parameter value for which to find the knot span.
    knots : list of float | np.ndarray (m+1,) float
        A vector of knot values (non-decreasing sequence).
    
    Returns
    -------
    r : int
        The index of the knot span containing the parameter t.
    
    Notes
    -----
    - The input `t` should be within the range covered by the `knots` for meaningful evaluation of the curve.
    - The input `knots` should have at least 2 elements to define at least one knot span.
    - If `t` is exactly the first knot value, the function will return the first non-degenerate span index.
    - If `t` is exactly the last knot value, the function will return the last non-degenerate span index.

    """
    # exclude duplicate knots at start and end due to clamping, since they do not affect the search.
    hi = len(knots) - 1
    while hi - 1 >= 0 and knots[hi] == knots[hi - 1]:
        hi -= 1
    # handle the case when t is exactly the last knot value (should return the last non-degenerate span index)
    if t == knots[-1]:
        return hi - 1
    lo = 0
    while lo + 1 < len(knots) and knots[lo] == knots[lo + 1]:
        lo += 1
    
    while lo < hi:
        # bias towards the upper half to ensure we find the last knot span that satisfies knots[r] <= t
        mid = (lo + hi + 1) // 2    # when hi = lo + 1 -> mid = hi
        if knots[mid] <= t:
            lo = mid
        else:
            hi = mid - 1            # exclude mid since knots[mid] > t
    return lo


def de_boor(t, knots, degree, controls):
    """
    Evaluate the B-spline curve at a given parameter using the de Boor algorithm.

    The de Boor algorithm is an efficient and stable method for evaluating B-spline curves
     at specific parameter values.
    It uses a recursive approach but, unlike in the Cox--de Boor recursion formula,
     the de Boor algorithm avoids computing basis functions which are zero at the given parameter.
    Instead, it directly computes the point on the curve by performing a series of linear interpolations
     between control points.
    
    Parameters
    ----------
    t : float
        The parameter value at which to evaluate the B-spline curve.
    knots : list of float | np.ndarray (m+1,) float
        A vector of knot values (non-decreasing sequence).
    degree : int >= 0
        The degree of the B-spline basis functions.
    controls : list of list of float | np.ndarray (M, dim) float
        A list of control points, where each control point is an arbitrary dimensional vector.

    Returns
    -------
    point : np.ndarray (dim,) float
        The point on the B-spline curve corresponding to the parameter t.

    Notes
    -----
    - The input `t` should be within the range covered by the `knots` for meaningful evaluation of the curve.
    - The input `knots` should have at least degree + 2 elements to define the basis functions.
    - The input `controls` should have M control points, where M = number of basis functions = number of knots - degree - 1.
    - The algorithm works even in case the knots are not clamped and can handle duplicate knots.
    
    """
    # find knot span index r such that knots[r] <= t < knots[r+1]
    r = find_knot_span(t, knots)

    # initialize d with control points corresponding to the knot span
    d = [controls[r - degree + i] for i in range(degree + 1)]

    # perform de Boor iterations for k=1..degree
    for k in range(1, degree + 1):
        for i in range(degree, k - 1, -1):   # i goes from degree down to k
            alpha_num = t - knots[r - degree + i]
            alpha_den = knots[r + i + 1 - k] - knots[r - degree + i]
            alpha = alpha_num / alpha_den if alpha_den != 0 else 0
            d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i]
    return d[degree]

bspline_eval = de_boor


# ==================================================


# --- Plotting ---


def plot_loss(losses, log=True, 
              path=None, name="training_loss.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss")
    
    plt.xlabel("Iteration")
    if log:
        plt.yscale("log")
        plt.ylabel("Loss (log scale)")
    else:
        plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(path, name))   # save plot to external file


def plot_curve(knots, degree, controls, 
               path=None, name="curve.png"):
    t_grid = np.linspace(0., 1., 1000)  # dense grid for smooth curve
    curve_points = np.array([bspline_eval(t, knots, degree, controls) for t in t_grid])
    knot_points  = np.array([bspline_eval(u, knots, degree, controls) for u in knots])

    plt.figure(figsize=(10, 10))
    plt.plot(curve_points[:, 0], curve_points[:, 1], 
                linestyle='-', color='blue', label='B-spline Curve')
    plt.plot(knot_points[:, 0], knot_points[:, 1], 
                marker='^', linestyle='none', color='green', markersize=10, label='Knot Points')

    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(path, name))   # save plot to external file


def plot_curve_fit(points, knots, degree, controls, error, 
                   path=None, name="curve_fit.png"):
    t_grid = np.linspace(0., 1., 1000)  # dense grid for smooth curve
    curve_points = np.array([bspline_eval(t, knots, degree, controls) for t in t_grid])
    knot_points  = np.array([bspline_eval(u, knots, degree, controls) for u in knots])

    plt.figure(figsize=(10, 8))
    plt.plot(points[:, 0], points[:, 1], 
                marker='o', linestyle='none', color='gray', markersize=4, label='Data Points')
    plt.plot(curve_points[:, 0], curve_points[:, 1], 
                linestyle='-', color='blue', label='B-spline Curve')
    plt.plot(knot_points[:, 0], knot_points[:, 1], 
                marker='^', linestyle='none', color='green', markersize=10, label='Knot Points')

    # make room on the right for legend/text
    plt.subplots_adjust(right=0.75)   # leave 25% width on the right for annotations
    # place legend in the reserved right margin
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    # add a red rectangle in the lower right corner to display the final error
    plt.text(1.03, 0.02, f"Error: {error:.2e}", 
                transform=plt.gca().transAxes,  # convert to percentage coordinates of the axes
                fontsize=12, color='red', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))

    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

    plt.savefig(os.path.join(path, name))   # save plot to external file
