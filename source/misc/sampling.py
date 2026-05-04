import torch
import numpy as np
from source.functions import *


def uniform_samples(num_points):
    """
    Uniformly sample num_points parameter values in [0, 1].
    """
    return torch.linspace(0.0, 1.0, num_points, dtype=torch.float64)

# def chord_length_samples(full_knots, ctrls, d, num_points, n_iter=20, tol=1e-10):
#     """
#     Find params such that chord_length_params(curve(params)) == params.
#     Fixed-point iteration starting from uniform initialization.
#     """
#     params = torch.linspace(0.0, 1.0, num_points, dtype=full_knots.dtype)

#     for _ in range(n_iter):
#         B      = bspline_basis_matrix(params, full_knots, d, soft=False)
#         pts    = B @ ctrls                        # (num_points, dim)
#         params_new = chord_length_params(pts)     # (num_points,)

#         if torch.max(torch.abs(params_new - params)) < tol:
#             break
#         params = params_new

#     return params

def chord_length_samples(full_knots, ctrls, d, num_points, n_dense=2000):
    """
    Sample num_points parameter values such that chord_length_params(curve(params)) == linspace(0,1,N).
    Uses arc-length inversion on a dense uniform evaluation.
    """
    # 1. Dense evaluation to approximate arc-length function s(t)
    t_dense = torch.linspace(0.0, 1.0, n_dense, dtype=full_knots.dtype)
    B_dense = bspline_basis_matrix(t_dense, full_knots, d, soft=False)
    pts_dense = B_dense @ ctrls                                     # (n_dense, dim)

    # 2. Cumulative chord lengths, normalized to [0, 1]  →  s(t_dense[i])
    seg_lengths = torch.norm(pts_dense[1:] - pts_dense[:-1], dim=1) # (n_dense-1,)
    cumlen = torch.cat([torch.zeros(1, dtype=seg_lengths.dtype),
                        torch.cumsum(seg_lengths, dim=0)])           # (n_dense,)
    total = cumlen[-1]
    if total == 0:
        return torch.linspace(0.0, 1.0, num_points, dtype=full_knots.dtype)
    s = cumlen / total                                               # arc-length param in [0,1]

    # 3. Invert s(t): find t at equally-spaced arc-length fractions via linear interpolation
    s_target = torch.linspace(0.0, 1.0, num_points, dtype=full_knots.dtype)
    params = torch.tensor(
        np.interp(s_target.numpy(), s.numpy(), t_dense.numpy()),
        dtype=full_knots.dtype
    )
    return params


full_knots = np.loadtxt("data/DNN-Solver/bspline-data/knot/99_knot.txt")
full_knots = torch.tensor(full_knots, dtype=torch.float64)
ctrl_pts   = np.loadtxt("data/DNN-Solver/bspline-data/control/99_control.txt")
ctrl_pts   = torch.tensor(ctrl_pts, dtype=torch.float64)
degree = 3
num_points = 100

uniform_params      = uniform_samples(num_points)
print("Uniform params:", uniform_params)
chord_length_params = chord_length_samples(full_knots, ctrl_pts, degree, num_points)
print("Chord length params:", chord_length_params)


output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

def plot_params(params, title, filename):
    plt.figure(figsize=(10, 10))
    plt.plot(params.numpy(), label=title)
    plt.legend()
    plt.title(f"{title} Parameter Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Parameter Value")
    plt.savefig(os.path.join(output_dir, filename))

def plot_samples(params, title, filename):
    B   = bspline_basis_matrix(params, full_knots, degree, soft=False)
    pts = B @ ctrl_pts
    pts = pts.numpy()
    
    plt.figure(figsize=(10, 10))
    plt.plot(pts[:, 0], pts[:, 1],
                marker='o', linestyle='none', color='gray', markersize=4, label='Data Points')
    plt.legend()
    plt.title(f"{title} Sampled Points on B-spline Curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(output_dir, filename))

plot_params(uniform_params, "Uniform", "uniform_params.png")
plot_params(chord_length_params, "Chord Length", "chord_length_params.png")

plot_samples(uniform_params, "Uniform", "uniform_samples.png")
plot_samples(chord_length_params, "Chord Length", "chord_length_samples.png")
