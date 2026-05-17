"""
[task1/KAN4] Evaluation script for the sliding-window KAN trained by train_gpu.py.

Loads the saved model weights and all hyperparameters from outputs/, reconstructs the
sliding-window KAN with the same architecture and forward pipeline as at training time,
then runs inference on samples from both the training set (2d_train.npz) and the eval
set (2d_eval.npz), saving per-sample results (error, knots, controls) and curve-fit
plots to outputs/eval/.

Changes vs KAN3/eval.py
------------------------
- Loads quantile_sharpness and mult_arity from the checkpoint instead of
  histogram_bandwidth (which is no longer stored in model.pth).
- Reconstructs _tau and _alpha (quantile target levels) instead of hist_weights.
- kan_forward_to_knots implements the density → CDF → soft-quantile pipeline:
    scores    = model(extract_windows(pts))          (B, N_w)
    density   = softplus(scores) + 1e-8              positive, normalised
    CDF       = cumsum(density, dim=1)               monotone, approaches 1
    weights   = softmax(-S * |CDF - alpha_j|, i)     (B, N_w, K)
    knots[j]  = (tau * weights[:,j]).sum(dim=1)       (B, K) interior knots
- Model is constructed with mult_arity to match the training-time architecture.
- extract_windows computes chord lengths and turning angles (rotation-invariant
  intrinsic features), matching train_gpu.py exactly.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kan import KAN

import numpy as np

from source.functions import *

import os


# Set seed for reproducibility
torch.manual_seed(0)

# --------------------------------------------------

# Require the output folder to exist (created by train_gpu.py)
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
if not os.path.isdir(output_dir):
    raise FileNotFoundError(
        f"Output folder '{output_dir}' not found. "
        "Please run train_gpu.py first to train the model."
    )
os.chdir(output_dir)

# Check if GPU is available and set device accordingly
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")

# Use double precision for better numerical stability
precision = torch.float64
torch.set_default_dtype(precision)
eps = torch.finfo(precision).eps

# --------------------------------------------------

# Load parameter file from the output folder (copied there by train_gpu.py)

prm_file = os.path.join(output_dir, "train.prm")
if not os.path.exists(prm_file):
    raise FileNotFoundError(f"Parameter file '{prm_file}' not found in the output folder.")
prm = PrmParser().parse(prm_file)

# --------------------------------------------------

# Load the trained model and all hyperparameters from the output folder.
# All pipeline parameters are stored in model.pth so that eval.py can reconstruct
# the full forward pass without re-reading train.prm.

model_file = os.path.join(output_dir, "model.pth")
if not os.path.exists(model_file):
    print(f"Model file {model_file} not found. Please run train_gpu.py first.")
    exit(1)

print(f"Loading model from {model_file}")
ckpt = torch.load(model_file, map_location=device)

num_points         = ckpt['num_points']
dim                = ckpt['dim']
num_knots          = ckpt['num_knots']
width              = ckpt['width']
grid_intervals     = ckpt['grid_intervals']
spline_order       = ckpt['spline_order']
degree             = ckpt['degree']
window_size        = ckpt['window_size']
stride             = ckpt['stride']
quantile_sharpness = ckpt['quantile_sharpness']
mult_arity         = ckpt['mult_arity']
n_features         = ckpt['n_features']

model = KAN(
    width      = width,
    grid       = grid_intervals,
    k          = spline_order,
    mult_arity = mult_arity,
    seed       = 0,
    device     = device,
    auto_save  = False,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# --------------------------------------------------

# Reconstruct sliding-window geometry and quantile parameters.
# These must match the construction in train_gpu.py exactly so that the forward
# pass at eval time is numerically identical to the training forward pass.

N_w          = (num_points - window_size) // stride + 1
num_interior = num_knots - 2

# Window parameter positions τ_i = i/(N_w-1), used in the weighted centroid.
_tau   = torch.linspace(0.0, 1.0, N_w, dtype=precision, device=device)  # (N_w,)

# Quantile targets α_j = (j+1)/(K+1) for j = 0..K-1.
# These implement the equidistribution principle: each interior knot captures
# an equal share of the total learned curvature weight.
_alpha = torch.linspace(
    1.0 / (num_interior + 1),
    float(num_interior) / (num_interior + 1),
    num_interior, dtype=precision, device=device
)   # (num_interior,)

# Pre-allocated padding for build_full_knots
_zeros_pad = torch.zeros(1, degree + 1, dtype=precision, device=device)
_ones_pad  = torch.ones( 1, degree + 1, dtype=precision, device=device)

# --------------------------------------------------

def extract_windows(pts_batch):
    """
    Extract rotation-invariant intrinsic features from sliding windows.

    Identical to train_gpu.py: chord lengths + turning angles (not centroid-subtracted
    raw coordinates).  Must match training exactly for eval results to be meaningful.

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    torch.Tensor (B * N_w, n_features)
    """
    B      = pts_batch.shape[0]
    pts_3d = pts_batch.view(B, num_points, dim)

    bb_range     = pts_3d.max(dim=1).values - pts_3d.min(dim=1).values
    global_scale = bb_range.max(dim=1).values.clamp(min=1e-8)

    windows = pts_3d.unfold(1, window_size, stride)           # (B, N_w, dim, window_size)
    windows = windows.permute(0, 1, 3, 2).contiguous()        # (B, N_w, window_size, dim)

    chords = windows[:, :, 1:, :] - windows[:, :, :-1, :]    # (B, N_w, window_size-1, dim)
    chords = chords / global_scale.view(B, 1, 1, 1)

    chord_lengths = chords.norm(dim=-1)                        # (B, N_w, window_size-1)

    d_cur  = chords[:, :, :-1, :]
    d_next = chords[:, :,  1:, :]
    d_cur_n  = d_cur  / d_cur.norm( dim=-1, keepdim=True).clamp(min=1e-8)
    d_next_n = d_next / d_next.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cross  = d_cur_n[..., 0] * d_next_n[..., 1] - d_cur_n[..., 1] * d_next_n[..., 0]
    dot    = (d_cur_n * d_next_n).sum(dim=-1)
    angles = torch.atan2(cross, dot)                           # (B, N_w, window_size-2)

    features = torch.cat([chord_lengths, angles], dim=-1)
    return features.reshape(B * N_w, n_features)


def kan_forward_to_knots(pts_batch):
    """
    Full sliding-window forward pass: curve points → knot vector in [0, 1].

    Mirrors kan_forward_to_knots from train_gpu.py (model, _tau, _alpha,
    quantile_sharpness, num_interior captured from the enclosing scope).

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    pred_knots : torch.Tensor (B, num_knots)   [0, interior knots..., 1]
    """
    B           = pts_batch.shape[0]
    kan_input   = extract_windows(pts_batch)          # (B*N_w, n_features)
    scores_flat = model(kan_input)                    # (B*N_w, 1)
    scores      = scores_flat.view(B, N_w)            # (B, N_w)

    density = F.softplus(scores) + 1e-8
    density = density / density.sum(dim=1, keepdim=True)
    CDF     = torch.cumsum(density, dim=1)            # (B, N_w)

    cdf_ex   = CDF.unsqueeze(2)                       # (B, N_w, K)
    alpha_ex = _alpha.view(1, 1, num_interior)        # (1,  1,  K)
    tau_ex   = _tau.view(1, N_w, 1)                   # (1, N_w,  1)

    weights        = F.softmax(-quantile_sharpness * (cdf_ex - alpha_ex).abs(), dim=1)
    interior_knots = (tau_ex * weights).sum(dim=1)    # (B, K)

    zero = torch.zeros(B, 1, dtype=precision, device=device)
    one  = torch.ones( B, 1, dtype=precision, device=device)
    return torch.cat([zero, interior_knots, one], dim=1)   # (B, num_knots)

# --------------------------------------------------

eval_dir = os.path.join(output_dir, "eval")
os.makedirs(eval_dir, exist_ok=True)

# Evaluate on selected samples from both the training and eval sets
modes = ("train", "eval")
for mode in modes:

    # Derive mode-specific dataset path from the training path stored in the parameter file
    dataset_dir = os.path.dirname(prm.get("Dataset", "Path"))
    path = os.path.join(ROOT, dataset_dir, f"2d_{mode}.npz")

    dataset = BSplineDataset(path, num_knots, num_points)
    loader  = DataLoader(dataset[:10], batch_size=1, shuffle=False)

    for i, (pts_flat, label, params) in enumerate(loader):
        pts_flat = pts_flat.to(device).squeeze(0)       # (num_points*dim,)
        points   = pts_flat.reshape(num_points, dim)    # (num_points, dim)
        label    = label.to(device).squeeze(0)          # (num_interior,)
        t_grid   = params.to(device).squeeze(0)         # (num_points,)

        with torch.no_grad():
            # Sliding-window forward pass: (1, num_points*dim) → (1, num_knots)
            pred_knots = kan_forward_to_knots(pts_flat.unsqueeze(0)).squeeze(0)  # (num_knots,)

        full_knots = torch.cat((
            torch.zeros(degree + 1, dtype=pred_knots.dtype, device=pred_knots.device),
            pred_knots[1:-1],
            torch.ones( degree + 1, dtype=pred_knots.dtype, device=pred_knots.device)
        ))

        basis_matrix = bspline_basis_matrix(t_grid, full_knots, degree, soft=False, k=1000.0)
        controls     = solve_control_points(basis_matrix, points, reg=1e-6)

        err = torch.sum((points - basis_matrix @ controls) ** 2).item()

        points     = points.cpu().numpy()
        full_knots = full_knots.cpu().numpy()
        controls   = controls.cpu().numpy()

        print(f"\n{mode} sample {i}:")
        print(f"  Error: {err:.16f}")
        print(f"  True interior knots: {label.cpu().numpy()}")
        print(f"  Pred interior knots: {pred_knots[1:-1].cpu().numpy()}")

        results_path = os.path.join(eval_dir, f"{mode}{i}-results.txt")
        with open(results_path, "w") as f:
            f.write(f"Error: {err:.16f}\n\n")
            f.write(f"Degree: {degree}\n\n")
            f.write(f"Knots:\n{full_knots}\n\n")
            f.write(f"Controls:\n{controls}\n\n")

        plot_curve_fit(points, full_knots, degree, controls, err,
                       path=eval_dir, name=f"{mode}{i}-curve_fit.png")
