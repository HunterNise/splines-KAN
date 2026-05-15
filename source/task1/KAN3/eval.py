"""
[task1/KAN3] Evaluation script for the sliding-window KAN trained by train_gpu.py.

Loads the saved model weights and all hyperparameters from outputs/, reconstructs the
sliding-window KAN with the same architecture and forward pipeline as at training time,
then runs inference on samples from both the training set (2d_train.npz) and the eval
set (2d_eval.npz), saving per-sample results (error, knots, controls) and curve-fit
plots to outputs/eval/.

Changes vs KAN2/eval.py
------------------------
- Loads the additional checkpoint fields: window_size, stride, histogram_bandwidth.
- Reconstructs N_w, n_intervals, hist_weights (soft-histogram aggregation weights) from
  those fields — identical to the construction in train_gpu.py.
- Defines extract_windows (per-window centroid subtraction + global bounding-box scale
  division) and kan_forward_to_knots (full sliding-window pipeline) so the forward pass
  at eval time exactly mirrors training.
- The KAN width is [window_size*dim, *hidden, 1] (scalar output per window) rather than
  [num_points*dim, *hidden, num_intervals].
- The eval loop passes pts_flat.unsqueeze(0) through kan_forward_to_knots (which handles
  window extraction and histogram aggregation internally) and squeezes the result, so the
  rest of the loop (basis matrix, control-point solve, plotting) is unchanged.
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
# Change the working directory to the output folder so figures are saved there
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

# Load parameter file from the output folder

prm_file = os.path.join(output_dir, "train.prm")
if not os.path.exists(prm_file):
    raise FileNotFoundError(f"Parameter file '{prm_file}' not found in the output folder.")
prm = PrmParser().parse(prm_file)

# --------------------------------------------------

# Load the trained model and all hyperparameters from the output folder.
# All sliding-window parameters are stored in model.pth so that eval.py can reconstruct
# the full forward pipeline without re-reading train.prm.

model_file = os.path.join(output_dir, "model.pth")
if not os.path.exists(model_file):
    print(f"Model file {model_file} not found. Please run train_gpu.py first.")
    exit(1)

print(f"Loading model from {model_file}")
ckpt = torch.load(model_file, map_location=device)

num_points          = ckpt['num_points']
dim                 = ckpt['dim']
num_knots           = ckpt['num_knots']
width               = ckpt['width']
grid_intervals      = ckpt['grid_intervals']
spline_order        = ckpt['spline_order']
degree              = ckpt['degree']
window_size         = ckpt['window_size']
stride              = ckpt['stride']
histogram_bandwidth = ckpt['histogram_bandwidth']

model = KAN(
    width     = width,   # [window_size*dim, *hidden_layers, 1]
    grid      = grid_intervals,
    k         = spline_order,
    seed      = 0,
    device    = device,
    auto_save = False,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# --------------------------------------------------

# Reconstruct sliding-window geometry and soft-histogram weights.
# These are identical to the construction in train_gpu.py; they must match exactly
# so that the eval forward pass is numerically identical to the training forward pass.

n_intervals = num_knots - 1
N_w         = (num_points - window_size) // stride + 1

_tau         = torch.linspace(0.0, 1.0, N_w, dtype=precision, device=device)
_bin_centers = (torch.arange(n_intervals, dtype=precision, device=device) + 0.5) / n_intervals
_sigma       = histogram_bandwidth / n_intervals
hist_weights = torch.exp(
    -0.5 * ((_tau.unsqueeze(1) - _bin_centers.unsqueeze(0)) / _sigma) ** 2
)   # (N_w, n_intervals)

# --------------------------------------------------

def extract_windows(pts_batch):
    """
    Extract normalised sliding windows from a batch of curves.

    Identical to the function in train_gpu.py (module-level variables num_points,
    dim, window_size, stride, N_w are captured from the enclosing scope).

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    torch.Tensor (B * N_w, window_size * dim)
    """
    B      = pts_batch.shape[0]
    pts_3d = pts_batch.view(B, num_points, dim)

    windows  = pts_3d.unfold(1, window_size, stride)            # (B, N_w, dim, window_size)
    windows  = windows.permute(0, 1, 3, 2).contiguous()         # (B, N_w, window_size, dim)

    centroid = windows.mean(dim=2, keepdim=True)
    windows  = windows - centroid

    bb_range     = pts_3d.max(dim=1).values - pts_3d.min(dim=1).values   # (B, dim)
    global_scale = bb_range.max(dim=1).values.clamp(min=1e-8)            # (B,)
    windows      = windows / global_scale.view(B, 1, 1, 1)

    return windows.reshape(B * N_w, window_size * dim)


def kan_forward_to_knots(pts_batch):
    """
    Full sliding-window forward pass: curve points → knots in [0, 1].

    Mirrors kan_forward_to_knots from train_gpu.py (model and hist_weights are
    captured from the enclosing scope).

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    pred_knots : torch.Tensor (B, num_knots)
    """
    B              = pts_batch.shape[0]
    kan_input      = extract_windows(pts_batch)          # (B*N_w, window_size*dim)
    scores_flat    = model(kan_input)                    # (B*N_w, 1)
    scores         = scores_flat.view(B, N_w)            # (B, N_w)
    bin_scores     = scores @ hist_weights               # (B, n_intervals)
    pred_intervals = F.softmax(bin_scores, dim=1)        # (B, n_intervals)
    zero = torch.zeros(B, 1, dtype=pred_intervals.dtype, device=pred_intervals.device)
    return torch.cumsum(torch.cat((zero, pred_intervals), dim=1), dim=1)   # (B, num_knots)

# --------------------------------------------------

eval_dir = os.path.join(output_dir, "eval")
os.makedirs(eval_dir, exist_ok=True)

# Evaluate on selected samples from both the training and test sets
modes = ("train", "eval")
for mode in modes:

    # Derive mode-specific dataset path from the training path stored in the parameter file
    dataset_dir = os.path.dirname(prm.get("Dataset", "Path"))
    path = os.path.join(ROOT, dataset_dir, f"2d_{mode}.npz")

    dataset = BSplineDataset(path, num_knots, num_points)
    loader  = DataLoader(dataset[:10], batch_size=1, shuffle=False)

    # --------------------------------------------------

    # Evaluation loop: run inference, compute fitting error, save results and plots.

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
