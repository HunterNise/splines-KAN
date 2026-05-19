"""
[task1/KAN5] Evaluation script for the per-point KAN trained by train_gpu.py.

Loads the saved Stage 1 (and optionally Stage 2) model weights and all
hyperparameters from outputs/, reconstructs the full forward pipeline, then
runs inference on samples from both the training set (2d_train.npz) and the
eval set (2d_eval.npz), saving per-sample results and curve-fit plots to
outputs/eval/.

Changes vs KAN4/eval.py
------------------------
- Loads rebalancer_state_dict and rebalancer_width from the checkpoint and
  reconstructs the Stage 2 rebalancer KAN if rebalancer_hidden_nodes > 0.
- kan_forward_to_knots applies Stage 2 after Stage 1 if the rebalancer is
  present (identical logic to train_gpu.py).
- In-variable labels updated for 3-feature per-point input: [l_0, l_1, θ_0].
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

precision = torch.float64
torch.set_default_dtype(precision)
eps = torch.finfo(precision).eps

# --------------------------------------------------

prm_file = os.path.join(output_dir, "train.prm")
if not os.path.exists(prm_file):
    raise FileNotFoundError(f"Parameter file '{prm_file}' not found in the output folder.")
prm = PrmParser().parse(prm_file)

# --------------------------------------------------

model_file = os.path.join(output_dir, "model.pth")
if not os.path.exists(model_file):
    print(f"Model file {model_file} not found. Please run train_gpu.py first.")
    exit(1)

print(f"Loading model from {model_file}")
ckpt = torch.load(model_file, map_location=device)

num_points              = ckpt['num_points']
dim                     = ckpt['dim']
num_knots               = ckpt['num_knots']
width                   = ckpt['width']
rebalancer_width        = ckpt['rebalancer_width']
grid_intervals          = ckpt['grid_intervals']
spline_order            = ckpt['spline_order']
degree                  = ckpt['degree']
window_size             = ckpt['window_size']
stride                  = ckpt['stride']
quantile_sharpness      = ckpt['quantile_sharpness']
mult_arity              = ckpt['mult_arity']
n_features              = ckpt['n_features']
rebalancer_hidden_nodes = ckpt['rebalancer_hidden_nodes']
rebalancer_delta_scale  = ckpt['rebalancer_delta_scale']

# Reconstruct Stage 1 KAN
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

# Reconstruct Stage 2 rebalancer (if enabled during training)
if rebalancer_hidden_nodes > 0 and ckpt.get('rebalancer_state_dict') is not None:
    rebalancer_model = KAN(
        width     = rebalancer_width,
        grid      = grid_intervals,
        k         = spline_order,
        seed      = 1,
        device    = device,
        auto_save = False,
    )
    rebalancer_model.load_state_dict(ckpt['rebalancer_state_dict'])
    rebalancer_model.eval()
else:
    rebalancer_model = None

# --------------------------------------------------

# Reconstruct sliding-window geometry and quantile parameters.

N_w          = (num_points - window_size) // stride + 1
num_interior = num_knots - 2

_tau   = torch.linspace(0.0, 1.0, N_w, dtype=precision, device=device)  # (N_w,)
_alpha = torch.linspace(
    1.0 / (num_interior + 1),
    float(num_interior) / (num_interior + 1),
    num_interior, dtype=precision, device=device
)   # (num_interior,)

_zeros_pad = torch.zeros(1, degree + 1, dtype=precision, device=device)
_ones_pad  = torch.ones( 1, degree + 1, dtype=precision, device=device)

_min_knot_gap = 0.02


def _enforce_min_gap(knots):
    cols = knots.unbind(dim=1)
    pushed = [cols[0]]
    for c in cols[1:]:
        pushed.append(torch.maximum(c, pushed[-1] + _min_knot_gap))
    return torch.stack(pushed, dim=1).clamp(1e-3, 1.0 - 1e-3)


# --------------------------------------------------

def extract_windows(pts_batch):
    """
    Extract rotation-invariant intrinsic features from sliding windows.

    Identical to train_gpu.py: chord lengths + turning angles.  Must match
    training exactly for eval results to be meaningful.

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

    windows = pts_3d.unfold(1, window_size, stride)            # (B, N_w, dim, window_size)
    windows = windows.permute(0, 1, 3, 2).contiguous()         # (B, N_w, window_size, dim)

    chords = windows[:, :, 1:, :] - windows[:, :, :-1, :]     # (B, N_w, window_size-1, dim)
    chords = chords / global_scale.view(B, 1, 1, 1)

    chord_lengths = chords.norm(dim=-1).clamp(min=1e-8)

    d_cur    = chords[:, :, :-1, :]
    d_next   = chords[:, :,  1:, :]
    d_cur_n  = d_cur  / d_cur.norm( dim=-1, keepdim=True).clamp(min=1e-8)
    d_next_n = d_next / d_next.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cross  = d_cur_n[..., 0] * d_next_n[..., 1] - d_cur_n[..., 1] * d_next_n[..., 0]
    dot    = (d_cur_n * d_next_n).sum(dim=-1)
    angles = torch.atan2(cross, dot)

    features = torch.cat([chord_lengths, angles.abs()], dim=-1)
    return features.reshape(B * N_w, n_features)


def kan_forward_to_knots(pts_batch):
    """
    Full forward pass: curve points → knot vector in [0, 1].

    Mirrors kan_forward_to_knots from train_gpu.py.  Applies Stage 2
    rebalancer if present (loaded from checkpoint).

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    pred_knots : torch.Tensor (B, num_knots)   [0, interior knots..., 1]
    """
    B           = pts_batch.shape[0]
    kan_input   = extract_windows(pts_batch)
    scores_flat = model(kan_input)
    scores      = scores_flat.view(B, N_w).clamp(-20.0, 20.0)

    density = F.softplus(scores) + 1e-8
    density = density / density.sum(dim=1, keepdim=True)
    CDF     = torch.cumsum(density, dim=1)

    cdf_ex   = CDF.unsqueeze(2)
    alpha_ex = _alpha.view(1, 1, num_interior)
    tau_ex   = _tau.view(1, N_w, 1)

    weights        = F.softmax(-quantile_sharpness * (cdf_ex - alpha_ex).abs(), dim=1)
    interior_knots = (tau_ex * weights).sum(dim=1)

    interior_knots, _ = interior_knots.sort(dim=1)

    if rebalancer_model is not None:
        delta          = rebalancer_model(interior_knots)
        interior_knots = interior_knots + rebalancer_delta_scale * torch.tanh(delta)
        interior_knots, _ = interior_knots.sort(dim=1)
        interior_knots    = interior_knots.clamp(1e-3, 1.0 - 1e-3)

    zero = torch.zeros(B, 1, dtype=precision, device=device)
    one  = torch.ones( B, 1, dtype=precision, device=device)
    return torch.cat([zero, interior_knots, one], dim=1)

# --------------------------------------------------

eval_dir = os.path.join(output_dir, "eval")
os.makedirs(eval_dir, exist_ok=True)

# Evaluate on selected samples from both the training and eval sets
modes = ("train", "eval")
for mode in modes:

    dataset_dir = os.path.dirname(prm.get("Dataset", "Path"))
    path = os.path.join(ROOT, dataset_dir, f"2d_{mode}.npz")

    dataset = BSplineDataset(path, num_knots, num_points)
    loader  = DataLoader(dataset[:10], batch_size=1, shuffle=False)

    for i, (pts_flat, label, params) in enumerate(loader):
        pts_flat = pts_flat.to(device).squeeze(0)
        points   = pts_flat.reshape(num_points, dim)
        label    = label.to(device).squeeze(0)
        t_grid   = params.to(device).squeeze(0)

        with torch.no_grad():
            pred_knots = kan_forward_to_knots(pts_flat.unsqueeze(0)).squeeze(0)

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
