"""
[task1/NN2] Attribution evaluation — gradient saliency for the NN2 model.

Extends the standard evaluation with Jacobian-based input attribution analysis.
Loads model weights and hyperparameters from outputs/, evaluates on samples from both
the official training set (2d_train.npz) and eval set (2d_eval.npz), and for each sample:
- Computes the prediction error and the predicted knot vector (as in eval.py).
- Computes the saliency Jacobian J[j,i] = ||∂t_j/∂x_i||₂ for each interior knot t_j
  and input point x_i, measuring how sensitive each predicted knot is to each input point.
- Aggregates over all interior knots to produce a per-point importance score.
- Saves results (error, knots, controls, saliency, Jacobian) and a curve_fit + saliency
  overlay plot for each sample to outputs/eval-attr/.
"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

from source.functions import *


import os
import importlib.util


# Set seed for reproducibility
torch.manual_seed(0)

# --------------------------------------------------

# require the output folder to exist (created by train.py)
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
if not os.path.isdir(output_dir):
    raise FileNotFoundError(
        f"Output folder '{output_dir}' not found. "
        "Please run train.py first to train the model."
    )
# Change the working directory to the current file's directory to ensure that the figures are saved in the correct location
os.chdir(output_dir)

# Check if GPU is available and set device accordingly
# the syntax depends on the PyTorch version
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")

# use double precision for better numerical stability
precision = torch.float64
torch.set_default_dtype(precision)
eps = torch.finfo(precision).eps    # machine epsilon for the chosen precision, used as a lower bound for the tolerance to avoid numerical issues

# --------------------------------------------------

# Load parameter file and model class from the output folder

prm_file = os.path.join(output_dir, "train.prm")
if not os.path.exists(prm_file):
    raise FileNotFoundError(f"Parameter file '{prm_file}' not found in the output folder.")
prm = PrmParser().parse(prm_file)

_spec = importlib.util.spec_from_file_location("model", os.path.join(output_dir, "model.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
NN = _mod.NN

# Load the trained model from the output folder

model_file = os.path.join(output_dir, "model.pth")

if os.path.exists(model_file):
    print(f"Loading model from {model_file}")
    ckpt  = torch.load(model_file, map_location=device)
    model = NN(ckpt['num_points'], ckpt['dim'], ckpt['num_knots'],
               ckpt['num_neurons'], ckpt['degree'],
               ckpt.get('dropout', 0.0)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    num_points = ckpt['num_points']
    dim        = ckpt['dim']
    num_knots  = ckpt['num_knots']
    degree     = ckpt['degree']
else:
    print(f"Model file {model_file} not found. Please run train.py first to train the model and save it to the correct location.")
    exit(1)

# --------------------------------------------------

eval_dir = os.path.join(output_dir, "eval-attr")
os.makedirs(eval_dir, exist_ok=True)

# evaluate on selected samples from both the training and test sets
modes = ("train", "eval")
for mode in modes:

    # Load points from text file

    # derive mode-specific dataset path from the training path stored in the parameter file
    dataset_dir = os.path.dirname(prm.get("Dataset", "Path"))
    path = os.path.join(ROOT, dataset_dir, f"2d_{mode}.npz")   # prepend ROOT since the prm stores a relative path

    print(f"\nLoading dataset from {path}")
    dataset = BSplineDataset(path, num_knots, num_points)

    loader  = DataLoader(dataset[:10],  batch_size=1, shuffle=False)

    # --------------------------------------------------

    # Evaluation loop to test the trained model on unseen data points from the test set and compute the B-spline fitting error.

    for i, (pts_flat, label, params) in enumerate(loader):
        pts_flat = pts_flat.to(device).squeeze(0)       # (num_points*dim,)
        points   = pts_flat.reshape(num_points, dim)    # (num_points, dim)
        label    = label.to(device).squeeze(0)          # (num_interior,)
        t_grid   = params.to(device).squeeze(0)         # (num_points,)

        with torch.no_grad():
            pred_knots = model(pts_flat.unsqueeze(0)).squeeze(0)   # (num_knots,)

        full_knots = torch.cat((
            torch.zeros(degree + 1, dtype=pred_knots.dtype, device=pred_knots.device),
            pred_knots[1:-1],
            torch.ones( degree + 1, dtype=pred_knots.dtype, device=pred_knots.device)
        ))

        basis_matrix = bspline_basis_matrix(t_grid, full_knots, degree, soft=False, k=1000.0)
        controls = solve_control_points(basis_matrix, points, reg=1e-6)

        err = torch.sum((points - basis_matrix @ controls) ** 2).item()

        full_knots = full_knots.cpu().numpy()
        controls   = controls.cpu().numpy()
        points_np  = points.cpu().numpy()


        # --- Gradient saliency ---
        # Compute the Jacobian J[j, i] = || ∂t_j / ∂x_i ||_2
        # where t_j is the j-th interior (free) knot and x_i is the i-th input point.
        # This quantifies how sensitive each predicted knot is to each input point.
        x_sal = pts_flat.detach().requires_grad_(True)   # (num_points * dim,)

        pred_knots_sal = model(x_sal.unsqueeze(0)).squeeze(0)
        full_knots_sal = torch.cat((
            torch.zeros(degree + 1, dtype=pred_knots_sal.dtype, device=pred_knots_sal.device),
            pred_knots_sal[1:-1],
            torch.ones( degree + 1, dtype=pred_knots_sal.dtype, device=pred_knots_sal.device)
        ))

        # extract only the free (interior) knots — the endpoints are clamped constants
        interior_knots_sal = full_knots_sal[degree + 1 : -(degree + 1)]
        num_free = interior_knots_sal.shape[0]

        # Jacobian tensor: J[j, i, d] = ∂t_j / ∂x_{i,d}
        jacobian = torch.zeros(num_free, num_points, dim, dtype=precision)
        for j in range(num_free):
            if x_sal.grad is not None:
                x_sal.grad.zero_()
            interior_knots_sal[j].backward(retain_graph=(j < num_free - 1))
            jacobian[j] = x_sal.grad.view(num_points, dim).abs()

        # per-knot per-point saliency: L2 norm over coordinate dims  (num_free, num_points)
        knot_jacobian  = jacobian.norm(dim=-1).detach().cpu().numpy()
        # aggregate over all interior knots -> single importance score per point  (num_points,)
        point_saliency = knot_jacobian.mean(axis=0)


        print(f"\n{mode} sample {i}:")
        print(f"  Error: {err:.16f}")
        print(f"  True interior knots: {label.cpu().numpy()}")
        print(f"  Pred interior knots: {pred_knots[1:-1].cpu().numpy()}")
        #print(f"  Point saliency (mean |∂tⱼ/∂xᵢ|): {point_saliency}")

        # print final results to file
        results_path = os.path.join(eval_dir, f"{mode}{i}-results.txt")
        with open(results_path, "w") as f:
            f.write(f"Error: {err:.16f}\n\n")
            f.write(f"Degree: {degree}\n\n")
            f.write(f"Knots:\n{full_knots}\n\n")
            f.write(f"Controls:\n{controls}\n\n")
            f.write(f"Point saliency (mean |∂tⱼ/∂xᵢ| over interior knots):\n{point_saliency}\n\n")
            f.write(f"Knot-point Jacobian (num_free_knots × num_points):\n{knot_jacobian}\n\n")

        # plot the final curve fit with saliency attribution to file
        plot_curve_fit_saliency(points_np, full_knots, degree, controls, err,
                                point_saliency, knot_jacobian,
                                path = eval_dir, name = f"{mode}{i}-curve_fit.png")
