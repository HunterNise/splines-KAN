"""
[task1/KAN2] Evaluation script for the KAN model trained by train.py.

Loads the saved model weights and KAN hyperparameters from outputs/, reconstructs the KAN
with the same architecture as at training time, then runs inference on samples from both
the training set (2d_train.npz) and the eval set (2d_eval.npz), saving per-sample results
(error, knots, controls) and curve-fit plots to outputs/eval/.

Kept separate from train.py so evaluation can be re-run without retraining.
"""


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from kan import KAN

import numpy as np

from source.functions import *


import os


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

# Load parameter file from the output folder

prm_file = os.path.join(output_dir, "train.prm")
if not os.path.exists(prm_file):
    raise FileNotFoundError(f"Parameter file '{prm_file}' not found in the output folder.")
prm = PrmParser().parse(prm_file)

# --------------------------------------------------

# Load the trained model from the output folder

model_file = os.path.join(output_dir, "model.pth")

if not os.path.exists(model_file):
    print(f"Model file {model_file} not found. Please run train.py first.")
    exit(1)

print(f"Loading model from {model_file}")
ckpt = torch.load(model_file, map_location=device)

num_points      = ckpt['num_points']
dim             = ckpt['dim']
num_knots       = ckpt['num_knots']
width           = ckpt['width']
grid_intervals  = ckpt['grid_intervals']
spline_order    = ckpt['spline_order']
degree          = ckpt['degree']

model = KAN(
    width     = width,          # restored from checkpoint: [input] + hidden_layers + [output]
    grid      = grid_intervals,
    k         = spline_order,
    seed      = 0,
    device    = device,
    auto_save = False,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# --------------------------------------------------

eval_dir = os.path.join(output_dir, "eval")
os.makedirs(eval_dir, exist_ok=True)

# evaluate on selected samples from both the training and test sets
modes = ("train", "eval")
for mode in modes:

    # Load points from text file
    
    # derive mode-specific dataset path from the training path stored in the parameter file
    dataset_dir = os.path.dirname(prm.get("Dataset", "Path"))
    path = os.path.join(ROOT, dataset_dir, f"2d_{mode}.npz")   # prepend ROOT since the prm stores a relative path

    dataset = BSplineDataset(path, num_knots, num_points)

    loader = DataLoader(dataset[:10], batch_size=1, shuffle=False)

    # --------------------------------------------------

    # Evaluation loop: run inference, compute fitting error, save results and plots.

    for i, (pts_flat, label, params) in enumerate(loader):
        pts_flat = pts_flat.to(device).squeeze(0)       # (num_points*dim,)
        points   = pts_flat.reshape(num_points, dim)    # (num_points, dim)
        label    = label.to(device).squeeze(0)          # (num_interior,)
        t_grid   = params.to(device).squeeze(0)         # (num_points,)

        with torch.no_grad():
            # KAN forward pass: (1, num_points*dim) -> (1, num_intervals)
            pred_intervals_raw = model(pts_flat.unsqueeze(0))
            pred_intervals = F.softmax(pred_intervals_raw.squeeze(0), dim=0)   # (num_intervals,)
            # cumsum to convert intervals back to knots
            zero = torch.zeros(1, dtype=pred_intervals.dtype, device=pred_intervals.device)
            pred_knots = torch.cumsum(torch.cat((zero, pred_intervals)), dim=0) # (num_knots,)

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


        print(f"\n{mode} sample {i}:")
        print(f"  Error: {err:.16f}")
        print(f"  True interior knots: {label.cpu().numpy()}")
        print(f"  Pred interior knots: {pred_knots[1:-1].cpu().numpy()}")

        # print final results to file
        results_path = os.path.join(eval_dir, f"{mode}{i}-results.txt")
        with open(results_path, "w") as f:
            f.write(f"Error: {err:.16f}\n\n")
            f.write(f"Degree: {degree}\n\n")
            f.write(f"Knots:\n{full_knots}\n\n")
            f.write(f"Controls:\n{controls}\n\n")

        # plot the final curve fit to file
        plot_curve_fit(points, full_knots, degree, controls, err,
                       path = eval_dir, name = f"{mode}{i}-curve_fit.png")
