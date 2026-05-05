"""
[task1/NN1] Evaluation script for the model trained by train.py.

Loads the saved model weights and runs inference on samples from both the training and
test sets, saving per-sample results (error, knots, controls) and curve-fit plots to
outputs/eval/.

Kept separate from train.py so evaluation can be re-run without retraining.
"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

from source.functions import *


import os

from tqdm import tqdm


# Set seed for reproducibility
torch.manual_seed(0)

# --------------------------------------------------

# create output folder if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)
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

# Load model from file

num_knots = 6       # number of knots (without repetitions/clamping)
num_points = 100    # number of data points sampled from the B-spline curve
degree = 3          # degree of the B-spline curve
method = "uniform"  # method to compute parameter values corresponding to data points
dim = 2


model_file = os.path.join(output_dir, "model.pth")


class NN(nn.Module):
    # class contructor to initialize the neural network architecture and parameters
    def __init__(self, num_knots=5, num_neurons=128, degree=3):
        super().__init__()              # call parent constructor
        
        self.num_knots = num_knots      # store number of knots as object variable for later use
        self.degree = degree            # store degree of B-spline as object variable for later use
        
        # the knot vector must be non-decreasing, so we predict intervals between knots and then convert back to knots
        num_intervals = num_knots - 1
        
        self.stack = nn.Sequential(
            nn.Linear(num_points * dim, num_neurons),   # input layer: takes flattened data points as input
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_intervals),
            nn.Softmax(dim=1)       # ensure intervals are positive and sum to 1; dim=1 applies softmax across the correct dimension for batch processing
        )

    def forward(self, x):
        x = self.stack(x)          # pass through the stack of layers
        # intervals_to_knots only handles 1D; prepend zero column and cumsum manually
        zero = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        x = torch.cumsum(torch.cat((zero, x), dim=1), dim=1)  # (batch, num_knots)
        return x


num_neurons = 512                   # number of neurons in each hidden layer
# create model instance and move to device
model = NN(num_knots, num_neurons, degree).to(device)


if os.path.exists(model_file):
    print(f"Loading model from {model_file}")
    model.load_state_dict(torch.load(model_file, map_location=device))
else:
    print(f"Model file {model_file} not found. Please run train.py first to train the model and save it to the correct location.")
    exit(1)

# --------------------------------------------------

# # Load 2D points from text file

# path = "/app/data/DNN-Solver/bspline-data"

# class BSplineDataset(Dataset):
#     def __init__(self, indices, path, degree=3):
#         self.samples = []
#         for i in indices:
#             pts   = np.loadtxt(f"{path}/pts/spl_data{i:02d}.txt")   # (100, 2)
#             knots = np.loadtxt(f"{path}/knot/{i:02d}_knot.txt")     # variable length
#             interior = knots[degree+1 : -(degree+1)]                # strictly interior knots in (0,1)
            
#             pts_t   = torch.from_numpy(pts).to(precision)
#             label_t = torch.from_numpy(interior).to(precision)
#             self.samples.append((pts_t.flatten(), label_t))
        
#         self.pts_shape = pts.shape
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         return self.samples[idx]


# idx = (num_knots - 5) * 10
# dataset = BSplineDataset(range(idx, idx + 10), path, degree=degree)
# num_points, dim = dataset.pts_shape

# loader  = DataLoader(dataset,  batch_size=1, shuffle=False)

# # --------------------------------------------------

# # Evaluation loop to test the trained model on unseen data points from the test set and compute the B-spline fitting error.

# eval_dir = os.path.join(output_dir, "eval")
# os.makedirs(eval_dir, exist_ok=True)

# for i, (pts_flat, label) in enumerate(loader):
#     pts_flat = pts_flat.to(device).squeeze(0)       # (num_points*dim,)
#     label    = label.to(device).squeeze(0)          # (num_interior,)

#     points = pts_flat.reshape(num_points, dim)
#     t_grid = make_grid(points, method=method)

#     with torch.no_grad():
#         pred_knots = model(pts_flat.unsqueeze(0)).squeeze(0)   # (num_knots,)

#     full_knots = torch.cat((
#         torch.zeros(degree + 1, dtype=pred_knots.dtype, device=pred_knots.device),
#         pred_knots[1:-1],
#         torch.ones( degree + 1, dtype=pred_knots.dtype, device=pred_knots.device)
#     ))

#     basis_matrix = bspline_basis_matrix(t_grid, full_knots, degree, soft=False, k=1000.0)
#     controls = solve_control_points(basis_matrix, points, reg=1e-6)

#     err = torch.sum((points - basis_matrix @ controls) ** 2).item()

#     full_knots = full_knots.cpu().numpy()
#     controls = controls.cpu().numpy()


#     print(f"\n{split.capitalize()} sample {i}:")
#     print(f"  Error: {err:.16f}")
#     print(f"  True interior knots: {label.cpu().numpy()}")
#     print(f"  Pred interior knots: {pred_knots[1:-1].cpu().numpy()}")

#     # print final results to file
#     results_path = os.path.join(eval_dir, f"{split}{i}-results.txt")
#     with open(results_path, "w") as f:
#         f.write(f"Error: {err:.16f}\n\n")
#         f.write(f"Degree: {degree}\n\n")
#         f.write(f"Knots:\n{full_knots}\n\n")
#         f.write(f"Controls:\n{controls}\n\n")

#     # plot the final curve fit to file
#     plot_curve_fit(points, full_knots, degree, controls, err, 
#                     path = eval_dir, name = f"{split}{i}-curve_fit.png")


# --------------------------------------------------
# --------------------------------------------------

# Load points from text file

num_knots = 6       # number of knots (without repetitions/clamping)
num_points = 100    # number of data points sampled from the B-spline curve
method = "uniform"  # method to compute parameter values corresponding to data points


path = "/app/data/SplinegenDataset/2d_eval.npz"

class BSplineDataset(Dataset):
    def __init__(self, path, num_knots, num_points=100):
        self.samples = []
        self.num_knots = num_knots
        self.num_points = num_points
        self.method = method

        with np.load(path) as data:
            ctrl_pts    = data['ctrl_pts']
            knots       = data['knots']
            degree      = data['degree']

        self.degree = int(degree)
        self.dim = ctrl_pts.shape[2]

        
        # filter curves whose knot count matches num_knots
        
        d = self.degree
        # formula derived from the padded representation: count_nonzero counts
        # the (num_knots-2) interior knots plus (d+1) trailing ones
        knot_count = np.count_nonzero(knots, axis=1) + (d + 1) - 2 * d
        mask = knot_count == num_knots      # bool (N_ds,)

        ctrl_pts = ctrl_pts[mask]           # (N, max_ctrl, dim)
        knots    = knots[mask]              # (N, max_knots_padded)

        # actual (unpadded) sizes for this num_knots
        full_knot_len   = num_knots + 2 * d         # full clamped knot vector length
        n_ctrl          = num_knots + d - 1         # number of control points

        
        # resample curves at num_points uniform parameter values in [0, 1]
        
        t_uniform = torch.linspace(0.0, 1.0, num_points, dtype=torch.float64)

        for i in range(len(ctrl_pts)):
            full_knots  = torch.tensor(knots[i, :full_knot_len], dtype=torch.float64)
            ctrls       = torch.tensor(ctrl_pts[i, :n_ctrl],     dtype=torch.float64)   # (n_ctrl, dim)

            # evaluate B-spline at num_points uniform parameter values
            B   = bspline_basis_matrix(t_uniform, full_knots, d, soft=False)    # (num_points, n_ctrl)
            pts = B @ ctrls                                                     # (num_points, dim)

            # interior knots as labels (exclude the d+1 leading zeros and d+1 trailing ones)
            interior_knots = full_knots[d + 1 : -(d + 1)]   # (num_knots - 2,)

            self.samples.append((pts.reshape(-1), interior_knots))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


dataset = BSplineDataset(path, num_knots, num_points)
degree  = dataset.degree        # degree of the B-spline curve
dim     = dataset.dim           # dimension of the data points (2 for 2D, 3 for 3D)

loader  = DataLoader(dataset[:10],  batch_size=1, shuffle=False)

# --------------------------------------------------

# Evaluation loop to test the trained model on unseen data points from the test set and compute the B-spline fitting error.

eval_dir = os.path.join(output_dir, "eval")
os.makedirs(eval_dir, exist_ok=True)

for i, (pts_flat, label) in enumerate(loader):
    pts_flat = pts_flat.to(device).squeeze(0)       # (num_points*dim,)
    label    = label.to(device).squeeze(0)          # (num_interior,)

    points = pts_flat.reshape(num_points, dim)
    t_grid = make_grid(points, method=method)

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
    controls = controls.cpu().numpy()


    print(f"\nSample {i}:")
    print(f"  Error: {err:.16f}")
    print(f"  True interior knots: {label.cpu().numpy()}")
    print(f"  Pred interior knots: {pred_knots[1:-1].cpu().numpy()}")

    # print final results to file
    results_path = os.path.join(eval_dir, f"eval{i}-results.txt")
    with open(results_path, "w") as f:
        f.write(f"Error: {err:.16f}\n\n")
        f.write(f"Degree: {degree}\n\n")
        f.write(f"Knots:\n{full_knots}\n\n")
        f.write(f"Controls:\n{controls}\n\n")

    # plot the final curve fit to file
    plot_curve_fit(points, full_knots, degree, controls, err, 
                    path = eval_dir, name = f"eval{i}-curve_fit.png")
