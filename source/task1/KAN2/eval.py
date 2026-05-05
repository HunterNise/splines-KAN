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

from tqdm import tqdm


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

class PrmParser:
    """Simple parser for deal.II-style .prm parameter files.
    Supports subsection/end blocks, set Key = Value entries, and # comments.
    """
    def __init__(self):
        self._data  = {}
        self._stack = []

    def parse(self, path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.lower().startswith('subsection '):
                    self._stack.append(line[len('subsection '):].strip())
                elif line.lower() == 'end':
                    self._stack.pop()
                elif line.lower().startswith('set '):
                    rest        = line[4:]
                    key, _, val = rest.partition('=')
                    full_key    = tuple(self._stack + [key.strip()])
                    self._data[full_key] = val.strip()
        return self

    def get(self, *keys):
        return self._data[tuple(keys)]

    def get_int(self, *keys):
        return int(self.get(*keys))

    def get_float(self, *keys):
        return float(self.get(*keys))


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
num_hidden      = ckpt['num_hidden']
grid_intervals  = ckpt['grid_intervals']
spline_order    = ckpt['spline_order']
degree          = ckpt['degree']
num_intervals   = num_knots - 1

model = KAN(
    width     = [num_points * dim, num_hidden, num_intervals],
    grid      = grid_intervals,
    k         = spline_order,
    seed      = 0,
    device    = device,
    auto_save = False,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# --------------------------------------------------

# evaluate on selected samples from both the training and test sets
modes = ("train", "eval")
for mode in modes:

    method = "uniform"  # method to compute parameter values corresponding to data points

    # derive mode-specific dataset path from the training path stored in the parameter file
    dataset_dir = os.path.dirname(prm.get("Dataset", "Path"))
    path = os.path.join(ROOT, dataset_dir, f"2d_{mode}.npz")   # prepend ROOT since the prm stores a relative path

    class BSplineDataset(Dataset):
        def __init__(self, path, num_knots, num_points=100):
            self.samples = []
            self.num_knots = num_knots
            self.num_points = num_points

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
            
            t_grid = torch.linspace(0.0, 1.0, num_points, dtype=torch.float64)

            for i in range(len(ctrl_pts)):
                full_knots  = torch.tensor(knots[i, :full_knot_len], dtype=torch.float64)
                ctrls       = torch.tensor(ctrl_pts[i, :n_ctrl],     dtype=torch.float64)   # (n_ctrl, dim)

                # evaluate B-spline at num_points uniform parameter values
                B   = bspline_basis_matrix(t_grid, full_knots, d, soft=False)    # (num_points, n_ctrl)
                pts = B @ ctrls                                                   # (num_points, dim)

                # interior knots as labels (exclude the d+1 leading zeros and d+1 trailing ones)
                interior_knots = full_knots[d + 1 : -(d + 1)]   # (num_knots - 2,)

                self.samples.append((pts.reshape(-1), interior_knots))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]


    dataset = BSplineDataset(path, num_knots, num_points)

    loader = DataLoader(dataset[:10], batch_size=1, shuffle=False)

    # --------------------------------------------------

    # Evaluation loop: run inference, compute fitting error, save results and plots.

    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    for i, (pts_flat, label) in enumerate(loader):
        pts_flat = pts_flat.to(device).squeeze(0)       # (num_points*dim,)
        label    = label.to(device).squeeze(0)          # (num_interior,)

        points = pts_flat.reshape(num_points, dim)
        t_grid = make_grid(points, method=method)

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
