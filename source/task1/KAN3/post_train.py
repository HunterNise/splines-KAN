"""
[task1/KAN3] Post-training analysis for the sliding-window KAN.

Loads the trained KAN model and performs post-training analysis, including:
- Plotting the spline activations for each edge in the network.
- Extracting and printing the symbolic formula for each output dimension.

Changes vs KAN2/post_train.py
------------------------------
- Loads the additional checkpoint fields: window_size, stride, histogram_bandwidth.
- Reconstructs N_w and the soft-histogram weight matrix hist_weights from those fields.
- Defines extract_windows so the warm-up forward pass feeds the KAN its actual input
  domain — window-extracted, centroid-subtracted, globally-scaled tensors of shape
  (batch*N_w, window_size*dim) — rather than raw flattened curve points.
  This is required for model.acts / model.spline_postacts to be populated correctly
  before save_spline_plots accesses them.
- The symbolic formula extracted by print_formula is a function of window_size*dim
  normalised inputs (one per point-coordinate in the window after centroid removal
  and global scale division).  With a small window (e.g. w=5, d=2 → 10 inputs) the
  formula is compact enough to be interpretable as a local geometric feature.
"""


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kan import KAN

import numpy as np

from source.functions import *

import os
import copy

# Set seed for reproducibility
torch.manual_seed(0)

# Define global variables
PLOT    = True      # whether to plot the spline activations after training
FORMULA = True      # whether to print the symbolic formula after training

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

# Load the trained model and all hyperparameters from the output folder

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

# Reconstruct sliding-window geometry (needed for extract_windows and warm-up)

n_intervals = num_knots - 1
N_w         = (num_points - window_size) // stride + 1

# --------------------------------------------------

# Load a small subset of the dataset to drive the warm-up forward pass

path = os.path.join(ROOT, prm.get("Dataset", "Path"))
print(f"Loading dataset from {path}")
dataset = BSplineDataset(path, num_knots, num_points)

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


# Warm-up forward pass with window-extracted inputs.
# pykan requires batch >= 2 (torch.std uses Bessel's correction) and save_act=True to
# populate model.acts and model.spline_postacts before plotting or symbolic extraction.
# We temporarily re-enable save_act for the warm-up pass only, then restore it.
_warmup_batch = next(iter(DataLoader(dataset[:10], batch_size=2, shuffle=False)))
_warmup_pts   = _warmup_batch[0].to(device)   # (2, num_points*dim)
_warmup_input = extract_windows(_warmup_pts)  # (2*N_w, window_size*dim)

model.save_act = True
with torch.no_grad():
    model(_warmup_input)
model.save_act = False   # restore speed-mode state

# --------------------------------------------------

def save_spline_plots(model, folder="figures"):
    """Save per-edge activation plots without composing the full overview figure."""
    os.makedirs(folder, exist_ok=True)
    depth = len(model.width) - 1
    for l in range(depth):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l + 1]):
                symbolic_mask = model.symbolic_fun[l].mask[j][i]
                numeric_mask  = model.act_fun[l].mask[i][j]
                if symbolic_mask > 0. and numeric_mask > 0.:
                    color = "purple"
                elif symbolic_mask > 0. and numeric_mask == 0.:
                    color = "red"
                elif symbolic_mask == 0. and numeric_mask > 0.:
                    color = "black"
                else:
                    color = "white"

                rank = torch.argsort(model.acts[l][:, i])
                fig, ax = plt.subplots(figsize=(2.0, 2.0))
                ax.plot(
                    model.acts[l][:, i][rank].cpu().detach().numpy(),
                    model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(),
                    color=color, lw=2,
                )
                ax.axis("off")
                fig.savefig(os.path.join(folder, f"sp_{l}_{i}_{j}.png"),
                            bbox_inches="tight", dpi=400)
                plt.close(fig)


def print_formula(model, lib, path):
    """
    Print to file the symbolic formula for each output dimension.

    For KAN3 the formula takes window_size*dim normalised inputs, where each input
    is a coordinate of a point in the local window after centroid subtraction and
    global bounding-box scale division.  With a small window the formula is expected
    to be a compact local geometric feature (e.g. a function of chord lengths and
    turning angles).
    """
    m = copy.deepcopy(model)
    m.auto_symbolic(lib=lib)

    formulas, vars_ = m.symbolic_formula()
    with open(path, "w") as f:
        f.write(f"# KAN3 sliding-window formula\n")
        f.write(f"# Input: {window_size} normalised 2D points (window_size={window_size}, dim={dim})\n")
        f.write(f"# Output: scalar knot-density score for the window\n\n")
        for k, expr in enumerate(formulas):
            f.write(f"output[{k}] = {expr}\n")

# --------------------------------------------------

if PLOT:
    print("Plotting splines ...")
    save_spline_plots(model)
    print("Done.")
else:
    print("Skipping plotting.")


if FORMULA:
    libs = {
        "formula_full.txt":   ['x','x^2','x^3','1/x','1/x^2','1/x^3','sqrt','sin','cos','tan','tanh','exp','log','abs','sgn','0'],
        "formula_linear.txt": ['x'],
    }

    for filename, lib in libs.items():
        print(f"\nExtracting formula with library: {lib} ...")
        print_formula(model, lib, os.path.join(output_dir, filename))
else:
    print("Skipping formula extraction.")
