"""
Post-training analysis for KAN.

This script loads the trained KAN model and performs post-training analysis, including:
- Plotting the spline activations for each edge in the network.
- Extracting and printing the symbolic formula for each output dimension.
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


# Define global variables

PLOT    = True      # whether to plot the spline activations after training
FORMULA = True      # whether to print the symbolic formula after training

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

path = os.path.join(ROOT, prm.get("Dataset", "Path"))   # prepend ROOT since the prm stores a relative path

print(f"Loading dataset from {path}")
dataset = BSplineDataset(path, num_knots, num_points)

# --------------------------------------------------

# warm-up forward pass (batch >= 2 needed to avoid NaN from torch.std Bessel's correction)
from torch.utils.data import DataLoader as _DL
_warmup_batch = next(iter(_DL(dataset[:10], batch_size=2, shuffle=False)))
_warmup_pts = _warmup_batch[0].to(device)
with torch.no_grad():
    model(_warmup_pts)

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


import copy

def print_formula(model, lib, path):
    """Print to file the symbolic formula for each output dimension."""
    m = copy.deepcopy(model)    # create a copy of the model to avoid modifying the original one
    m.auto_symbolic(lib=lib)

    formulas, vars_ = m.symbolic_formula()
    with open(path, "w") as f:
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
