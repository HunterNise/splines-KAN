"""
[task1/KAN3] Dataset-level B-spline knot predictor — sliding-window KAN, GPU-optimised.

Replaces the monolithic KAN from KAN2 with a weight-shared sliding-window KAN that maps
each local window of w consecutive curve points to a scalar "knot-density score".
The global knot vector is recovered by aggregating all window scores into a soft Gaussian
histogram over the parameter domain [0, 1], then applying softmax + cumsum.

Motivation
----------
The KAN2 input dimension (num_points * dim, e.g. 60) leads to an overparameterised network
whose symbolic formula must encode *all* points simultaneously — no compact geometric meaning
exists for such a formula.  Optimal knot placement is instead a *local* problem: a knot is
needed wherever local curvature / arc-length variation is high (equidistribution principle).
A small shared KAN applied to a handful of consecutive points is forced to discover exactly
these local geometric features (turning angle, arc length, relative elongation), giving the
symbolic formula a clear interpretation.

Architecture changes vs KAN2/train_gpu.py
------------------------------------------
1. Window extraction (extract_windows):
     pts (B, num_points*dim)
       → 3-D reshape (B, num_points, dim)
       → unfold along points dim → (B, N_w, window_size, dim)
       → per-window centroid subtraction (translation invariance)
       → global-scale division by curve bounding-box size (scale invariance,
          preserves relative local arc-length between windows)
       → flatten → (B*N_w, window_size*dim)   ← KAN input

2. Shared KAN forward:
     (B*N_w, window_size*dim) → KAN → (B*N_w, 1)
       → reshape → scores (B, N_w)            ← scalar score per window

3. Soft-histogram aggregation (hist_weights, pre-computed once):
     scores (B, N_w) @ hist_weights (N_w, n_intervals) → bin_scores (B, n_intervals)
     where hist_weights[i, k] = exp(-0.5 * ((τ_i - c_k) / σ)^2),
           τ_i = i / (N_w - 1)  ∈ [0, 1]   (window position in parameter domain)
           c_k = (k + 0.5) / n_intervals     (bin centre)
           σ   = histogram_bandwidth / n_intervals

4. Knot recovery (unchanged):
     softmax(bin_scores) → pred_intervals
     cumsum([0, pred_intervals]) → pred_knots ∈ [0, 1]^{num_knots}

KAN model width: [window_size*dim, *hidden_layers, 1]
  — much smaller than KAN2's [num_points*dim, *hidden_layers, num_intervals].

New parameters (in train.prm under Model)
------------------------------------------
- Window size:         w — number of consecutive points per window.
- Stride:              step between consecutive windows (≥ 1).
- Histogram bandwidth: σ̃ — Gaussian width as a multiple of one bin width (σ = σ̃/n_intervals).

Constraint: N_w = (num_points - window_size) // stride + 1  must be ≥ n_intervals.

GPU optimisation notes (inherited from KAN2/train_gpu.py)
----------------------------------------------------------
- Full dataset pre-loaded on device; no DataLoader / host-to-device copies during training.
- torch.randperm-based shuffling directly on device.
- Batched B-spline basis (bspline_basis_matrix_batched) and batched least-squares solve
  (solve_control_points_batched) replace per-sample Python for-loops.
- Pre-allocated _zeros_pad, _ones_pad, _eye_reg buffers reused across all epochs.
- model.speed() disables pykan's visualisation overhead (save_act / symbolic_enabled).
- update_grid_from_samples() receives the already window-extracted KAN input directly
  (no separate warm-up pass needed; get_act() is called internally).

Compatibility
-------------
The saved model.pth format includes all fields needed to reconstruct the model at eval time:
  model_state_dict, num_points, dim, num_knots, width, grid_intervals, spline_order,
  degree, window_size, stride, histogram_bandwidth.
An updated eval.py for KAN3 must use extract_windows + hist_weights aggregation instead of
the direct KAN forward pass used in KAN2/eval.py.
"""


import torch
import torch.nn.functional as F

from kan import KAN

import numpy as np

from source.functions import *

import os
import shutil

from tqdm import tqdm


# Set seed for reproducibility
torch.manual_seed(0)

# --------------------------------------------------

# Create output folder if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)
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
eps = torch.finfo(precision).eps    # machine epsilon, used as early-stopping tolerance

# --------------------------------------------------

# Parse parameter file

prm_file = os.path.join(os.path.dirname(__file__), "train.prm")
prm      = PrmParser().parse(prm_file)

# Save a copy of the parameter file for reproducibility
shutil.copy2(prm_file, os.path.join(output_dir, "train.prm"))

# --------------------------------------------------

path       = prm.get("Dataset", "Path")
full_path  = os.path.join(ROOT, path)
num_knots  = prm.get_int("Dataset", "Number of knots")
num_points = prm.get_int("Dataset", "Number of points")

# Sliding-window and aggregation hyperparameters
window_size         = prm.get_int("Model", "Window size")
stride              = prm.get_int("Model", "Stride")
histogram_bandwidth = prm.get_float("Model", "Histogram bandwidth")


# ==================================================
# GPU-optimised B-spline utilities (identical to KAN2/train_gpu.py)
# ==================================================

def bspline_basis_matrix_batched(t_grid, knots_batch, degree):
    """
    Batched B-spline basis matrix via Cox-de Boor recursion.

    Parameters
    ----------
    t_grid      : torch.Tensor (N,)    Parameter values; shared across all batch samples.
    knots_batch : torch.Tensor (B, K)  Batch of full clamped knot vectors.
    degree      : int                  B-spline degree.

    Returns
    -------
    Bprev : torch.Tensor (B, N, M)  where M = K - degree - 1.
    """
    B_size = knots_batch.shape[0]
    N      = t_grid.shape[0]
    K      = knots_batch.shape[1]
    m      = K - 1

    t      = t_grid.view(1, N, 1)
    lefts  = knots_batch[:, :-1].unsqueeze(1)   # (B, 1, m)
    rights = knots_batch[:,  1:].unsqueeze(1)   # (B, 1, m)

    mask     = (t >= lefts) & (t < rights)       # (B, N, m)
    nondegen = (knots_batch[:, :-1] < knots_batch[:, 1:])   # (B, m)

    rev_cum       = torch.flip(
                        torch.cumsum(torch.flip(nondegen.to(knots_batch.dtype), [1]), dim=1),
                    [1])
    last_nondegen = (rev_cum == 1.0) & nondegen
    mask          = mask | ((t == rights) & last_nondegen.unsqueeze(1))

    Bprev = mask.to(knots_batch.dtype)   # (B, N, m)

    for p in range(1, degree + 1):
        M = K - p - 1
        if M <= 0:
            return torch.zeros(B_size, N, 0, dtype=knots_batch.dtype, device=knots_batch.device)

        num1   = t - knots_batch[:, :M].unsqueeze(1)
        num2   = knots_batch[:, p+1:p+1+M].unsqueeze(1) - t
        denom1 = (knots_batch[:, p:p+M]     - knots_batch[:, :M]).unsqueeze(1)
        denom2 = (knots_batch[:, p+1:p+1+M] - knots_batch[:, 1:M+1]).unsqueeze(1)

        safe1 = torch.where(denom1 != 0, denom1, torch.ones_like(denom1))
        coef1 = torch.where(denom1 != 0, num1 / safe1, torch.zeros_like(num1))
        safe2 = torch.where(denom2 != 0, denom2, torch.ones_like(denom2))
        coef2 = torch.where(denom2 != 0, num2 / safe2, torch.zeros_like(num2))

        Bprev = coef1 * Bprev[:, :, :M] + coef2 * Bprev[:, :, 1:M+1]

    return Bprev   # (B, N, K-degree-1)


def solve_control_points_batched(B_mat, X, eye_reg):
    """
    Batched least-squares solve for B-spline control points.

    Parameters
    ----------
    B_mat   : torch.Tensor (batch, N, M)
    X       : torch.Tensor (batch, N, d)
    eye_reg : torch.Tensor (1, M, M)   reg * I, pre-allocated; broadcast over batch.

    Returns
    -------
    C : torch.Tensor (batch, M, d)
    """
    Bt   = B_mat.transpose(1, 2)
    G    = Bt @ B_mat
    Greg = G + eye_reg
    rhs  = Bt @ X
    return torch.linalg.solve(Greg, rhs)


# ==================================================

class BSplineDataset:
    """
    B-spline curve dataset with the entire data preloaded onto a device.

    Attributes
    ----------
    pts    : torch.Tensor (N, num_points * dim)  Flattened curve points, on device.
    knots  : torch.Tensor (N, num_knots - 2)     Interior knots, on device.
    t_grid : torch.Tensor (num_points,)          Shared parameter grid, on device.
    degree : int
    dim    : int
    """

    def __init__(self, path, num_knots, num_points=100, device='cpu'):
        pts_list   = []
        knots_list = []

        with np.load(path) as data:
            knots_np    = data['knots']
            ctrl_pts_np = data['ctrl_pts']
            degree      = data['degree']

        self.degree = int(degree)
        self.dim    = ctrl_pts_np.shape[2]

        d = self.degree

        knot_count = np.count_nonzero(knots_np, axis=1) + (d + 1) - 2 * d
        mask       = knot_count == num_knots

        knots_np    = knots_np[mask]
        ctrl_pts_np = ctrl_pts_np[mask]

        full_knot_len = num_knots + 2 * d
        n_ctrl        = num_knots + d - 1

        t_grid_cpu = torch.linspace(0.0, 1.0, num_points, dtype=torch.float64)

        for i in range(len(ctrl_pts_np)):
            full_knots = torch.tensor(knots_np[i, :full_knot_len], dtype=torch.float64)
            ctrls      = torch.tensor(ctrl_pts_np[i, :n_ctrl],     dtype=torch.float64)

            B_mat = bspline_basis_matrix(t_grid_cpu, full_knots, d, soft=False)
            pts   = B_mat @ ctrls   # (num_points, dim)

            interior_knots = full_knots[d + 1 : -(d + 1)]   # (num_knots - 2,)

            pts_list.append(pts.reshape(-1))
            knots_list.append(interior_knots)

        self.pts    = torch.stack(pts_list).to(device)    # (N, num_points * dim)
        self.knots  = torch.stack(knots_list).to(device)  # (N, num_knots - 2)
        self.t_grid = t_grid_cpu.to(device)               # (num_points,)

    def __len__(self):
        return self.pts.shape[0]


print("Loading dataset ...")
dataset = BSplineDataset(full_path, num_knots, num_points, device=device)
degree  = dataset.degree
dim     = dataset.dim

# Validate sliding-window parameters against the dataset geometry.
# N_w: number of windows produced by sliding a window of size `window_size` with `stride`
# along a sequence of num_points points.
n_intervals = num_knots - 1
N_w = (num_points - window_size) // stride + 1
if N_w < n_intervals:
    raise ValueError(
        f"Sliding-window produces only N_w={N_w} windows but n_intervals={n_intervals} "
        f"bins are needed for the soft histogram.  "
        f"Reduce window_size/stride or increase num_points."
    )

# Pre-compute soft-histogram weights (static; reused every forward pass).
#
# Each window i sits at normalised position τ_i = i / (N_w - 1) ∈ [0, 1].
# The n_intervals bins have centres c_k = (k + 0.5) / n_intervals.
# The Gaussian proximity weight for window i contributing to bin k is:
#   hist_weights[i, k] = exp(-0.5 * ((τ_i - c_k) / σ)^2)
# where σ = histogram_bandwidth / n_intervals.
#
# Aggregation: bin_scores = scores @ hist_weights  — (B, N_w) × (N_w, n_intervals)
# The softmax in kan_forward_to_knots absorbs any overall scale, so the weights
# do not need to be normalised.
_tau         = torch.linspace(0.0, 1.0, N_w, dtype=precision, device=device)                           # (N_w,)
_bin_centers = (torch.arange(n_intervals, dtype=precision, device=device) + 0.5) / n_intervals         # (n_intervals,)
_sigma       = histogram_bandwidth / n_intervals
hist_weights = torch.exp(
    -0.5 * ((_tau.unsqueeze(1) - _bin_centers.unsqueeze(0)) / _sigma) ** 2
)   # (N_w, n_intervals)

# Reproducible 80/20 train/test split
n_train   = int(0.8 * len(dataset))
n_test    = len(dataset) - n_train
perm      = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
train_idx = perm[:n_train]
test_idx  = perm[n_train:]

train_pts   = dataset.pts[train_idx]    # (n_train, num_points * dim)
train_knots = dataset.knots[train_idx]  # (n_train, num_knots - 2)
test_pts    = dataset.pts[test_idx]     # (n_test,  num_points * dim)
test_knots  = dataset.knots[test_idx]   # (n_test,  num_knots - 2)
t_grid      = dataset.t_grid            # (num_points,)

batch_size = prm.get_int("Training", "Batch size")

# --------------------------------------------------

# Define the sliding-window KAN model.
#
# Architecture: [window_size*dim, *hidden_layers, 1]
#   - Input:  w consecutive normalised points flattened to (window_size * dim,).
#   - Output: scalar score (knot-density vote for this window's region).
#
# The same KAN instance (shared weights) is applied to every window of every curve,
# so it is forced to learn a translation- and scale-invariant local geometric feature
# detector.  The symbolic formula after training is a function of only window_size*dim
# normalised inputs — compact enough to be interpretable.

hidden_layers  = [int(x) for x in prm.get("Model", "Hidden layers").split(",")]
grid_intervals = prm.get_int("Model", "Grid intervals")
spline_order   = prm.get_int("Model", "Spline order")

# Full width list: input = window_size*dim, output = 1 (scalar score)
width = [window_size * dim] + hidden_layers + [1]

model = KAN(
    width     = width,
    grid      = grid_intervals,
    k         = spline_order,
    seed      = 0,
    device    = device,
    auto_save = False,
)

# Disable pykan's visualisation overhead (save_act, symbolic_enabled) for the whole run.
# update_grid_from_samples() restores save_act internally for its own pass.
model.speed()

# Print architecture and parameter count to file
summary_path = os.path.join(output_dir, "summary.txt")
print("Saving model summary ...")
with open(summary_path, "w") as f:
    f.write("\nModel architecture (shared sliding-window KAN):\n")
    f.write(str(model) + "\n\n")

    f.write(f"\nInput dimension:  {window_size * dim} = {window_size} points * {dim}D  (per window, normalised)")
    f.write(f"\nOutput dimension: 1 (scalar knot-density score per window)")
    f.write(f"\nB-spline degree:  {degree}\n")

    f.write(f"\nSliding window:    size={window_size}, stride={stride}")
    f.write(f"\nWindows per curve: N_w={N_w}  ({num_points} points, window={window_size}, stride={stride})")
    f.write(f"\nHistogram bins:    n_intervals={n_intervals},  bandwidth={histogram_bandwidth},  sigma={_sigma:.4f}\n")

    f.write(f"\nKAN width:          {width}")
    f.write(f"\nKAN grid intervals: {grid_intervals}")
    f.write(f"\nKAN spline order:   {spline_order}\n")

    total     = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    f.write(f"\nTotal parameters:     {total:>9,d}")
    f.write(f"\nTrainable parameters: {trainable:>9,d}")
    f.write(f"\nFrozen parameters:    {total - trainable:>9,d}")

# --------------------------------------------------

# Pre-allocate reusable buffers (avoid repeated cudaMalloc inside the training loop).

_zeros_pad = torch.zeros(1, degree + 1, dtype=precision, device=device)   # (1, d+1)
_ones_pad  = torch.ones( 1, degree + 1, dtype=precision, device=device)   # (1, d+1)

_n_ctrl  = num_knots + degree - 1
_eye_reg = torch.eye(_n_ctrl, dtype=precision, device=device).unsqueeze(0) * 1e-6  # (1, M, M)


def extract_windows(pts_batch):
    """
    Extract normalised sliding windows from a batch of curves.

    Steps
    -----
    1. Reshape (B, num_points*dim) → (B, num_points, dim).
    2. Unfold along the points dimension with (window_size, stride) to get
       (B, N_w, window_size, dim).
    3. Subtract the per-window centroid to achieve translation invariance.
    4. Divide by the global bounding-box scale of each curve (largest side of the
       axis-aligned bounding box of all num_points points).  This removes the overall
       size of the curve while preserving the *relative* local arc-length between
       windows — faster-moving windows remain proportionally larger than slow ones,
       which is a useful curvature proxy.
    5. Flatten and merge batch and window dimensions →
       (B * N_w, window_size * dim)   ← ready for the KAN forward pass.

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    torch.Tensor (B * N_w, window_size * dim)
    """
    B = pts_batch.shape[0]
    pts_3d = pts_batch.view(B, num_points, dim)   # (B, num_points, dim)

    # unfold(dimension, size, step) adds a trailing dimension of length `size`.
    windows = pts_3d.unfold(1, window_size, stride)   # (B, N_w, dim, window_size)
    windows = windows.permute(0, 1, 3, 2).contiguous()  # (B, N_w, window_size, dim)

    # Per-window centroid subtraction (translation invariance)
    centroid = windows.mean(dim=2, keepdim=True)   # (B, N_w, 1, dim)
    windows  = windows - centroid

    # Global scale: largest side of the curve's bounding box, one scalar per curve.
    # Using the global (not per-window) scale preserves the relative speed signal.
    bb_range     = pts_3d.max(dim=1).values - pts_3d.min(dim=1).values   # (B, dim)
    global_scale = bb_range.max(dim=1).values.clamp(min=1e-8)            # (B,)
    windows      = windows / global_scale.view(B, 1, 1, 1)

    return windows.reshape(B * N_w, window_size * dim)   # (B*N_w, window_size*dim)


def build_full_knots_batched(pred_knots):
    """
    Pad a batch of predicted knot vectors to full clamped form.

    Parameters
    ----------
    pred_knots : torch.Tensor (B, num_knots)

    Returns
    -------
    full_knots : torch.Tensor (B, num_knots + 2 * degree)
    """
    B = pred_knots.shape[0]
    return torch.cat([
        _zeros_pad.expand(B, -1),
        pred_knots[:, 1:-1],
        _ones_pad.expand(B, -1),
    ], dim=1)


def kan_forward_to_knots(model, pts_batch):
    """
    Full sliding-window forward pass: curve points → knots in [0, 1].

    Pipeline
    --------
    pts_batch (B, num_points*dim)
      → extract_windows → (B*N_w, window_size*dim)    [normalised local windows]
      → shared KAN      → (B*N_w, 1)                  [scalar score per window]
      → reshape         → scores (B, N_w)
      → @ hist_weights  → bin_scores (B, n_intervals)  [soft histogram aggregation]
      → softmax         → pred_intervals (B, n_intervals)
      → [0] ‖ cumsum   → pred_knots (B, num_knots)    [monotone, in [0, 1]]

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    pred_knots : torch.Tensor (B, num_knots)
    """
    B           = pts_batch.shape[0]
    kan_input   = extract_windows(pts_batch)              # (B*N_w, window_size*dim)
    scores_flat = model(kan_input)                        # (B*N_w, 1)
    scores      = scores_flat.view(B, N_w)                # (B, N_w)
    bin_scores  = scores @ hist_weights                   # (B, n_intervals)  soft histogram
    pred_intervals = F.softmax(bin_scores, dim=1)         # (B, n_intervals)
    zero = torch.zeros(B, 1, dtype=pred_intervals.dtype, device=pred_intervals.device)
    return torch.cumsum(torch.cat((zero, pred_intervals), dim=1), dim=1)   # (B, num_knots)


# --------------------------------------------------

def compute_epoch_loss(model, pts_tensor, knots_tensor, batch_size, beta):
    """
    Evaluate mean/max/min/std batch losses over a data split without gradient updates.

    Uses bspline_basis_matrix_batched + solve_control_points_batched.
    kan_forward_to_knots encapsulates the full sliding-window pipeline.
    """
    batch_losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pts_tensor), batch_size):
            points_batch = pts_tensor[start : start + batch_size]
            labels_batch = knots_tensor[start : start + batch_size]
            B_size       = points_batch.shape[0]

            pred_knots       = kan_forward_to_knots(model, points_batch)
            full_knots_batch = build_full_knots_batched(pred_knots)

            B_mat    = bspline_basis_matrix_batched(t_grid, full_knots_batch, degree)
            pts_3d   = points_batch.view(B_size, num_points, dim)
            controls = solve_control_points_batched(B_mat, pts_3d, _eye_reg)

            residuals    = pts_3d - B_mat @ controls
            physics_loss = torch.sum(residuals ** 2) / B_size

            supervised_loss = F.mse_loss(pred_knots[:, 1:-1], labels_batch)

            batch_losses.append((physics_loss + beta * supervised_loss).item())

    model.train()
    return batch_losses


def train(model, train_pts, train_knots, test_pts, test_knots,
          num_epochs=100, tol=1e-6, lr=1e-3, beta=0.0, patience=10, grid_update_interval=10,
          checkpoint_file=None, checkpoint_interval=5):
    """
    Train the sliding-window KAN using fully batched GPU-friendly operations.

    Changes vs KAN2/train_gpu.py
    -----------------------------
    - kan_forward_to_knots now runs extract_windows internally, so the training loop
      passes raw pts_batch (B, num_points*dim) and receives pred_knots (B, num_knots)
      exactly as before — all sliding-window logic is encapsulated.
    - Grid update: update_grid_from_samples receives the window-extracted KAN input
      extract_windows(train_pts[:batch_size]) of shape (batch_size*N_w, window_size*dim)
      instead of the raw flattened points.  No separate warm-up pass is needed.
    - All other conventions (checkpoint/resume, early stopping, ReduceLROnPlateau,
      batched basis+solve, pre-allocated buffers) are identical to KAN2/train_gpu.py.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=8)
    train_losses = []
    test_losses  = []

    best_test_mean   = float("inf")
    best_state       = None
    patience_counter = 0
    start_epoch      = 0

    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint '{checkpoint_file}' ...")
        ckpt = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        train_losses     = ckpt['train_losses']
        test_losses      = ckpt['test_losses']
        best_test_mean   = ckpt['best_test_mean']
        best_state       = ckpt['best_state']
        patience_counter = ckpt['patience_counter']
        start_epoch      = ckpt['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch} (best test mean so far: {best_test_mean:.6f})")

    n_train = len(train_pts)

    for epoch in range(start_epoch, num_epochs):
        batch_losses = []
        model.train()

        # Periodically refine the KAN spline grids from observed window activations.
        # extract_windows produces the (batch_size*N_w, window_size*dim) tensor that
        # update_grid_from_samples expects as the KAN's actual input domain.
        if grid_update_interval > 0 and epoch > 0 and epoch % grid_update_interval == 0:
            model.update_grid_from_samples(extract_windows(train_pts[:batch_size]))

        indices = torch.randperm(n_train, device=device)

        for i in tqdm(range(0, n_train, batch_size), ncols=100, desc=f"Epoch {epoch}"):
            idx          = indices[i : i + batch_size]
            points_batch = train_pts[idx]    # (B, num_points*dim)
            labels_batch = train_knots[idx]  # (B, num_knots-2)
            B_size       = points_batch.shape[0]

            # Full sliding-window forward pass → knots in [0, 1]
            pred_knots = kan_forward_to_knots(model, points_batch)   # (B, num_knots)

            # Build full clamped knot vectors
            full_knots_batch = build_full_knots_batched(pred_knots)   # (B, full_knot_len)

            # Batched B-spline basis and least-squares control-point solve
            B_mat    = bspline_basis_matrix_batched(t_grid, full_knots_batch, degree)
            pts_3d   = points_batch.view(B_size, num_points, dim)
            controls = solve_control_points_batched(B_mat, pts_3d, _eye_reg)

            residuals    = pts_3d - B_mat @ controls
            physics_loss = torch.sum(residuals ** 2) / B_size

            supervised_loss = F.mse_loss(pred_knots[:, 1:-1], labels_batch)

            loss = physics_loss + beta * supervised_loss
            batch_losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_losses.append((
            np.mean(batch_losses), np.max(batch_losses),
            np.min(batch_losses),  np.std(batch_losses)
        ))

        test_batch_losses = compute_epoch_loss(model, test_pts, test_knots, batch_size, beta)
        test_mean = np.mean(test_batch_losses)
        test_losses.append((
            test_mean, np.max(test_batch_losses),
            np.min(test_batch_losses), np.std(test_batch_losses)
        ))

        tr = train_losses[-1]
        te = test_losses[-1]
        print(f"  train  mean={tr[0]:.6f}  max={tr[1]:.6f}  min={tr[2]:.6f}  std={tr[3]:.6f}")
        print(f"  test   mean={te[0]:.6f}  max={te[1]:.6f}  min={te[2]:.6f}  std={te[3]:.6f}")
        print()

        scheduler.step(test_mean)

        if test_mean < best_test_mean - tol:
            best_test_mean   = test_mean
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        if checkpoint_file is not None and checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch'                : epoch,
                'model_state_dict'     : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'train_losses'         : train_losses,
                'test_losses'          : test_losses,
                'best_test_mean'       : best_test_mean,
                'best_state'           : best_state,
                'patience_counter'     : patience_counter,
            }, checkpoint_file)

    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return train_losses, test_losses


# --------------------------------------------------

model_file        = os.path.join(output_dir, "model.pth")
train_losses_file = os.path.join(output_dir, "train_losses.npy")
test_losses_file  = os.path.join(output_dir, "test_losses.npy")
training_file     = os.path.join(output_dir, "training_info.txt")
checkpoint_file   = os.path.join(output_dir, "checkpoint.pth")

num_epochs           = prm.get_int("Training", "Number of epochs")
lr                   = prm.get_float("Training", "Learning rate")
beta                 = prm.get_float("Training", "Beta")
patience             = prm.get_int("Training", "Patience")
grid_update_interval = prm.get_int("Training", "Grid update interval")
checkpoint_interval  = prm.get_int("Training", "Checkpoint interval")

train_losses, test_losses = train(
    model, train_pts, train_knots, test_pts, test_knots,
    num_epochs=num_epochs, tol=eps, lr=lr, beta=beta, patience=patience,
    grid_update_interval=grid_update_interval,
    checkpoint_file=checkpoint_file, checkpoint_interval=checkpoint_interval,
)


# Save model and training artefacts.
# All sliding-window parameters are stored so eval.py can reconstruct the full
# forward pipeline (extract_windows + hist_weights + KAN) without re-reading train.prm.
torch.save({
    'model_state_dict'   : model.state_dict(),
    'num_points'         : num_points,
    'dim'                : dim,
    'num_knots'          : num_knots,
    'width'              : width,
    'grid_intervals'     : grid_intervals,
    'spline_order'       : spline_order,
    'degree'             : degree,
    'window_size'        : window_size,
    'stride'             : stride,
    'histogram_bandwidth': histogram_bandwidth,
}, model_file)

np.save(train_losses_file, np.array(train_losses, dtype=np.float64))
np.save(test_losses_file,  np.array(test_losses,  dtype=np.float64))

with open(training_file, "w") as f:
    f.write("\nDataset information:\n")
    f.write(f"  Path: {path}\n")
    f.write(f"  {num_points} points {dim}D per curve, {num_knots} knots, degree {degree}\n")
    f.write(f"  Number of (total) samples:  {len(dataset):>7,d}\n")
    f.write(f"  Number of training samples: {n_train:>7,d}\n")
    f.write(f"  Number of test samples:     {n_test:>7,d}\n")

    f.write("\nModel hyperparameters:\n")
    f.write(f"  KAN width:               {width}\n")
    f.write(f"  KAN grid intervals:      {grid_intervals}\n")
    f.write(f"  KAN spline order:        {spline_order}\n")
    f.write(f"  Window size:             {window_size}\n")
    f.write(f"  Stride:                  {stride}\n")
    f.write(f"  Windows per curve (N_w): {N_w}\n")
    f.write(f"  Histogram bandwidth:     {histogram_bandwidth}  (sigma = {_sigma:.4f})\n")
    f.write(f"  Grid update interval:    {grid_update_interval}\n")

    f.write("\nTraining configuration:\n")
    f.write(f"  Batch size: {batch_size}\n")
    f.write(f"  Number of epochs: {num_epochs}\n")
    f.write(f"  Learning rate: {lr}\n")
    f.write(f"  Beta (supervised loss weight): {beta}\n")
    f.write(f"  Patience (early stopping): {patience} epochs\n")

    f.write("\nTraining information and final results:\n")
    f.write(f"  Training completed in {len(train_losses)} epochs.\n")
    f.write(f"  Final train loss: {train_losses[-1][0]:>.6f}\n")
    f.write(f"  Final test loss:  {test_losses[-1][0]:>.6f}\n")
    f.write(f"  Best test loss:   {min(test_losses, key=lambda x: x[0])[0]:>.6f}\n")


plot_train_losses(train_losses, log=True, path=output_dir, name="train_losses.png")
plot_train_losses(test_losses,  log=True, path=output_dir, name="test_losses.png")
plot_train_and_test_losses(train_losses, test_losses, log=True,
                           path=output_dir, name="train_vs_test_losses.png")
