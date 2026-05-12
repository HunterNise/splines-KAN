"""
[task1/NN2] Dataset-level B-spline knot predictor — GPU-optimised training.

This file mirrors train.py but replaces all per-sample Python loops with fully batched
tensor operations to maximise GPU throughput.  The produced model file, loss curves, and
training-info text are identical in format to those from train.py, so the same eval.py
can be used without modification.

GPU optimisation changes vs train.py
--------------------------------------
1. Full dataset on GPU: BSplineDataset pre-stacks all tensors and moves them to the
   target device inside __init__; no host-to-device copies occur during training.

2. No DataLoader: the training loop uses torch.randperm + direct tensor slicing,
   eliminating DataLoader worker IPC, pin_memory copies, and per-batch Python overhead.

3. Batched basis matrix (bspline_basis_matrix_batched): evaluates the B-spline basis
   for the entire batch at once, producing a (B, N, M) tensor in a single kernel
   sequence instead of B separate (N, M) calls inside a Python for-loop.

4. Batched linear solve (solve_control_points_batched): calls torch.linalg.solve in
   batched mode — one (B, M, M) dispatch instead of B sequential (M, M) solves.

5. Safe division with torch.where: replaces the boolean-indexed scatter
   (coef[mask] = num[mask] / denom[mask]) from functions.bspline_basis_matrix with a
   branch-free elementwise select.  Boolean scatter causes non-contiguous memory
   access patterns that are expensive on GPU.

6. Pre-allocated buffers: the knot-clamping zero/one pad tensors and the regularisation
   identity matrix are created once before the training loop instead of inside every
   per-batch / per-sample iteration, avoiding repeated cudaMalloc calls.

7. No debug assertions in hot paths: torch.all() on a GPU tensor forces an implicit
   GPU→CPU synchronisation barrier.  All assertions are omitted from the two batched
   utility functions so the GPU pipeline is never stalled.
"""


import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from source.functions import *

import os
import shutil
import importlib.util

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

# Parse parameter file (identical to train.py)

prm_file = os.path.join(os.path.dirname(__file__), "train.prm")
prm      = PrmParser().parse(prm_file)

# Save copies of the parameter and model files for reproducibility
shutil.copy2(prm_file, os.path.join(output_dir, "train.prm"))
arch_file = os.path.join(os.path.dirname(__file__), "model.py")
shutil.copy2(arch_file, os.path.join(output_dir, "model.py"))

# --------------------------------------------------

path       = prm.get("Dataset", "Path")
full_path  = os.path.join(ROOT, path)
num_knots  = prm.get_int("Dataset", "Number of knots")
num_points = prm.get_int("Dataset", "Number of points")


# ==================================================
# GPU-optimised B-spline utilities (inlined; do not modify functions.py)
# ==================================================

def bspline_basis_matrix_batched(t_grid, knots_batch, degree):
    """
    Batched B-spline basis matrix via Cox–de Boor recursion.

    Replaces per-sample calls to functions.bspline_basis_matrix() inside a Python
    for-loop.  All B samples are processed in a single sequence of batched matrix
    operations, avoiding B separate kernel launches per batch.

    Changes vs functions.bspline_basis_matrix
    ------------------------------------------
    - Accepts knots_batch (B, K) instead of a single knots vector (K,), and returns
      (B, N, M) instead of (N, M), where M = K - degree - 1.
    - Safe division uses torch.where instead of boolean-indexed scatter/gather
      (coef[mask] = num[mask] / denom[mask]).  Boolean scatter causes irregular
      (gather-like) memory access which is expensive on GPU.
    - Debug assertions (which call torch.all() and force GPU→CPU sync barriers)
      are omitted to keep the GPU pipeline uninterrupted.
    - Only the hard-indicator path (soft=False) is implemented; the soft/STE path
      is not needed for the physics-loss training used here.

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
    m      = K - 1   # number of intervals = number of degree-0 basis functions

    # Broadcast layout:
    #   t       : (1, N, 1)  — parameter grid, no batch or interval dim
    #   lefts   : (B, 1, m)  — left  endpoints of each interval for each sample
    #   rights  : (B, 1, m)  — right endpoints of each interval for each sample
    t      = t_grid.view(1, N, 1)
    lefts  = knots_batch[:, :-1].unsqueeze(1)   # (B, 1, m)
    rights = knots_batch[:,  1:].unsqueeze(1)   # (B, 1, m)

    # Degree-0 basis: indicator for the half-open interval [left, right)
    mask = (t >= lefts) & (t < rights)           # (B, N, m)

    # Handle the right endpoint t == 1.0 for clamped knot vectors.
    # The last non-degenerate interval [a, 1) with a < 1 must also capture t == 1.
    # Strategy: for each sample b, find the last j where lefts[b,j] < rights[b,j]
    # (last_nondegen), then include t == rights[b,j] in the mask for that interval.
    nondegen = (knots_batch[:, :-1] < knots_batch[:, 1:])   # (B, m) bool

    # Reverse-cumsum trick: rev_cum[b, j] = number of non-degenerate intervals in [j, m).
    # last_nondegen[b, j] == True  <=>  j is the LAST j' where nondegen[b, j'] == True.
    rev_cum       = torch.flip(
                        torch.cumsum(torch.flip(nondegen.to(knots_batch.dtype), [1]), dim=1),
                    [1])                                         # (B, m)
    last_nondegen = (rev_cum == 1.0) & nondegen                  # (B, m)

    # Add the right-endpoint correction: t == right AND this is the last nondegen interval
    mask = mask | ((t == rights) & last_nondegen.unsqueeze(1))   # (B, N, m)

    Bprev = mask.to(knots_batch.dtype)   # (B, N, m) — float64 basis at degree 0

    # Cox–de Boor recursion: build up from degree 0 to `degree`.
    # At each step p the number of basis functions shrinks by 1: M = K - p - 1.
    for p in range(1, degree + 1):
        M = K - p - 1
        if M <= 0:
            return torch.zeros(B_size, N, 0, dtype=knots_batch.dtype, device=knots_batch.device)

        # Numerators — shape (B, N, M) via broadcasting (1,N,1) with (B,1,M)
        num1 = t - knots_batch[:, :M].unsqueeze(1)               # (B, N, M)
        num2 = knots_batch[:, p+1:p+1+M].unsqueeze(1) - t        # (B, N, M)

        # Denominators — shape (B, 1, M); broadcast to (B, N, M) via num1/num2
        denom1 = (knots_batch[:, p:p+M]   - knots_batch[:, :M]).unsqueeze(1)          # (B, 1, M)
        denom2 = (knots_batch[:, p+1:p+1+M] - knots_batch[:, 1:M+1]).unsqueeze(1)    # (B, 1, M)

        # Safe division: where denom == 0 the coefficient is 0 (B-spline convention).
        # Using torch.where avoids boolean-indexed scatter (non-contiguous GPU access).
        # Replace zero denominators with 1 before dividing so no NaN is produced;
        # the torch.where then selects 0.0 for those positions anyway.
        safe1 = torch.where(denom1 != 0, denom1, torch.ones_like(denom1))
        coef1 = torch.where(denom1 != 0, num1 / safe1, torch.zeros_like(num1))

        safe2 = torch.where(denom2 != 0, denom2, torch.ones_like(denom2))
        coef2 = torch.where(denom2 != 0, num2 / safe2, torch.zeros_like(num2))

        # Advance to next degree
        Bprev = coef1 * Bprev[:, :, :M] + coef2 * Bprev[:, :, 1:M+1]   # (B, N, M)

    return Bprev   # (B, N, K-degree-1)


def solve_control_points_batched(B_mat, X, eye_reg):
    """
    Batched least-squares solve for B-spline control points.

    Solves  min_C ||B @ C - X||^2 + reg * ||C||^2  for every sample in the batch
    using torch.linalg.solve in batched mode (single cuBLAS dispatch) instead of
    B sequential scalar solves.

    Changes vs functions.solve_control_points
    ------------------------------------------
    - Accepts (B, N, M) / (B, N, d) tensors instead of (N, M) / (N, d).
    - The regularised identity matrix is passed in as `eye_reg` (pre-allocated once
      outside the training loop) rather than allocated with torch.eye on every call.
    - Returns (B, M, d) instead of (M, d).
    - No debug assertions (avoids GPU→CPU sync).

    Parameters
    ----------
    B_mat   : torch.Tensor (batch, N, M)  B-spline basis matrix (batched).
    X       : torch.Tensor (batch, N, d)  Target curve points (batched).
    eye_reg : torch.Tensor (1, M, M)      Pre-allocated reg * I matrix; broadcast over batch.

    Returns
    -------
    C : torch.Tensor (batch, M, d)  Control points (batched).
    
    """
    Bt   = B_mat.transpose(1, 2)   # (batch, M, N)
    G    = Bt @ B_mat              # (batch, M, M)  — batched Gram matrix
    Greg = G + eye_reg             # (batch, M, M)  — eye_reg broadcasts over batch
    rhs  = Bt @ X                  # (batch, M, d)
    return torch.linalg.solve(Greg, rhs)   # (batch, M, d) — single batched solve


# ==================================================

class BSplineDataset:
    """
    B-spline curve dataset with the entire data preloaded onto a device.

    Changes vs the original Dataset in train.py
    ---------------------------------------------
    - Does NOT subclass torch.utils.data.Dataset: the training loop uses direct
      tensor slicing instead of a DataLoader, so the DataLoader API (__getitem__,
      __len__) is not needed.
    - All curve samples are stacked into contiguous tensors (pts, knots) and moved
      to `device` in a single .to() call inside __init__, so no host-to-device
      copies happen during the training loop.
    - t_grid is stored as a single shared (num_points,) tensor instead of being
      replicated once per sample.

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

        # Filter curves whose knot count matches num_knots (same logic as train.py)
        knot_count = np.count_nonzero(knots_np, axis=1) + (d + 1) - 2 * d
        mask       = knot_count == num_knots

        knots_np    = knots_np[mask]
        ctrl_pts_np = ctrl_pts_np[mask]

        full_knot_len = num_knots + 2 * d
        n_ctrl        = num_knots + d - 1

        # Shared parameter grid — identical for every curve (built on CPU; moved below)
        t_grid_cpu = torch.linspace(0.0, 1.0, num_points, dtype=torch.float64)

        for i in range(len(ctrl_pts_np)):
            full_knots = torch.tensor(knots_np[i, :full_knot_len], dtype=torch.float64)
            ctrls      = torch.tensor(ctrl_pts_np[i, :n_ctrl],     dtype=torch.float64)

            # Use the original (CPU) bspline_basis_matrix for dataset construction —
            # this runs once and doesn't need to be batched.
            B_mat = bspline_basis_matrix(t_grid_cpu, full_knots, d, soft=False)
            pts   = B_mat @ ctrls   # (num_points, dim)

            interior_knots = full_knots[d + 1 : -(d + 1)]   # (num_knots - 2,)

            pts_list.append(pts.reshape(-1))     # (num_points * dim,)
            knots_list.append(interior_knots)    # (num_knots - 2,)

        # Stack into contiguous tensors and transfer to device in one shot.
        # GPU optimisation: a single .to(device) for all N samples avoids N separate
        # cudaMalloc / host-to-device memcpy calls during the training loop.
        self.pts    = torch.stack(pts_list).to(device)    # (N, num_points * dim)
        self.knots  = torch.stack(knots_list).to(device)  # (N, num_knots - 2)
        self.t_grid = t_grid_cpu.to(device)               # (num_points,)

    def __len__(self):
        return self.pts.shape[0]


print("Loading dataset ...")
dataset = BSplineDataset(full_path, num_knots, num_points, device=device)
degree  = dataset.degree
dim     = dataset.dim

# Reproducible 80/20 train/test split — replicates random_split(..., seed=0) from train.py
n_train = int(0.8 * len(dataset))
n_test  = len(dataset) - n_train
perm      = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
train_idx = perm[:n_train]
test_idx  = perm[n_train:]

# Pre-slice into contiguous train/test tensors on device.
# GPU optimisation: avoid repeated fancy-index gathers inside the training loop.
train_pts   = dataset.pts[train_idx]    # (n_train, num_points * dim)
train_knots = dataset.knots[train_idx]  # (n_train, num_knots - 2)
test_pts    = dataset.pts[test_idx]     # (n_test,  num_points * dim)
test_knots  = dataset.knots[test_idx]   # (n_test,  num_knots - 2)
t_grid      = dataset.t_grid            # (num_points,) — already on device

batch_size = prm.get_int("Training", "Batch size")

# --------------------------------------------------

# Define the neural network model (architecture unchanged from train.py)

_spec = importlib.util.spec_from_file_location("model", os.path.join(os.path.dirname(__file__), "model.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
NN = _mod.NN

num_neurons = prm.get_int("Model", "Number of neurons")
dropout     = prm.get_float("Model", "Dropout")

model = NN(num_points, dim, num_knots, num_neurons, degree, dropout).to(device)

# Print architecture and parameter count to file
summary_path = os.path.join(output_dir, "summary.txt")
print("Saving model summary ...")
with open(summary_path, "w") as f:
    f.write("\nModel architecture:\n")
    f.write(str(model) + "\n\n")

    f.write(f"\nInput dimension:  {model.num_points * model.dim} = {model.num_points} * {model.dim}  (flattened data points)")
    f.write(f"\nOutput dimension: {model.num_knots - 1} (intervals without clamping)")
    f.write(f"\nB-spline degree:  {model.degree}")

    f.write("\n\nLayer shapes (weight, bias):\n")
    for name, param in model.named_parameters():
        f.write(f"  {name:<15} : {list(param.shape)}\n")

    total     = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    f.write(f"\nTotal parameters:     {total:>9,d}")
    f.write(f"\nTrainable parameters: {trainable:>9,d}")
    f.write(f"\nFrozen parameters:    {total - trainable:>9,d}")

# --------------------------------------------------

# Pre-allocate buffers that are reused every iteration.
#
# GPU optimisation: creating torch.zeros / torch.ones / torch.eye inside the training
# loop triggers a cudaMalloc (or sub-allocator call) and a kernel launch per call.
# Allocating them once and reusing via expand / broadcast is far cheaper.

# Knot-clamping pads: prepend (degree+1) zeros and append (degree+1) ones to the
# model's interior knot predictions to build the full clamped knot vector.
_zeros_pad = torch.zeros(1, degree + 1, dtype=precision, device=device)   # (1, d+1)
_ones_pad  = torch.ones( 1, degree + 1, dtype=precision, device=device)   # (1, d+1)

# Regularisation identity matrix for the batched linear solve.
# Shape (1, n_ctrl, n_ctrl) broadcasts over the batch dimension automatically.
_n_ctrl  = num_knots + degree - 1
_eye_reg = torch.eye(_n_ctrl, dtype=precision, device=device).unsqueeze(0) * 1e-6  # (1, M, M)


def build_full_knots_batched(pred_knots):
    """
    Pad a batch of predicted knot vectors to full clamped form.

    Takes the (B, num_knots) model output and wraps the interior knots
    pred_knots[:, 1:-1] with (degree+1) exact zeros and ones to produce the full
    open/clamped knot vector.  Uses pre-allocated _zeros_pad / _ones_pad to avoid
    repeated allocation inside the training loop.

    Parameters
    ----------
    pred_knots : torch.Tensor (B, num_knots)

    Returns
    -------
    full_knots : torch.Tensor (B, num_knots + 2 * degree)
    
    """
    B = pred_knots.shape[0]
    return torch.cat([
        _zeros_pad.expand(B, -1),   # (B, degree+1)
        pred_knots[:, 1:-1],        # (B, num_knots-2) — drop first/last to enforce exact 0/1
        _ones_pad.expand(B, -1),    # (B, degree+1)
    ], dim=1)


# --------------------------------------------------

def compute_epoch_loss(model, pts_tensor, knots_tensor, batch_size, beta):
    """
    Evaluate mean/max/min/std batch losses over a data split without gradient updates.

    Changes vs train.py
    --------------------
    - Uses bspline_basis_matrix_batched + solve_control_points_batched instead of a
      per-sample Python for-loop, reducing kernel launches from O(B) to O(1) per batch.
    - Data is already on device; no .to(device) calls are needed.
    - Physics loss aggregated with a single torch.sum over the (B, N, d) residual
      tensor instead of accumulated with += inside a Python loop.
    
    """
    batch_losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pts_tensor), batch_size):
            points_batch = pts_tensor[start : start + batch_size]    # (B, num_points*dim)
            labels_batch = knots_tensor[start : start + batch_size]  # (B, num_knots-2)
            B_size       = points_batch.shape[0]

            pred_knots       = model(points_batch)                   # (B, num_knots)
            full_knots_batch = build_full_knots_batched(pred_knots)  # (B, full_knot_len)

            # Single batched kernel sequence for all B samples
            B_mat    = bspline_basis_matrix_batched(t_grid, full_knots_batch, degree)  # (B, N, M)
            pts_3d   = points_batch.view(B_size, num_points, dim)                      # (B, N, d)
            controls = solve_control_points_batched(B_mat, pts_3d, _eye_reg)           # (B, M, d)

            # Physics loss: mean over batch, sum over (points, dims)
            residuals    = pts_3d - B_mat @ controls                                # (B, N, d)
            physics_loss = torch.sum(residuals ** 2) / B_size

            # Supervised loss: mean over batch and interior knots
            supervised_loss = F.mse_loss(pred_knots[:, 1:-1], labels_batch)

            batch_losses.append((physics_loss + beta * supervised_loss).item())

    model.train()
    return batch_losses


def train(model, train_pts, train_knots, test_pts, test_knots,
          num_epochs=100, tol=1e-6, lr=1e-3, beta=0.0, patience=10,
          checkpoint_file=None, checkpoint_interval=5):
    """
    Train the model using fully batched GPU-friendly operations.

    Changes vs train.py
    --------------------
    - No DataLoader: uses torch.randperm on-device + direct tensor slicing for
      batching.  All data is already resident on the device, so there is no
      per-batch host-to-device transfer.
    - Inner for-loop over batch samples replaced by bspline_basis_matrix_batched +
      solve_control_points_batched, reducing CUDA kernel launches from O(batch_size)
      to O(degree) per batch (one kernel per recursion level in the Cox–de Boor loop).
    - Physics loss aggregated with a single torch.sum over the full (B, N, d)
      residual tensor instead of accumulated with += inside a Python loop.
    - Pre-allocated _zeros_pad, _ones_pad, and _eye_reg buffers are reused across
      all epochs and batches.
    
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    train_losses = []
    test_losses  = []

    best_test_mean   = float("inf")
    best_state       = None
    patience_counter = 0
    start_epoch      = 0

    # Resume from checkpoint if one exists (unfinished previous run)
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

        # GPU optimisation: generate shuffle indices directly on device (no host round-trip)
        # and slice GPU tensors directly — no DataLoader worker process or IPC overhead.
        indices = torch.randperm(n_train, device=device)

        for i in tqdm(range(0, n_train, batch_size), ncols=100, desc=f"Epoch {epoch}"):
            idx          = indices[i : i + batch_size]
            points_batch = train_pts[idx]    # (B, num_points*dim) — already on device
            labels_batch = train_knots[idx]  # (B, num_knots-2)
            B_size       = points_batch.shape[0]

            # Forward pass
            pred_knots = model(points_batch)   # (B, num_knots)

            # Build full clamped knot vectors for the batch using pre-allocated pads
            full_knots_batch = build_full_knots_batched(pred_knots)   # (B, full_knot_len)

            # Batched B-spline basis: O(degree) kernel launches for all B samples at once,
            # versus O(B * degree) separate launches in the original per-sample loop.
            B_mat = bspline_basis_matrix_batched(t_grid, full_knots_batch, degree)  # (B, N, M)

            # Reshape flattened points to (B, N, d) for the matrix multiply
            pts_3d = points_batch.view(B_size, num_points, dim)   # (B, N, d)

            # Batched least-squares solve — single (B, M, M) dispatch to cuBLAS
            controls = solve_control_points_batched(B_mat, pts_3d, _eye_reg)   # (B, M, d)

            # Physics loss: mean over batch, sum over (points, dims) — one torch.sum call
            # versus B separate torch.sum calls accumulated in a Python loop.
            residuals    = pts_3d - B_mat @ controls     # (B, N, d)
            physics_loss = torch.sum(residuals ** 2) / B_size

            # Supervised loss: mean over batch and interior knots
            supervised_loss = F.mse_loss(pred_knots[:, 1:-1], labels_batch)

            loss = physics_loss + beta * supervised_loss
            batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Epoch train statistics
        train_losses.append((
            np.mean(batch_losses), np.max(batch_losses),
            np.min(batch_losses),  np.std(batch_losses)
        ))

        # Evaluate on test split
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

        # Early stopping: save best model and track patience
        if test_mean < best_test_mean - tol:
            best_test_mean   = test_mean
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        # Save checkpoint so training can be resumed if interrupted
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
        model.load_state_dict(best_state)   # restore best weights

    # Remove checkpoint on clean completion — the final model.pth takes over
    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return train_losses, test_losses


# --------------------------------------------------

model_file        = os.path.join(output_dir, "model.pth")
train_losses_file = os.path.join(output_dir, "train_losses.npy")
test_losses_file  = os.path.join(output_dir, "test_losses.npy")
training_file     = os.path.join(output_dir, "training_info.txt")
checkpoint_file   = os.path.join(output_dir, "checkpoint.pth")

num_epochs          = prm.get_int("Training", "Number of epochs")
lr                  = prm.get_float("Training", "Learning rate")
beta                = prm.get_float("Training", "Beta")
patience            = prm.get_int("Training", "Patience")
checkpoint_interval = prm.get_int("Training", "Checkpoint interval")

# Train the model using the GPU-optimised batched pipeline
train_losses, test_losses = train(
    model, train_pts, train_knots, test_pts, test_knots,
    num_epochs=num_epochs, tol=eps, lr=lr, beta=beta, patience=patience,
    checkpoint_file=checkpoint_file, checkpoint_interval=checkpoint_interval
)


# Save model and training artefacts (same format as train.py for eval.py compatibility)
torch.save({
    'model_state_dict'  : model.state_dict(),
    'num_points'        : num_points,
    'dim'               : dim,
    'num_knots'         : num_knots,
    'num_neurons'       : num_neurons,
    'degree'            : degree,
    'dropout'           : dropout,
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
    f.write(f"  Dropout: {dropout}\n")

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


# Plot training and test loss curves (identical to train.py)
plot_train_losses(train_losses, log=True, path=output_dir, name="train_losses.png")
plot_train_losses(test_losses,  log=True, path=output_dir, name="test_losses.png")
plot_train_and_test_losses(train_losses, test_losses, log=True,
                           path=output_dir, name="train_vs_test_losses.png")
