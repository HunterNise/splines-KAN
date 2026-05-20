"""
[task1/KAN5] Dataset-level B-spline knot predictor — per-point KAN with
entropy regularisation, chord-length curriculum, and optional global rebalancer.

KAN5 addresses the structural limitations of KAN3/4 through four changes:

1. Per-point (w=3) feature extraction  [replaces w=5 window]
   -------------------------------------------------------
   The minimum window that captures one turning angle has size 3.  Each interior
   sample i (for i = 0..N-3) produces exactly one 3-dimensional feature vector:
       x_i = [l_0, l_1, θ_0]
   where l_0 = ‖p_{i+1} - p_i‖ / scale  (left chord length, scale-normalised),
         l_1 = ‖p_{i+2} - p_{i+1}‖ / scale (right chord length),
         θ_0 = atan2(d_0 × d_1, d_0 · d_1) (signed turning angle at p_{i+1}).
   This is the smallest feature set that directly encodes discrete curvature:
       κ ≈ |θ_0| / (l_0 · l_1)
   A KAN with one multiplication node can represent this formula exactly in a
   single hidden layer.  The 3-input symbolic formula is directly interpretable.

   Compared to KAN3/4 (w=5, 7 features): the 3-feature space contains less
   redundancy, the spline grid covers a much smaller domain (no multi-angle mixing),
   and the KAN formula converges to the theoretically optimal equidistribution
   measure more easily.

2. Chord-length equidistribution curriculum  [new auxiliary loss]
   ---------------------------------------------------------------
   Computes a parameter-free knot estimate from the input points alone:
       k_j^{cl} = t value where cumulative chord length reaches α_j = j/(K+1)
   This estimate equidistributes arc length and is a good first approximation to
   the optimal B-spline knots.  Adding MSE(pred_knots, k^{cl}) as an auxiliary
   loss gives the model a well-motivated curriculum signal that:
     (a) does not require ground-truth labels — computable from inputs,
     (b) is always available, so it can replace beta when beta → 0.
   The weight `chord_length_lambda` decays over epochs as the physics loss takes
   over.  Set to 0.0 to disable.

3. Entropy regularisation on the density  [prevents density collapse]
   ------------------------------------------------------------------
   KAN4's main failure mode (eval6: error=0.083): the density distribution
   collapses to near-zero on all windows except a single tight feature, placing
   all K soft-argmin levels inside the resulting step function.
   Adding −λ_H(density) to the loss (maximise density entropy) directly penalises
   this configuration.  The regular-entropy term is:
       L_entropy = Σ_i density_i * log(density_i)   (negated entropy, to minimise)
   Setting lambda_entropy ≈ 0.01–0.05 is typically sufficient to eliminate the
   worst-case outliers without over-smoothing.

4. Optional global rebalancer  [Stage 2: adds global context]
   -----------------------------------------------------------
   The local window KAN (Stage 1) cannot compare distant parts of the curve.
   An optional second KAN (Stage 2) takes the K provisional interior knot
   positions from Stage 1 and outputs K delta corrections:
       δ = rebalancer_kan(k_interior^{stage1})        (B, K)
       k_interior = k_interior^{stage1} + δ_scale · tanh(δ)
   The rebalancer sees the full K-dimensional knot configuration and can learn
   to spread out overly clustered knots.  At initialisation δ ≈ 0 (KAN
   activations are near-zero), so the rebalancer is initially the identity.
   tanh bounds the delta to (−δ_scale, +δ_scale) even at extreme activations.
   Setting Rebalancer hidden nodes = 0 disables Stage 2 (default: disabled).

Architecture
------------
   Stage 1 input:  n_features = 2*window_size - 3  (= 3 for default window_size=3)
                   [l_0, l_1, θ_0] per interior sample
   Stage 1 KAN:    [n_features, [n_sum, n_mult], 1] → scalar density per sample
   Stage 2 input:  K = num_knots - 2  provisional interior knot positions (optional)
   Stage 2 KAN:    [K, rebalancer_hidden, K]        → K delta corrections (optional)
   Aggregation:    density → CDF → soft-quantile  (identical to KAN4)

New parameters vs KAN4 (train.prm Model section)
-------------------------------------------------
   Rebalancer hidden nodes:  hidden-layer width for Stage 2 (0 = disabled)
   Rebalancer delta scale:   max correction magnitude = δ_scale (default 0.2)

New parameters vs KAN4 (train.prm Training section)
-----------------------------------------------------
   Entropy lambda:              λ for density entropy regularisation (default 0.01)
   Chord length lambda:         λ_cl for chord-length auxiliary loss (default 0.1)
   Chord length lambda decay:   multiplicative decay for λ_cl per epoch (default 0.995)

Removed vs KAN4
---------------
   Window size default changed from 5 to 3.
   Histogram bandwidth: already removed in KAN4.

GPU / compatibility notes (identical to KAN2/KAN3/KAN4)
--------------------------------------------------------
   Full dataset pre-loaded on device; batched Cox-de Boor and least-squares solve;
   pre-allocated _zeros_pad / _ones_pad / _eye_reg buffers; model.speed() disables
   pykan's visualisation overhead.

   model.pth saves Stage 1 and (if enabled) Stage 2 state dicts plus all
   hyperparameters needed to reconstruct the forward pipeline at eval time.
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

# Sliding-window hyperparameters
window_size        = prm.get_int("Model", "Window size")
stride             = prm.get_int("Model", "Stride")
quantile_sharpness = prm.get_float("Model", "Quantile sharpness")

# Per-point (w=3) intrinsic features:
#   (window_size - 1) chord lengths + (window_size - 2) turning angles = 2*window_size - 3.
# For the default window_size=3: n_features = 3 → [l_0, l_1, θ_0].
n_features = 2 * window_size - 3


# ==================================================
# GPU-optimised B-spline utilities (identical to KAN2/KAN3/KAN4)
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

# Compute sliding-window count
N_w = (num_points - window_size) // stride + 1

# --------------------------------------------------

# Pre-compute tensors that are constant across all forward passes.
#
# _tau[i]   = i / (N_w - 1) ∈ [0, 1]:  normalised parameter position of window i.
# _alpha[j] = (j+1) / (K+1):  equidistribution quantile targets for K interior knots.
num_interior = num_knots - 2
_tau   = torch.linspace(0.0, 1.0, N_w,         dtype=precision, device=device)  # (N_w,)
_alpha = torch.linspace(
    1.0 / (num_interior + 1),
    float(num_interior) / (num_interior + 1),
    num_interior, dtype=precision, device=device
)   # (num_interior,)

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

# Stage 1: shared per-point sliding-window KAN.
#
# Architecture: [n_features, [hidden_sum_nodes, hidden_mult_nodes], 1]
#
# For the default window_size=3: n_features=3 = [l_0, l_1, θ_0].
# A single multiplication node can represent the discrete curvature
# κ ≈ |θ_0| / (l_0 · l_1) in one step, which is the theoretically
# optimal density weighting for B-spline knot equidistribution.

hidden_sum_nodes  = prm.get_int("Model", "Hidden sum nodes")
hidden_mult_nodes = prm.get_int("Model", "Hidden mult nodes")
mult_arity        = prm.get_int("Model", "Mult arity")
grid_intervals    = prm.get_int("Model", "Grid intervals")
spline_order      = prm.get_int("Model", "Spline order")

width = [n_features, [hidden_sum_nodes, hidden_mult_nodes], 1]

model = KAN(
    width      = width,
    grid       = grid_intervals,
    k          = spline_order,
    mult_arity = mult_arity,
    seed       = 0,
    device     = device,
    auto_save  = False,
)
model.speed()

# --------------------------------------------------

# Stage 2: optional global rebalancer KAN.
#
# Takes K provisional interior knot positions from Stage 1 and outputs K delta
# corrections.  The final knot vector is Stage 1 + delta_scale * tanh(delta),
# clipped and sorted for validity.
#
# Architecture: [K, rebalancer_hidden, K] where K = num_interior.
# Setting rebalancer_hidden_nodes = 0 disables Stage 2 entirely.

rebalancer_hidden_nodes = prm.get_int("Model", "Rebalancer hidden nodes")
rebalancer_delta_scale  = prm.get_float("Model", "Rebalancer delta scale")

if rebalancer_hidden_nodes > 0:
    rebalancer_width = [num_interior, rebalancer_hidden_nodes, num_interior]
    rebalancer_model = KAN(
        width     = rebalancer_width,
        grid      = grid_intervals,
        k         = spline_order,
        seed      = 1,       # different seed from Stage 1 to avoid symmetric init
        device    = device,
        auto_save = False,
    )
    rebalancer_model.speed()
else:
    rebalancer_width = None
    rebalancer_model = None

# --------------------------------------------------

# Print architecture and parameter count to file

summary_path = os.path.join(output_dir, "summary.txt")
print("Saving model summary ...")
with open(summary_path, "w") as f:
    f.write("\nStage 1 — per-point sliding-window KAN:\n")
    f.write(str(model) + "\n\n")

    f.write(f"\nInput dimension:  {n_features}")
    if window_size == 3:
        f.write(f" = [l_0 (left chord), l_1 (right chord), θ_0 (turning angle)]")
    else:
        f.write(f" = ({window_size}-1) chord lengths + ({window_size}-2) turning angles")
    f.write(f"\nOutput dimension: 1 (scalar density score per window)")
    f.write(f"\nB-spline degree:  {degree}\n")

    f.write(f"\nSliding window:    size={window_size}, stride={stride}")
    f.write(f"\nWindows per curve: N_w={N_w}  ({num_points} points, window={window_size}, stride={stride})\n")

    f.write(f"\nAggregation: density → CDF → soft-quantile")
    f.write(f"\n  Quantile sharpness S = {quantile_sharpness}")
    f.write(f"\n  Interior knots K     = {num_interior}")
    f.write(f"\n  Quantile targets α   = {_alpha.cpu().numpy().round(4)}\n")

    f.write(f"\nStage 1 KAN width:          {width}")
    f.write(f"\n  Sum nodes:                {hidden_sum_nodes}")
    f.write(f"\n  Multiplication nodes:     {hidden_mult_nodes}  (arity={mult_arity})")
    f.write(f"\nKAN grid intervals:         {grid_intervals}")
    f.write(f"\nKAN spline order:           {spline_order}\n")

    s1_total     = sum(p.numel() for p in model.parameters())
    s1_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write(f"\nStage 1 total parameters:     {s1_total:>9,d}")
    f.write(f"\nStage 1 trainable parameters: {s1_trainable:>9,d}")

    if rebalancer_model is not None:
        f.write(f"\n\nStage 2 — global rebalancer KAN:\n")
        f.write(str(rebalancer_model) + "\n")
        f.write(f"\nRebalancer width:             {rebalancer_width}")
        f.write(f"\nRebalancer delta scale:       {rebalancer_delta_scale}")
        s2_total     = sum(p.numel() for p in rebalancer_model.parameters())
        s2_trainable = sum(p.numel() for p in rebalancer_model.parameters() if p.requires_grad)
        f.write(f"\nStage 2 total parameters:     {s2_total:>9,d}")
        f.write(f"\nStage 2 trainable parameters: {s2_trainable:>9,d}")
        total_trainable = s1_trainable + s2_trainable
    else:
        total_trainable = s1_trainable

    f.write(f"\n\nTotal trainable parameters:   {total_trainable:>9,d}\n")

# Pre-allocate reusable buffers (avoid repeated cudaMalloc inside the training loop)

_zeros_pad = torch.zeros(1, degree + 1, dtype=precision, device=device)   # (1, d+1)
_ones_pad  = torch.ones( 1, degree + 1, dtype=precision, device=device)   # (1, d+1)

# Minimum allowed gap between adjacent interior knots.
# Prevents near-singular B-spline basis when Stage 2 pushes knots together.
_min_knot_gap = 0.02


def _enforce_min_gap(knots):
    """
    Ensure adjacent interior knots are at least _min_knot_gap apart, then clamp
    the whole vector to [1e-3, 1-1e-3].

    Assumes knots is already sorted along dim=1.  A forward sequential pass
    (each knot >= prev + gap) is sufficient when input is sorted.  Uses
    torch.maximum so the operation is differentiable.
    """
    cols = knots.unbind(dim=1)
    pushed = [cols[0]]
    for c in cols[1:]:
        pushed.append(torch.maximum(c, pushed[-1] + _min_knot_gap))
    return torch.stack(pushed, dim=1).clamp(1e-3, 1.0 - 1e-3)


_n_ctrl  = num_knots + degree - 1
_eye_reg = torch.eye(_n_ctrl, dtype=precision, device=device).unsqueeze(0) * 1e-6  # (1, M, M)

# Gaussian sigma for direct density supervision targets.
# Set to half the average spacing between interior knots so each Gaussian
# covers roughly one inter-knot interval without overlapping its neighbours.
_density_sigma = 0.5 / (num_interior + 1)

# Collect all model parameters once (used for grad clipping in training loop)
_all_model_params = list(model.parameters())
if rebalancer_model is not None:
    _all_model_params += list(rebalancer_model.parameters())


def extract_windows(pts_batch):
    """
    Extract rotation-invariant intrinsic features from sliding windows of a batch of curves.

    Each window of window_size consecutive points is converted to
    n_features = 2*window_size - 3 intrinsic geometric features:
      - (window_size - 1) chord lengths:  l_i = ‖p_{i+1} - p_i‖ / global_scale
      - (window_size - 2) turning angles: θ_i = atan2(d_i × d_{i+1}, d_i · d_{i+1})

    For the default window_size=3:
      - 2 chord lengths:  l_0, l_1
      - 1 turning angle:  θ_0 at the centre point p_{i+1}
    This gives the minimal feature set encoding discrete curvature κ ≈ |θ_0|/(l_0·l_1).

    Features are translation-, scale-, and rotation-invariant by construction.

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    torch.Tensor (B * N_w, n_features)
    """
    B      = pts_batch.shape[0]
    pts_3d = pts_batch.view(B, num_points, dim)   # (B, num_points, dim)

    # Global scale: largest bounding-box side per curve.
    bb_range     = pts_3d.max(dim=1).values - pts_3d.min(dim=1).values   # (B, dim)
    global_scale = bb_range.max(dim=1).values.clamp(min=1e-8)            # (B,)

    # Unfold along the points dimension to extract overlapping windows.
    windows = pts_3d.unfold(1, window_size, stride)          # (B, N_w, dim, window_size)
    windows = windows.permute(0, 1, 3, 2).contiguous()       # (B, N_w, window_size, dim)

    # Scale-normalised chord vectors
    chords = windows[:, :, 1:, :] - windows[:, :, :-1, :]   # (B, N_w, window_size-1, dim)
    chords = chords / global_scale.view(B, 1, 1, 1)

    # Chord lengths — (window_size - 1) per window.
    # Clamped to prevent NaN in backward: ‖x‖.grad at x=0 is x/‖x‖ = 0/0.
    chord_lengths = chords.norm(dim=-1).clamp(min=1e-8)   # (B, N_w, window_size-1)

    # Turning angles — (window_size - 2) per window.
    d_cur    = chords[:, :, :-1, :]                                 # (B, N_w, w-2, dim)
    d_next   = chords[:, :,  1:, :]                                 # (B, N_w, w-2, dim)
    d_cur_n  = d_cur  / d_cur.norm( dim=-1, keepdim=True).clamp(min=1e-8)
    d_next_n = d_next / d_next.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cross  = d_cur_n[..., 0] * d_next_n[..., 1] - d_cur_n[..., 1] * d_next_n[..., 0]
    dot    = (d_cur_n * d_next_n).sum(dim=-1)
    angles = torch.atan2(cross, dot)   # (B, N_w, window_size-2), signed, in [-π, π]

    features = torch.cat([chord_lengths, angles.abs()], dim=-1)   # (B, N_w, n_features)
    return features.reshape(B * N_w, n_features)             # (B*N_w, n_features)


def compute_chord_length_knots(pts_batch):
    """
    Chord-length equidistribution: compute K interior knot positions that
    equidistribute the cumulative arc length of each curve.

    For each curve, the arc-length CDF is:
        arc(t_i) = Σ_{j < i} ‖p_{j+1} - p_j‖  /  Σ_{j=0}^{N-2} ‖p_{j+1} - p_j‖

    Interior knot j is placed at the t-value where arc(t) = α_j = (j+1)/(K+1),
    found by linear interpolation between sample points.

    This is a parameter-free estimate that equidistributes arc length.  It is
    the natural curriculum prior when ground-truth knots are not available.
    The KAN refines it toward optimal B-spline knot placement.

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    cl_knots : torch.Tensor (B, num_interior)   in [1e-3, 1 - 1e-3]
    """
    B      = pts_batch.shape[0]
    pts_3d = pts_batch.view(B, num_points, dim)   # (B, N, dim)

    # Chord lengths between consecutive sample points
    chords = pts_3d[:, 1:, :] - pts_3d[:, :-1, :]          # (B, N-1, dim)
    cl     = chords.norm(dim=-1).clamp(min=1e-8)             # (B, N-1)

    # Cumulative arc length with zero prepended: arc[b, i] = arc to point i
    arc = torch.cat([
        torch.zeros(B, 1, dtype=precision, device=device),
        cl.cumsum(dim=1)
    ], dim=1)  # (B, N)
    arc = arc / arc[:, -1:].clamp(min=1e-8)   # normalize to [0, 1]

    # Parameter values at each sample point (shared t_grid, expanded to batch)
    t_pts = t_grid.unsqueeze(0).expand(B, -1)                # (B, N)

    # For each quantile α_j, find the interpolated t-value via searchsorted.
    # Vectorized: searchsorted on each row of arc (B, N) for each target (B, K).
    alpha_bk = _alpha.unsqueeze(0).expand(B, -1)             # (B, K)
    idx      = torch.searchsorted(arc.contiguous(), alpha_bk.contiguous())  # (B, K)
    idx      = idx.clamp(1, num_points - 1)

    # Gather arc and t-values bracketing the quantile
    arc0 = arc.gather(1,   idx - 1)   # (B, K)
    arc1 = arc.gather(1,   idx)        # (B, K)
    t0   = t_pts.gather(1, idx - 1)   # (B, K)
    t1   = t_pts.gather(1, idx)        # (B, K)

    # Linear interpolation within each interval
    denom    = (arc1 - arc0).clamp(min=1e-10)
    frac     = (alpha_bk - arc0) / denom
    cl_knots = t0 + frac * (t1 - t0)  # (B, K)

    return cl_knots.clamp(1e-3, 1.0 - 1e-3)


def build_full_knots_batched(pred_knots):
    """
    Pad a batch of predicted knot vectors to full clamped form.

    Parameters
    ----------
    pred_knots : torch.Tensor (B, num_knots)   values in [0, 1]

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


def kan_forward_to_knots(pts_batch, return_density=False):
    """
    Full forward pass: curve points → interior knot positions in [0, 1].

    Stage 1 — density → CDF → soft-quantile (identical to KAN4):
        extract_windows(pts_batch)  → (B*N_w, n_features)
        Stage 1 KAN                 → scores (B, N_w), unconstrained
        softplus(scores) + ε        → density (B, N_w), positive
        density / Σ density         → probability distribution over windows
        cumsum(density, dim=1)      → CDF (B, N_w)
        softmax(-S*|CDF-α_j|, i=1)  → weights (B, N_w, K) per quantile
        Σ_i τ_i * weights[:, :, j]  → interior_knots (B, K)

    Stage 2 — optional global rebalancer (new in KAN5):
        rebalancer_kan(interior_knots) → delta (B, K)
        interior_knots += δ_scale * tanh(delta)
        sort + clamp to [1e-3, 1-1e-3]

    Parameters
    ----------
    pts_batch      : torch.Tensor (B, num_points * dim)
    return_density : bool  If True, also return the normalised density tensor
                           (needed for entropy regularisation in the training loop).

    Returns
    -------
    pred_knots : torch.Tensor (B, num_knots)   [0, interior knots..., 1]
    density    : torch.Tensor (B, N_w)         only if return_density=True
    """
    B           = pts_batch.shape[0]
    kan_input   = extract_windows(pts_batch)              # (B*N_w, n_features)
    scores_flat = model(kan_input)                        # (B*N_w, 1)
    scores      = scores_flat.view(B, N_w)                # (B, N_w)

    # Clamp raw scores to prevent degenerate density concentration.
    # A single spike collapses all K soft-argmin levels into one location,
    # making the Gram matrix near-singular and amplifying gradients through solve.
    scores = scores.clamp(-20.0, 20.0)

    # Stage 1 — density → CDF → soft-quantile
    density = F.softplus(scores) + 1e-8                          # (B, N_w)
    density = density / density.sum(dim=1, keepdim=True)         # → prob dist

    CDF = torch.cumsum(density, dim=1)                           # (B, N_w)

    cdf_ex   = CDF.unsqueeze(2)                                  # (B, N_w, K)
    alpha_ex = _alpha.view(1, 1, num_interior)                   # (1,  1,  K)
    tau_ex   = _tau.view(1, N_w, 1)                              # (1, N_w,  1)

    weights        = F.softmax(-quantile_sharpness * (cdf_ex - alpha_ex).abs(), dim=1)
    interior_knots = (tau_ex * weights).sum(dim=1)               # (B, K)

    interior_knots, _ = interior_knots.sort(dim=1)

    # Stage 2 — optional global rebalancer
    if rebalancer_model is not None:
        delta          = rebalancer_model(interior_knots)        # (B, K)
        interior_knots = interior_knots + rebalancer_delta_scale * torch.tanh(delta)
        interior_knots, _ = interior_knots.sort(dim=1)
        interior_knots    = interior_knots.clamp(1e-3, 1.0 - 1e-3)

    zero = torch.zeros(B, 1, dtype=precision, device=device)
    one  = torch.ones( B, 1, dtype=precision, device=device)
    pred_knots = torch.cat([zero, interior_knots, one], dim=1)   # (B, num_knots)

    if return_density:
        return pred_knots, density
    return pred_knots


# --------------------------------------------------

def compute_epoch_loss(model, pts_tensor, knots_tensor, batch_size, beta):
    """
    Evaluate mean/max/min/std batch losses over a data split without gradient updates.

    Reports physics + beta * supervised loss (same metric as training) to allow
    fair comparison with KAN3/4 results.  Entropy and chord-length auxiliary terms
    are excluded here so the reported loss is consistent across models.
    """
    batch_losses = []
    model.eval()
    if rebalancer_model is not None:
        rebalancer_model.eval()

    with torch.no_grad():
        for start in range(0, len(pts_tensor), batch_size):
            points_batch = pts_tensor[start : start + batch_size]
            labels_batch = knots_tensor[start : start + batch_size]
            B_size       = points_batch.shape[0]

            pred_knots       = kan_forward_to_knots(points_batch)
            full_knots_batch = build_full_knots_batched(pred_knots)

            B_mat    = bspline_basis_matrix_batched(t_grid, full_knots_batch, degree)
            pts_3d   = points_batch.view(B_size, num_points, dim)
            controls = solve_control_points_batched(B_mat, pts_3d, _eye_reg)

            residuals    = pts_3d - B_mat @ controls
            physics_loss = torch.sum(residuals ** 2) / B_size

            supervised_loss = F.mse_loss(pred_knots[:, 1:-1], labels_batch)

            batch_losses.append((physics_loss + beta * supervised_loss).item())

    model.train()
    if rebalancer_model is not None:
        rebalancer_model.train()
    return batch_losses


def train(num_epochs=100, tol=1e-6, lr=1e-3, weight_decay=0.0,
          lr_factor=0.5, lr_patience=10,
          beta=0.0, beta_decay=1.0, beta_min=0.0,
          entropy_lambda=0.0,
          chord_length_lambda=0.0, chord_length_lambda_decay=1.0,
          density_lambda=0.0,
          patience=10,
          rebalancer_warmup=0,
          rebalancer_grad_clip=0.1,
          grid_update_interval=0, checkpoint_file=None, checkpoint_interval=5):
    """
    Train Stage 1 (and optionally Stage 2) end-to-end.

    Changes vs KAN4/train_gpu.py
    ----------------------------
    - Entropy regularisation: loss += entropy_lambda * Σ_i d_i log(d_i)
      (negative entropy added to loss → maximises density entropy → prevents collapse).
    - Chord-length auxiliary loss: loss += current_cl_lambda * MSE(knots, k^{cl})
      where k^{cl} is the parameter-free chord-length equidistribution estimate.
      current_cl_lambda decays per epoch by chord_length_lambda_decay.
    - Gradient clipping covers parameters from both Stage 1 and Stage 2.
    - Checkpoint stores current_cl_lambda for resume.
    """
    optimizer = torch.optim.Adam(
        _all_model_params, lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )
    train_losses = []
    test_losses  = []

    best_test_mean     = float("inf")
    best_state_s1      = None
    best_state_s2      = None
    patience_counter   = 0
    start_epoch        = 0
    current_beta       = beta
    current_cl_lambda  = chord_length_lambda
    _stopped_due_to_inf = False

    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint '{checkpoint_file}' ...")
        ckpt = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if rebalancer_model is not None and ckpt.get('rebalancer_state_dict') is not None:
            rebalancer_model.load_state_dict(ckpt['rebalancer_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        train_losses     = ckpt['train_losses']
        test_losses      = ckpt['test_losses']
        best_test_mean   = ckpt['best_test_mean']
        best_state_s1    = ckpt['best_state_s1']
        best_state_s2    = ckpt.get('best_state_s2')
        patience_counter = ckpt['patience_counter']
        start_epoch      = ckpt['epoch'] + 1
        current_beta     = ckpt.get('beta', beta)
        current_cl_lambda = ckpt.get('current_cl_lambda', chord_length_lambda)
        print(f"  Resumed at epoch {start_epoch} (best test mean so far: {best_test_mean:.6f})")

    n_train = len(train_pts)

    # Names of rebalancer parameters that must stay frozen after grid calibration.
    # Populated once at rebalancer_warmup epoch; the warmup gate must skip these.
    _rebalancer_frozen_param_names: set = set()

    for epoch in range(start_epoch, num_epochs):
        batch_losses = []
        model.train()

        # Freeze Stage 2 during warmup so Stage 1 must develop non-uniform density
        # on its own.  Without this, Stage 2 learns faster and compensates for a
        # flat Stage 1, removing Stage 1's gradient signal entirely.
        # Grid parameters are excluded: they are frozen after calibration and must
        # never be trained by gradient (gradient on grid collapses intervals → NaN).
        if rebalancer_model is not None:
            rebalancer_active = (epoch >= rebalancer_warmup)
            for n, p in rebalancer_model.named_parameters():
                if n not in _rebalancer_frozen_param_names:
                    p.requires_grad_(rebalancer_active)
            rebalancer_model.train()

        # Stage 1 grid calibration — once at epoch 0.
        # Calibrates spline knot grids to the actual feature distributions, which
        # would otherwise default to [-1, 1] — missing the angle range [-π, π].
        # No periodic updates: the input distribution is fixed by the dataset.
        if epoch == 0:
            print("Calibrating Stage 1 KAN grid from samples ...")
            n_calib = min(256, n_train)
            model.update_grid_from_samples(extract_windows(train_pts[:n_calib]))
            # Freeze Stage 1 grid parameters — gradient descent on grid points
            # collapses adjacent intervals to zero → NaN in spline backward.
            # update_grid_from_samples() uses .data directly, so the periodic
            # grid refresh (grid_update_interval > 0) still works when frozen.
            _n_s1_frozen = 0
            for _n, _p in model.named_parameters():
                if 'grid' in _n:
                    _p.requires_grad_(False)
                    _n_s1_frozen += 1
            if _n_s1_frozen:
                print(f"  Froze {_n_s1_frozen} Stage 1 grid param tensors.")

        # Stage 2 grid calibration — at the epoch it first becomes active.
        # Calibrated after warmup so the grid reflects Stage 1's trained outputs,
        # not the random initialisation outputs at epoch 0.
        if rebalancer_model is not None and epoch == rebalancer_warmup:
            print("Calibrating Stage 2 rebalancer grid from Stage 1 outputs ...")
            with torch.no_grad():
                _calib_knots = kan_forward_to_knots(train_pts[:min(256, n_train)])
                _calib_interior = _calib_knots[:, 1:-1]   # (n_calib, K)
            rebalancer_model.update_grid_from_samples(_calib_interior)
            # Freeze Stage 2 grid parameters for the same reason as Stage 1.
            # Record names so the warmup gate (above) never re-enables them.
            for _n, _p in rebalancer_model.named_parameters():
                if 'grid' in _n:
                    _p.requires_grad_(False)
                    _rebalancer_frozen_param_names.add(_n)
            if _rebalancer_frozen_param_names:
                print(f"  Froze {len(_rebalancer_frozen_param_names)} Stage 2 grid param tensors.")

        # Periodic grid update (optional; off by default for stability with mult nodes)
        if grid_update_interval > 0 and epoch > 0 and epoch % grid_update_interval == 0:
            model.update_grid_from_samples(extract_windows(train_pts[:batch_size]))

        indices  = torch.randperm(n_train, device=device)
        _nan_hit        = False
        _nan_grad_count = 0   # batches where ≥1 gradient tensor was non-finite

        for i in tqdm(range(0, n_train, batch_size), ncols=100, desc=f"Epoch {epoch}"):
            idx          = indices[i : i + batch_size]
            points_batch = train_pts[idx]    # (B, num_points*dim)
            labels_batch = train_knots[idx]  # (B, num_knots-2)
            B_size       = points_batch.shape[0]

            # Full forward pass with density returned for entropy regularisation
            pred_knots, density = kan_forward_to_knots(
                points_batch, return_density=True
            )  # (B, num_knots), (B, N_w)

            full_knots_batch = build_full_knots_batched(pred_knots)
            B_mat    = bspline_basis_matrix_batched(t_grid, full_knots_batch, degree)
            pts_3d   = points_batch.view(B_size, num_points, dim)
            controls = solve_control_points_batched(B_mat, pts_3d, _eye_reg)

            residuals    = pts_3d - B_mat @ controls
            physics_loss = torch.sum(residuals ** 2) / B_size

            supervised_loss = F.mse_loss(pred_knots[:, 1:-1], labels_batch)

            loss = physics_loss + current_beta * supervised_loss

            # Entropy regularisation — prevents density collapse onto one feature.
            # neg_entropy = Σ d_i log(d_i); minimising this maximises H(density).
            if entropy_lambda > 0.0:
                neg_entropy = (density * torch.log(density + 1e-12)).sum(dim=1).mean()
                loss = loss + entropy_lambda * neg_entropy

            # Chord-length auxiliary loss — curriculum from parameter-free arc-length
            # equidistribution.  Computed with no_grad since k^{cl} is a fixed target.
            if current_cl_lambda > 1e-8:
                with torch.no_grad():
                    cl_knots = compute_chord_length_knots(points_batch)   # (B, K)
                chord_loss = F.mse_loss(pred_knots[:, 1:-1], cl_knots)
                loss = loss + current_cl_lambda * chord_loss

            # Direct density supervision — bypasses the flat soft-quantile gradient
            # by giving Stage 1 an explicit per-window target distribution.
            # Target: Gaussian bumps centred at each true interior knot position,
            # normalised to sum to 1 over N_w windows.
            # Applied only during the rebalancer warmup phase (when Stage 2 is
            # frozen) so Stage 1 develops meaningful density before Stage 2 trains.
            # When rebalancer_warmup == 0 (no warmup) it is applied every epoch.
            if density_lambda > 0.0 and (rebalancer_warmup == 0 or epoch < rebalancer_warmup):
                tau_bw   = _tau.view(1, N_w, 1)           # (1, N_w, 1)
                knots_bw = labels_batch.unsqueeze(1)       # (B, 1, K)
                target_dens = torch.exp(
                    -(tau_bw - knots_bw) ** 2 / (2 * _density_sigma ** 2)
                ).sum(dim=2)                               # (B, N_w)
                target_dens = target_dens / target_dens.sum(dim=1, keepdim=True).clamp(min=1e-8)
                loss = loss + density_lambda * F.mse_loss(density, target_dens)

            # On the first non-finite batch: print diagnostics and abort immediately.
            # All subsequent batches would also be NaN (corrupted parameters), so
            # there is no value in continuing the epoch.
            if not torch.isfinite(loss):
                with torch.no_grad():
                    _gaps = pred_knots[:, 1:] - pred_knots[:, :-1]
                    print(
                        f"  WARNING: non-finite loss at epoch {epoch}, "
                        f"batch {i // batch_size}/{n_train // batch_size} — aborting."
                    )
                    print(f"    physics_loss    = {physics_loss.item():.6g}"
                          f"  (finite: {physics_loss.isfinite().item()})")
                    print(f"    supervised_loss = {supervised_loss.item():.6g}"
                          f"  (finite: {supervised_loss.isfinite().item()})")
                    print(f"    density   finite={density.isfinite().all().item()}"
                          f"  min={density.min().item():.4g}  mean={density.mean().item():.4g}  max={density.max().item():.4g}")
                    _k = pred_knots[:, 1:-1]  # interior only
                    print(f"    interior_knots finite={_k.isfinite().all().item()}"
                          f"  min_gap={_gaps.min().item():.4f}  max_gap={_gaps.max().item():.4f}")
                    print(f"    interior_knots[0]: {_k[0].tolist()}")
                    if B_size > 1:
                        print(f"    interior_knots[1]: {_k[1].tolist()}")
                    print(f"    B_mat     finite={B_mat.isfinite().all().item()}"
                          f"  cond[0]={torch.linalg.cond(B_mat[0]).item():.4g}")
                    print(f"    controls  finite={controls.isfinite().all().item()}"
                          f"  max_abs={controls.abs().max().item():.4g}")
                    print(f"    controls[0]: {controls[0].tolist()}")
                    if rebalancer_model is not None:
                        reb_norms = [(n, p.norm().item())
                                     for n, p in rebalancer_model.named_parameters()
                                     if p.requires_grad]
                        print("    Stage2 param norms: "
                              + "  ".join(f"{n}={v:.3g}" for n, v in reb_norms))
                optimizer.zero_grad()
                _nan_hit = True
                break  # exit inner batch loop immediately

            batch_losses.append(loss.item())

            loss.backward()
            # Abort immediately on the first non-finite gradient.
            # Continuing would corrupt weights (NaN loss path) or silently stop
            # learning (skip path).  On abort, dump every intermediate value so
            # the root cause can be pinpointed exactly.
            _s1_nan = any(p.grad is not None and not torch.isfinite(p.grad).all()
                          for p in model.parameters())
            _s2_nan = rebalancer_model is not None and any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in rebalancer_model.parameters())
            if _s1_nan or _s2_nan:
                with torch.no_grad():
                    _gaps   = pred_knots[:, 1:] - pred_knots[:, :-1]
                    _mingap = _gaps.min(dim=1).values            # (B,)
                    _worst  = int(_mingap.argmin().item())
                    _n_worst = min(5, B_size)
                    _worst5 = _mingap.topk(_n_worst, largest=False).indices

                    # Recompute all intermediates for the failing batch.
                    _ki  = extract_windows(points_batch)
                    _sc  = model(_ki).view(B_size, N_w).clamp(-20.0, 20.0)
                    _den = F.softplus(_sc) + 1e-8
                    _den = _den / _den.sum(dim=1, keepdim=True)
                    _cdf = torch.cumsum(_den, dim=1)
                    _wts = F.softmax(
                        -quantile_sharpness * (
                            _cdf.unsqueeze(2) - _alpha.view(1, 1, num_interior)
                        ).abs(), dim=1)
                    _s1k = (_tau.view(1, N_w, 1) * _wts).sum(dim=1).sort(dim=1).values

                    print(
                        f"\n  NaN GRADIENT at epoch {epoch}, "
                        f"batch {i // batch_size}/{n_train // batch_size} — aborting."
                    )
                    print(f"    NaN in: {'Stage1 ' if _s1_nan else ''}{'Stage2' if _s2_nan else ''}")
                    print(f"    loss={loss.item():.6g}  phys={physics_loss.item():.6g}"
                          f"  sup={supervised_loss.item():.6g}  (forward was finite)")
                    print(f"    scores:  min={_sc.min():.4g}  max={_sc.max():.4g}"
                          f"  mean={_sc.mean():.4g}")
                    print(f"    density: min={_den.min():.4g}  max={_den.max():.4g}"
                          f"  mean={_den.mean():.4g}")
                    print(f"    s1_knots[worst={_worst}]: {_s1k[_worst].tolist()}")
                    if rebalancer_model is not None:
                        _dl  = rebalancer_model(_s1k)
                        _s2k = (
                            _s1k + rebalancer_delta_scale * torch.tanh(_dl)
                        ).sort(dim=1).values.clamp(1e-3, 1.0 - 1e-3)
                        print(f"    delta_raw[worst]:  {_dl[_worst].tolist()}")
                        print(f"    s2_knots[worst]:   {_s2k[_worst].tolist()}")
                    print(f"    pred_knots[worst]: {pred_knots[_worst].tolist()}")
                    print(f"    knot gaps[worst]:  {_gaps[_worst].tolist()}")
                    print(f"    min_gap: min={_mingap.min():.6f}"
                          f"  mean={_mingap.mean():.6f}  (worst curve={_worst})")
                    _conds = [torch.linalg.cond(B_mat[int(j)]).item() for j in _worst5]
                    print(f"    B_mat cond (worst-gap curves): "
                          + "  ".join(f"{c:.4g}" for c in _conds))
                    print(f"    controls[worst] max_abs={controls[_worst].abs().max():.4g}")
                    if rebalancer_model is not None:
                        _rnorms = [(n, p.norm().item())
                                   for n, p in rebalancer_model.named_parameters()]
                        print("    Stage2 param norms: "
                              + "  ".join(f"{n}={v:.3g}" for n, v in _rnorms))
                optimizer.zero_grad()
                _nan_hit = True
                break
            # Separate Stage 2 clip: only needed when warmup > 0 and Stage 2 has
            # just been unfrozen.  Without a dedicated cap, Stage 1 dominates the
            # combined norm and Stage 2 can take unchecked post-warmup steps.
            # When warmup=0 (joint training from epoch 0), the combined 0.5 clip
            # below is sufficient — a separate cap would starve Stage 2 of its
            # natural gradient share and cause training divergence.
            if rebalancer_model is not None and rebalancer_warmup > 0 and epoch >= rebalancer_warmup:
                torch.nn.utils.clip_grad_norm_(
                    rebalancer_model.parameters(), max_norm=rebalancer_grad_clip
                )
            torch.nn.utils.clip_grad_norm_(_all_model_params, max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

        if _nan_hit or not batch_losses:
            if not _nan_hit:
                print(f"  WARNING: all batches in epoch {epoch} produced non-finite loss.")
            _stopped_due_to_inf = True
            break  # exit outer epoch loop
        train_losses.append((
            np.mean(batch_losses), np.max(batch_losses),
            np.min(batch_losses),  np.std(batch_losses)
        ))

        test_batch_losses = compute_epoch_loss(model, test_pts, test_knots, batch_size, current_beta)
        test_mean = np.mean(test_batch_losses)
        test_losses.append((
            test_mean, np.max(test_batch_losses),
            np.min(test_batch_losses), np.std(test_batch_losses)
        ))

        tr = train_losses[-1]
        te = test_losses[-1]
        print(f"  train  mean={tr[0]:.6f}  max={tr[1]:.6f}  min={tr[2]:.6f}  std={tr[3]:.6f}")
        print(f"  test   mean={te[0]:.6f}  max={te[1]:.6f}  min={te[2]:.6f}  std={te[3]:.6f}")
        print(f"  beta={current_beta:.6f}  cl_lambda={current_cl_lambda:.6f}")
        if rebalancer_model is not None:
            with torch.no_grad():
                _s2_pnorm = sum(p.norm().item() ** 2
                                for p in rebalancer_model.parameters()) ** 0.5
            print(f"  s2_param_norm={_s2_pnorm:.4f}  nan_grad_batches={_nan_grad_count}/{n_train // batch_size}")
        print()

        # Decay beta and chord-length lambda
        if epoch >= rebalancer_warmup:
            current_beta = max(beta_min, current_beta * beta_decay)
        current_cl_lambda = current_cl_lambda * chord_length_lambda_decay

        scheduler.step(test_mean)

        if test_mean < best_test_mean - tol:
            best_test_mean   = test_mean
            best_state_s1    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_state_s2    = (
                {k: v.cpu().clone() for k, v in rebalancer_model.state_dict().items()}
                if rebalancer_model is not None else None
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        if (checkpoint_file is not None
                and checkpoint_interval > 0
                and (epoch + 1) % checkpoint_interval == 0):
            torch.save({
                'epoch'                : epoch,
                'model_state_dict'     : model.state_dict(),
                'rebalancer_state_dict': (rebalancer_model.state_dict()
                                          if rebalancer_model is not None else None),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'train_losses'         : train_losses,
                'test_losses'          : test_losses,
                'best_test_mean'       : best_test_mean,
                'best_state_s1'        : best_state_s1,
                'best_state_s2'        : best_state_s2,
                'patience_counter'     : patience_counter,
                'beta'                 : current_beta,
                'current_cl_lambda'    : current_cl_lambda,
            }, checkpoint_file)

    if best_state_s1 is not None:
        model.load_state_dict(best_state_s1)
    if rebalancer_model is not None and best_state_s2 is not None:
        rebalancer_model.load_state_dict(best_state_s2)

    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        if _stopped_due_to_inf:
            print(f"  Keeping checkpoint '{checkpoint_file}' (stopped due to non-finite loss).")
        else:
            os.remove(checkpoint_file)

    return train_losses, test_losses


# --------------------------------------------------

model_file        = os.path.join(output_dir, "model.pth")
train_losses_file = os.path.join(output_dir, "train_losses.npy")
test_losses_file  = os.path.join(output_dir, "test_losses.npy")
training_file     = os.path.join(output_dir, "training_info.txt")
checkpoint_file   = os.path.join(output_dir, "checkpoint.pth")

num_epochs                 = prm.get_int("Training", "Number of epochs")
lr                         = prm.get_float("Training", "Learning rate")
lr_factor                  = prm.get_float("Training", "LR scheduler factor")
lr_patience                = prm.get_int("Training", "LR scheduler patience")
weight_decay               = prm.get_float("Training", "Weight decay")
beta                       = prm.get_float("Training", "Beta")
beta_decay                 = prm.get_float("Training", "Beta decay")
beta_min                   = prm.get_float("Training", "Beta min")
entropy_lambda             = prm.get_float("Training", "Entropy lambda")
chord_length_lambda        = prm.get_float("Training", "Chord length lambda")
chord_length_lambda_decay  = prm.get_float("Training", "Chord length lambda decay")
density_lambda             = prm.get_float("Training", "Density lambda")
patience                   = prm.get_int("Training", "Patience")
rebalancer_warmup          = prm.get_int("Training", "Rebalancer warmup epochs")
rebalancer_grad_clip       = prm.get_float("Training", "Rebalancer grad clip")
grid_update_interval       = prm.get_int("Training", "Grid update interval")
checkpoint_interval        = prm.get_int("Training", "Checkpoint interval")

train_losses, test_losses = train(
    num_epochs=num_epochs, tol=eps, lr=lr, weight_decay=weight_decay,
    lr_factor=lr_factor, lr_patience=lr_patience,
    beta=beta, beta_decay=beta_decay, beta_min=beta_min,
    entropy_lambda=entropy_lambda,
    chord_length_lambda=chord_length_lambda,
    chord_length_lambda_decay=chord_length_lambda_decay,
    density_lambda=density_lambda,
    patience=patience, rebalancer_warmup=rebalancer_warmup,
    rebalancer_grad_clip=rebalancer_grad_clip,
    grid_update_interval=grid_update_interval,
    checkpoint_file=checkpoint_file, checkpoint_interval=checkpoint_interval,
)

# --------------------------------------------------
# Save model and all hyperparameters needed to reconstruct the forward pipeline.

torch.save({
    'model_state_dict'        : model.state_dict(),
    'rebalancer_state_dict'   : (rebalancer_model.state_dict()
                                 if rebalancer_model is not None else None),
    'num_points'              : num_points,
    'dim'                     : dim,
    'num_knots'               : num_knots,
    'width'                   : width,
    'rebalancer_width'        : rebalancer_width,
    'grid_intervals'          : grid_intervals,
    'spline_order'            : spline_order,
    'degree'                  : degree,
    'window_size'             : window_size,
    'stride'                  : stride,
    'quantile_sharpness'      : quantile_sharpness,
    'mult_arity'              : mult_arity,
    'n_features'              : n_features,
    'rebalancer_hidden_nodes' : rebalancer_hidden_nodes,
    'rebalancer_delta_scale'  : rebalancer_delta_scale,
    'entropy_lambda'          : entropy_lambda,
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
    f.write(f"  Stage 1 KAN width:          {width}\n")
    f.write(f"  Sum nodes:                  {hidden_sum_nodes}\n")
    f.write(f"  Multiplication nodes:       {hidden_mult_nodes}  (arity={mult_arity})\n")
    f.write(f"  KAN grid intervals:         {grid_intervals}\n")
    f.write(f"  KAN spline order:           {spline_order}\n")
    f.write(f"  Window size:                {window_size}\n")
    f.write(f"  Stride:                     {stride}\n")
    f.write(f"  Windows per curve (N_w):    {N_w}\n")
    f.write(f"  Intrinsic features:         {n_features}")
    if window_size == 3:
        f.write(f" = [l_0, l_1, θ_0]  (two chord lengths + one turning angle)\n")
    else:
        f.write(f" = ({window_size}-1) chord lengths + ({window_size}-2) turning angles\n")
    f.write(f"  Quantile sharpness:         {quantile_sharpness}\n")
    f.write(f"  Quantile targets α:         {_alpha.cpu().numpy().round(4)}\n")
    f.write(f"  Rebalancer hidden nodes:    {rebalancer_hidden_nodes}\n")
    if rebalancer_model is not None:
        f.write(f"  Rebalancer width:           {rebalancer_width}\n")
        f.write(f"  Rebalancer delta scale:     {rebalancer_delta_scale}\n")
        f.write(f"  Rebalancer warmup epochs:   {rebalancer_warmup}\n")
    f.write(f"  Grid update interval:       {grid_update_interval}\n")

    f.write("\nTraining configuration:\n")
    f.write(f"  Batch size:                 {batch_size}\n")
    f.write(f"  Number of epochs:           {num_epochs}\n")
    f.write(f"  Learning rate:              {lr}\n")
    f.write(f"  LR scheduler:               ReduceLROnPlateau(factor={lr_factor}, patience={lr_patience})\n")
    f.write(f"  Weight decay:               {weight_decay}\n")
    f.write(f"  Beta (initial):             {beta}\n")
    f.write(f"  Beta decay (per epoch):     {beta_decay}\n")
    f.write(f"  Beta min (floor):           {beta_min}\n")
    f.write(f"  Entropy lambda:             {entropy_lambda}\n")
    f.write(f"  Chord length lambda:        {chord_length_lambda}\n")
    f.write(f"  Chord length lambda decay:  {chord_length_lambda_decay}\n")
    f.write(f"  Density lambda:             {density_lambda}\n")
    f.write(f"  Density sigma (auto):       {_density_sigma:.6f}\n")
    f.write(f"  Patience:                   {patience} epochs\n")

    f.write("\nTraining results:\n")
    f.write(f"  Completed in {len(train_losses)} epochs.\n")
    f.write(f"  Final train loss: {train_losses[-1][0]:>.6f}\n")
    f.write(f"  Final test loss:  {test_losses[-1][0]:>.6f}\n")
    f.write(f"  Best test loss:   {min(test_losses, key=lambda x: x[0])[0]:>.6f}\n")


plot_train_losses(train_losses, log=True, path=output_dir, name="train_losses.png")
plot_train_losses(test_losses,  log=True, path=output_dir, name="test_losses.png")
