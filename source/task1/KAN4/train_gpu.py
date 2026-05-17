"""
[task1/KAN4] Dataset-level B-spline knot predictor — sliding-window KAN with
density → CDF → soft-quantile aggregation and optional multiplication nodes.

KAN4 fixes the three root causes of KAN3's systematic underperformance:

1. Inverted aggregation semantics and softmax saturation (KAN3 root cause #1/#2)
   ---------------------------------------------------------------------------
   KAN3 mapped per-window scores to knot positions via:
       bin_scores = scores @ hist_weights        (Gaussian blur to n_intervals bins)
       pred_intervals = softmax(bin_scores)      ← HIGH score → large interval → FEWER knots
       pred_knots = cumsum([0, pred_intervals])  ← counterintuitive, hard to learn

   The softmax Jacobian at initialisation (all scores ≈ 0) is at most 0.2×0.8 = 0.16,
   further attenuated by the Gaussian averaging; gradients are weak from epoch 1.

   KAN4 replaces this with a density → CDF → soft-quantile pipeline:
       density   = softplus(scores) + 1e-8     HIGH score = high density = MORE knots ✓
       density  /= density.sum(...)            → probability distribution over windows
       CDF       = cumsum(density, dim=1)      → monotone map over [0, 1]
       weights   = softmax(-S * |CDF - α_j|)  → soft attention near quantile level α_j
       knot_j    = (τ · weights).sum(dim=1)   → weighted centroid of window positions

   Advantages:
     - Correct semantics: high-curvature windows get high density → small intervals → knots
     - softplus gradient never saturates
     - Window ordering encodes position in the CDF; no τ_i input to KAN needed
     - Quantile targets α_j = j/(K+1) implement the equidistribution principle directly

2. Multiplication nodes (KAN3 root cause: additive KAN cannot express products)
   ---------------------------------------------------------------------------
   The theoretically optimal density is discrete curvature:
       κ_i ≈ |θ_i| / (l_i · l_{i+1})
   which is a ratio of turning angle to chord-length product.  A pure-addition KAN
   [n, h, 1] can approximate this through iterated spline composition, but slowly and
   with uninterpretable formulas.

   KAN4 uses width = [n_features, [n_sum, n_mult], 1] where n_mult multiplication nodes
   (default arity 2) directly express pairwise products f(x_a) * g(x_b).  The network
   can learn |θ| and 1/l as separate spline activations and multiply them in one step.

3. Oversized hidden layer (KAN3 root cause: width=16 for a near-linear problem)
   ---------------------------------------------------------------------------
   The density proxy is a simple 7-feature function.  KAN3's [7, 16, 1] = 128 edges
   overfits the spline grids and produces uninterpretable formulas.  KAN4 defaults to
   [7, [4, 2], 1] — 4 addition + 2 multiplication nodes (≈60 edges), which is ample
   for the curvature formula and produces cleaner symbolic outputs.

Architecture
------------
   Input:  n_features = 2*window_size - 3
             (window_size-1) chord lengths + (window_size-2) turning angles
             (translation-, scale-, rotation-invariant; identical extraction to KAN3)
   Hidden: [hidden_sum_nodes, hidden_mult_nodes] mixed addition/multiplication layer
   Output: 1 scalar density score per window

   Soft-quantile pipeline (pre-computed _tau and _alpha, per-forward _density, _CDF):
     _tau[i]  = i / (N_w - 1)           window positions in [0, 1]         (N_w,)
     _alpha[j] = (j+1) / (K+1)          quantile targets for K interior knots (K,)
     density  = softplus(scores) + 1e-8                                    (B, N_w)
     density /= density.sum(dim=1, keepdim=True)
     CDF      = cumsum(density, dim=1)                                      (B, N_w)
     weights  = softmax(-sharpness * |CDF[:,i] - _alpha[j]|, over i)       (B, N_w, K)
     knots    = (_tau * weights).sum(dim=1)                                 (B, K)

New parameters (train.prm Model section)
-----------------------------------------
   Hidden sum nodes:   addition nodes in hidden layer (default 4)
   Hidden mult nodes:  multiplication nodes in hidden layer (default 2)
   Mult arity:         inputs per multiplication node (default 2)
   Quantile sharpness: S in the soft-argmin (default 20, range 10-30)

New parameters (train.prm Training section)
--------------------------------------------
   Beta min:  floor for decayed beta; beta = max(beta_min, beta * beta_decay)
              prevents pure-physics collapse seen in KAN3 experiments

Removed vs KAN3
---------------
   Histogram bandwidth, hist_weights, n_intervals, _bin_centers, _sigma.
   The constraint N_w >= n_intervals is no longer required.
   "Hidden layers" replaced by "Hidden sum nodes" + "Hidden mult nodes".

GPU / compatibility notes (identical to KAN2/KAN3)
---------------------------------------------------
   Full dataset pre-loaded on device; batched Cox-de Boor and least-squares solve;
   pre-allocated _zeros_pad / _ones_pad / _eye_reg buffers; model.speed() disables
   pykan's visualisation overhead.

   model.pth saves quantile_sharpness and mult_arity instead of histogram_bandwidth.
   eval.py and post_train.py reconstruct _tau and _alpha instead of hist_weights.
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
# Quantile sharpness S: controls how peaked the soft-argmin weights are around each
# quantile level α_j.  Higher values → sharper knot placement, smaller gradients once
# the CDF is steep.  Lower values → smoother gradients, less precise knot placement.
quantile_sharpness = prm.get_float("Model", "Quantile sharpness")

# Number of rotation-invariant intrinsic features per window (unchanged from KAN3):
#   (window_size - 1) chord lengths + (window_size - 2) turning angles = 2*window_size - 3
n_features = 2 * window_size - 3


# ==================================================
# GPU-optimised B-spline utilities (identical to KAN2/KAN3)
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
# _tau[i]   = i / (N_w - 1)  ∈ [0, 1]:  normalised position of window i in the parameter
#             domain.  Used as the τ coordinates in the soft-quantile weighted centroid.
#
# _alpha[j] = (j+1) / (K+1), j = 0..K-1:  quantile target levels for K = num_knots - 2
#             interior knots.  Spacing them at 1/(K+1), 2/(K+1), ..., K/(K+1) implements
#             the equidistribution principle: each knot captures an equal share of the
#             total probability mass (= curvature weight).
#             Pre-computed once and broadcast in kan_forward_to_knots.
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

# Define the sliding-window KAN model.
#
# Architecture: [n_features, [hidden_sum_nodes, hidden_mult_nodes], 1]
#
# The same KAN instance (shared weights) is applied to every window of every curve.
# It maps n_features = 2*window_size - 3 rotation-invariant geometric features
# (chord lengths + turning angles) to a scalar density score.
#
# The [n_sum, n_mult] hidden layer mixes:
#   - n_sum  addition nodes   (standard KAN sum over spline activations)
#   - n_mult multiplication nodes  (pairwise products of spline activations)
# Multiplication nodes can represent the discrete curvature κ ≈ |θ|/(l·l') directly,
# which would require at least two layers in a pure-addition KAN.

hidden_sum_nodes  = prm.get_int("Model", "Hidden sum nodes")
hidden_mult_nodes = prm.get_int("Model", "Hidden mult nodes")
mult_arity        = prm.get_int("Model", "Mult arity")
grid_intervals    = prm.get_int("Model", "Grid intervals")
spline_order      = prm.get_int("Model", "Spline order")

# pykan width format with multiplication nodes:
#   [n_features, [n_sum, n_mult], 1]
# where [n_sum, n_mult] specifies addition and multiplication nodes in the hidden layer.
# If hidden_mult_nodes == 0 this reduces to the standard [n_features, n_sum, 1] form.
width = [n_features, [hidden_sum_nodes, hidden_mult_nodes], 1]

model = KAN(
    width      = width,
    grid       = grid_intervals,
    k          = spline_order,
    mult_arity = mult_arity,   # arity of each multiplication node (default 2 = pairwise)
    seed       = 0,
    device     = device,
    auto_save  = False,
)

# Disable pykan's visualisation overhead (save_act, symbolic_enabled) for the whole run.
# update_grid_from_samples() restores save_act internally for its own pass.
model.speed()

# Print architecture and parameter count to file
summary_path = os.path.join(output_dir, "summary.txt")
print("Saving model summary ...")
with open(summary_path, "w") as f:
    f.write("\nModel architecture (shared sliding-window KAN with mult nodes):\n")
    f.write(str(model) + "\n\n")

    f.write(f"\nInput dimension:  {n_features} = ({window_size}-1) chord lengths + ({window_size}-2) turning angles  (per window, intrinsic)")
    f.write(f"\nOutput dimension: 1 (scalar density score per window)")
    f.write(f"\nB-spline degree:  {degree}\n")

    f.write(f"\nSliding window:    size={window_size}, stride={stride}")
    f.write(f"\nWindows per curve: N_w={N_w}  ({num_points} points, window={window_size}, stride={stride})\n")

    f.write(f"\nAggregation: density -> CDF -> soft-quantile")
    f.write(f"\n  Quantile sharpness S = {quantile_sharpness}")
    f.write(f"\n  Interior knots K     = {num_interior}")
    f.write(f"\n  Quantile targets     = {_alpha.cpu().numpy().round(4)}\n")

    f.write(f"\nKAN width (pykan format): {width}")
    f.write(f"\n  Addition nodes:       {hidden_sum_nodes}")
    f.write(f"\n  Multiplication nodes: {hidden_mult_nodes}  (arity={mult_arity})")
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
    Extract rotation-invariant intrinsic features from sliding windows of a batch of curves.

    Each window of window_size consecutive points is converted to
    n_features = 2*window_size - 3 intrinsic geometric features:
      - (window_size - 1) chord lengths:  l_i = ||p_{i+1} - p_i|| / global_scale
      - (window_size - 2) turning angles: θ_i = atan2(d_i × d_{i+1}, d_i · d_{i+1})
    where d_i = (p_{i+1} - p_i) / global_scale is the i-th scale-normalised chord vector.

    These features are:
      - Translation invariant  (chord vectors are differences of points)
      - Scale invariant        (chord lengths divided by global bounding-box size;
                                turning angles are dimensionless)
      - Rotation invariant     (chord lengths unchanged by rotation; turning angles
                                are relative, depending only on the angle between chords)

    High turning angles and short chord lengths are indicators of high curvature — the
    exact quantities that drive optimal knot placement via the equidistribution principle.

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    torch.Tensor (B * N_w, n_features)   where n_features = 2 * window_size - 3
    """
    B      = pts_batch.shape[0]
    pts_3d = pts_batch.view(B, num_points, dim)   # (B, num_points, dim)

    # Global scale: largest side of the curve's bounding box, one scalar per curve.
    # Using the global (not per-window) scale preserves the relative arc-length signal:
    # a fast-moving window remains proportionally larger than a slow one.
    bb_range     = pts_3d.max(dim=1).values - pts_3d.min(dim=1).values   # (B, dim)
    global_scale = bb_range.max(dim=1).values.clamp(min=1e-8)            # (B,)

    # Unfold along the points dimension to extract overlapping windows.
    # unfold(dim, size, step): (B, num_points, dim) → (B, N_w, dim, window_size)
    # Permute to (B, N_w, window_size, dim) for convenient per-window indexing.
    windows = pts_3d.unfold(1, window_size, stride)          # (B, N_w, dim, window_size)
    windows = windows.permute(0, 1, 3, 2).contiguous()       # (B, N_w, window_size, dim)

    # Scale-normalised chord vectors: d_i = (p_{i+1} - p_i) / global_scale
    chords = windows[:, :, 1:, :] - windows[:, :, :-1, :]   # (B, N_w, window_size-1, dim)
    chords = chords / global_scale.view(B, 1, 1, 1)

    # Chord lengths — (window_size - 1) per window.
    # Proportional to local arc-length; a useful scale-invariant curvature proxy.
    # Clamp to a small positive floor: the gradient of norm(x) at x=0 is x/||x|| = 0/0 = NaN,
    # which occurs whenever two consecutively sampled curve points are coincident.
    chord_lengths = chords.norm(dim=-1).clamp(min=1e-8)   # (B, N_w, window_size-1)

    # Turning angles between consecutive chord pairs — (window_size - 2) per window.
    # Normalise chord directions before atan2 for numerical stability near zero-length
    # chords (coincident sampled points).
    d_cur  = chords[:, :, :-1, :]   # (B, N_w, window_size-2, dim)
    d_next = chords[:, :,  1:, :]   # (B, N_w, window_size-2, dim)

    d_cur_n  = d_cur  / d_cur.norm( dim=-1, keepdim=True).clamp(min=1e-8)
    d_next_n = d_next / d_next.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # 2D signed cross product: d_i × d_{i+1} = dx_i*dy_{i+1} - dy_i*dx_{i+1}
    cross  = d_cur_n[..., 0] * d_next_n[..., 1] - d_cur_n[..., 1] * d_next_n[..., 0]
    dot    = (d_cur_n * d_next_n).sum(dim=-1)
    angles = torch.atan2(cross, dot)   # (B, N_w, window_size-2), signed, in [-π, π]

    # Concatenate: [chord_lengths | turning_angles] → (B, N_w, n_features)
    features = torch.cat([chord_lengths, angles], dim=-1)   # (B, N_w, 2*window_size-3)

    return features.reshape(B * N_w, n_features)   # (B*N_w, n_features)


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


def kan_forward_to_knots(model, pts_batch):
    """
    Full sliding-window forward pass: curve points → interior knot positions in [0, 1].

    Density → CDF → soft-quantile pipeline (replaces KAN3's histogram + softmax + cumsum)
    --------------------------------------------------------------------------------------
    Step 1 — KAN density:
        extract_windows(pts_batch) → (B*N_w, n_features)
        model(·)                   → scores (B, N_w)     raw, unconstrained
        softplus(scores) + ε       → density (B, N_w)    positive; HIGH = knot needed here
        density / density.sum()    → probability distribution over window positions

    Step 2 — CDF:
        cumsum(density, dim=1)     → CDF (B, N_w)        monotone, approaches 1 at N_w

    Step 3 — Soft-quantile extraction:
        For each of K = num_knots - 2 interior knots, we target quantile level
            α_j = (j+1) / (K+1),  j = 0..K-1
        The interior knot τ_j is extracted as a soft weighted centroid:
            weights[b, i, j] = softmax_i ( -S * |CDF[b, i] - α_j| )
            knot_j[b]        = Σ_i  τ_i * weights[b, i, j]
        where S = quantile_sharpness.  Larger S → the knot is pulled toward the unique
        window where CDF ≈ α_j (hard argmin limit); smaller S → more diffuse but
        smoother gradients.  Range 10–30 works well.

    Gradient properties:
        - softplus: gradient never saturates (unlike the softmax in KAN3)
        - The CDF is a smooth differentiable function of the scores
        - The soft-argmin is differentiable with respect to CDF values

    Parameters
    ----------
    model     : KAN    Shared sliding-window KAN.
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    pred_knots : torch.Tensor (B, num_knots)   [0, interior knots..., 1]
    """
    B           = pts_batch.shape[0]
    kan_input   = extract_windows(pts_batch)              # (B*N_w, n_features)
    scores_flat = model(kan_input)                        # (B*N_w, 1)
    scores      = scores_flat.view(B, N_w)                # (B, N_w)

    # Clamp raw scores to prevent extreme density concentration on one or two windows.
    # A transient spike (e.g., immediately after a grid update with mult nodes) can place
    # all K interior knots at the same τ, making the Gram matrix near-singular and
    # amplifying gradients through linalg.solve by up to 1/reg = 1e6.
    scores = scores.clamp(-20.0, 20.0)

    # --- Step 1: density distribution ---
    # softplus ensures positivity with non-vanishing gradients everywhere.
    # The small ε avoids zero density (which would make the CDF piecewise-constant
    # and produce a disconnected gradient).
    density = F.softplus(scores) + 1e-8                          # (B, N_w)
    density = density / density.sum(dim=1, keepdim=True)         # → prob distribution

    # --- Step 2: CDF ---
    # CDF[b, i] ≈ fraction of total density in windows 0..i.
    # Because windows are ordered by parameter position (_tau is increasing),
    # the CDF is a differentiable monotone map over [0, 1].
    CDF = torch.cumsum(density, dim=1)                           # (B, N_w)

    # --- Step 3: soft-quantile extraction ---
    # Broadcast shapes: (B, N_w, 1) vs (1, 1, K) vs (1, N_w, 1)
    cdf_ex   = CDF.unsqueeze(2)                                  # (B, N_w, K)
    alpha_ex = _alpha.view(1, 1, num_interior)                   # (1,  1,  K)
    tau_ex   = _tau.view(1, N_w, 1)                              # (1, N_w,  1)

    # Attention: for each knot j, concentrate weight on windows where CDF ≈ alpha_j
    weights       = F.softmax(-quantile_sharpness * (cdf_ex - alpha_ex).abs(), dim=1)
    interior_knots = (tau_ex * weights).sum(dim=1)               # (B, K)

    # Sort: quantile targets are ordered (α_0 < … < α_{K-1}) so the output is usually
    # sorted, but a steep or bimodal CDF can cause crossings.  An explicit sort is free
    # and guarantees a monotone knot vector.
    interior_knots, _ = interior_knots.sort(dim=1)

    # Clamp away from the domain boundaries so the leading interval [0, k_1] and the
    # trailing interval [k_K, 1] always have positive width.  Zero-width boundary
    # intervals produce basis functions with no support, degrading the Gram matrix.
    interior_knots = interior_knots.clamp(1e-3, 1.0 - 1e-3)

    # Prepend 0 and append 1 to obtain the full [0, k_1, ..., k_K, 1] knot vector
    zero = torch.zeros(B, 1, dtype=precision, device=device)
    one  = torch.ones( B, 1, dtype=precision, device=device)
    return torch.cat([zero, interior_knots, one], dim=1)         # (B, num_knots)


# --------------------------------------------------

def compute_epoch_loss(model, pts_tensor, knots_tensor, batch_size, beta):
    """
    Evaluate mean/max/min/std batch losses over a data split without gradient updates.

    Uses bspline_basis_matrix_batched + solve_control_points_batched.
    kan_forward_to_knots encapsulates the full density→CDF→soft-quantile pipeline.
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
          num_epochs=100, tol=1e-6, lr=1e-3, weight_decay=0.0,
          lr_factor=0.5, lr_patience=10,
          beta=0.0, beta_decay=1.0, beta_min=0.0, patience=10,
          grid_update_interval=10, checkpoint_file=None, checkpoint_interval=5):
    """
    Train the sliding-window KAN using fully batched GPU-friendly operations.

    Changes vs KAN3/train_gpu.py
    -----------------------------
    - kan_forward_to_knots uses the new density→CDF→soft-quantile pipeline; the
      training loop passes raw pts_batch and receives pred_knots exactly as before.
    - beta_min: current_beta is clamped to max(beta_min, current_beta * beta_decay)
      at the end of each epoch.  This prevents supervision from vanishing entirely —
      a pathology seen in KAN3 experiments where beta decayed toward 0.001 and the
      model collapsed to a degenerate physics minimum with all knots near zero.
    - All other conventions (checkpoint/resume, early stopping, ReduceLROnPlateau,
      batched basis+solve, pre-allocated buffers, grad clipping) are identical to
      KAN2/KAN3.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience)
    train_losses = []
    test_losses  = []

    best_test_mean   = float("inf")
    best_state       = None
    patience_counter = 0
    start_epoch      = 0
    current_beta     = beta   # decayed each epoch; saved in / restored from checkpoints

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
        current_beta     = ckpt.get('beta', beta)
        print(f"  Resumed at epoch {start_epoch} (best test mean so far: {best_test_mean:.6f})")

    n_train = len(train_pts)

    for epoch in range(start_epoch, num_epochs):
        batch_losses = []
        model.train()

        # Periodically refine the KAN spline grids from observed window activations.
        # extract_windows produces the (batch_size*N_w, n_features) tensor that
        # update_grid_from_samples expects as the KAN's actual input domain.
        if grid_update_interval > 0 and epoch > 0 and epoch % grid_update_interval == 0:
            model.update_grid_from_samples(extract_windows(train_pts[:batch_size]))

        indices = torch.randperm(n_train, device=device)

        for i in tqdm(range(0, n_train, batch_size), ncols=100, desc=f"Epoch {epoch}"):
            idx          = indices[i : i + batch_size]
            points_batch = train_pts[idx]    # (B, num_points*dim)
            labels_batch = train_knots[idx]  # (B, num_knots-2)
            B_size       = points_batch.shape[0]

            # Density → CDF → soft-quantile forward pass → knots in [0, 1]
            pred_knots = kan_forward_to_knots(model, points_batch)   # (B, num_knots)

            # Build full clamped knot vectors: [0^(d+1), interior..., 1^(d+1)]
            full_knots_batch = build_full_knots_batched(pred_knots)   # (B, full_knot_len)

            # Batched B-spline basis and least-squares control-point solve
            B_mat    = bspline_basis_matrix_batched(t_grid, full_knots_batch, degree)
            pts_3d   = points_batch.view(B_size, num_points, dim)
            controls = solve_control_points_batched(B_mat, pts_3d, _eye_reg)

            residuals    = pts_3d - B_mat @ controls
            physics_loss = torch.sum(residuals ** 2) / B_size

            supervised_loss = F.mse_loss(pred_knots[:, 1:-1], labels_batch)

            loss = physics_loss + current_beta * supervised_loss

            # Skip non-finite batches rather than propagating NaN into the model
            # parameters — a single NaN update via Adam corrupts every future batch.
            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss at epoch {epoch}. Skipping batch.")
                optimizer.zero_grad()
                continue

            batch_losses.append(loss.item())

            loss.backward()
            # Tighter clip (0.5 vs 1.0): mult nodes amplify gradients by the value of
            # the partner activation, so effective gradient magnitudes are larger than
            # in a pure-sum KAN.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

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
        print(f"  beta={current_beta:.6f}")
        print()

        # Decay beta then apply floor so supervised signal never vanishes completely.
        # max(beta_min, ...) prevents the degenerate physics-only minimum seen in KAN3.
        current_beta = max(beta_min, current_beta * beta_decay)

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
                'beta'                 : current_beta,
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
lr_factor            = prm.get_float("Training", "LR scheduler factor")
lr_patience          = prm.get_int("Training", "LR scheduler patience")
weight_decay         = prm.get_float("Training", "Weight decay")
beta                 = prm.get_float("Training", "Beta")
beta_decay           = prm.get_float("Training", "Beta decay")
beta_min             = prm.get_float("Training", "Beta min")
patience             = prm.get_int("Training", "Patience")
grid_update_interval = prm.get_int("Training", "Grid update interval")
checkpoint_interval  = prm.get_int("Training", "Checkpoint interval")

train_losses, test_losses = train(
    model, train_pts, train_knots, test_pts, test_knots,
    num_epochs=num_epochs, tol=eps, lr=lr, weight_decay=weight_decay,
    lr_factor=lr_factor, lr_patience=lr_patience,
    beta=beta, beta_decay=beta_decay, beta_min=beta_min,
    patience=patience, grid_update_interval=grid_update_interval,
    checkpoint_file=checkpoint_file, checkpoint_interval=checkpoint_interval,
)


# Save model and all hyperparameters needed to reconstruct the forward pipeline at
# eval/post-train time without re-reading train.prm.
# quantile_sharpness and mult_arity replace histogram_bandwidth from KAN3.
torch.save({
    'model_state_dict'    : model.state_dict(),
    'num_points'          : num_points,
    'dim'                 : dim,
    'num_knots'           : num_knots,
    'width'               : width,
    'grid_intervals'      : grid_intervals,
    'spline_order'        : spline_order,
    'degree'              : degree,
    'window_size'         : window_size,
    'stride'              : stride,
    'quantile_sharpness'  : quantile_sharpness,
    'mult_arity'          : mult_arity,
    'n_features'          : n_features,
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
    f.write(f"  KAN width (pykan format):   {width}\n")
    f.write(f"  Addition nodes:             {hidden_sum_nodes}\n")
    f.write(f"  Multiplication nodes:       {hidden_mult_nodes}  (arity={mult_arity})\n")
    f.write(f"  KAN grid intervals:         {grid_intervals}\n")
    f.write(f"  KAN spline order:           {spline_order}\n")
    f.write(f"  Window size:                {window_size}\n")
    f.write(f"  Stride:                     {stride}\n")
    f.write(f"  Windows per curve (N_w):    {N_w}\n")
    f.write(f"  Intrinsic features:         {n_features} = ({window_size}-1) chord lengths + ({window_size}-2) turning angles\n")
    f.write(f"  Quantile sharpness:         {quantile_sharpness}\n")
    f.write(f"  Quantile targets alpha:     {_alpha.cpu().numpy().round(4)}\n")
    f.write(f"  Grid update interval:       {grid_update_interval}\n")

    f.write("\nTraining configuration:\n")
    f.write(f"  Batch size:             {batch_size}\n")
    f.write(f"  Number of epochs:       {num_epochs}\n")
    f.write(f"  Learning rate:          {lr}\n")
    f.write(f"  LR scheduler:           ReduceLROnPlateau(factor={lr_factor}, patience={lr_patience})\n")
    f.write(f"  Weight decay:           {weight_decay}\n")
    f.write(f"  Beta (initial):         {beta}\n")
    f.write(f"  Beta decay (per epoch): {beta_decay}\n")
    f.write(f"  Beta min (floor):       {beta_min}\n")
    f.write(f"  Patience:               {patience} epochs\n")

    f.write("\nTraining results:\n")
    f.write(f"  Completed in {len(train_losses)} epochs.\n")
    f.write(f"  Final train loss: {train_losses[-1][0]:>.6f}\n")
    f.write(f"  Final test loss:  {test_losses[-1][0]:>.6f}\n")
    f.write(f"  Best test loss:   {min(test_losses, key=lambda x: x[0])[0]:>.6f}\n")


plot_train_losses(train_losses, log=True, path=output_dir, name="train_losses.png")
plot_train_losses(test_losses,  log=True, path=output_dir, name="test_losses.png")
plot_train_and_test_losses(train_losses, test_losses, log=True,
                           path=output_dir, name="train_vs_test_losses.png")
