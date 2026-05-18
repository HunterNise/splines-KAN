"""
[task1/KAN4] Post-training analysis for the sliding-window KAN.

Loads the trained KAN4 model and performs:
- Plotting the KAN overview (spline activation thumbnails, operator symbols,
  edge attributions) via plot_kan_overview().
- Extracting and printing the symbolic formula for each output dimension.

Overview plot (plot_kan_overview)
----------------------------------
- Figure width stretched by f = sqrt(max_edges / 5) to keep thumbnails readable.
- Per-layer thumbnail half-size: y1 = max(0.4 / max(N, 5), y2).
- All image insets (thumbnails, ⊕/⊗ symbols) are rendered square in display
  space; NFC width = NFC height × (figheight / figwidth) to undo x-stretching.
- Node scatter dots sized to match operator symbol diameter.
- Edge transparency: alpha = tanh(beta * attribution_score).
- Z-order: edges sorted by attribution so important thumbnails render on top.
- Spline PNGs saved at lw=4, dpi=600 for crisp curves.

Changes vs KAN3/post_train.py
------------------------------
- Loads quantile_sharpness and mult_arity from the checkpoint instead of
  histogram_bandwidth; reconstructs _tau and _alpha instead of hist_weights.
- extract_windows computes chord lengths + turning angles (rotation-invariant
  intrinsic features matching train_gpu.py).  KAN3/post_train.py incorrectly
  used centroid subtraction on raw coordinates — this is fixed in KAN4.
- The warm-up forward pass feeds window-extracted intrinsic features of shape
  (batch*N_w, n_features) to match the KAN's actual input domain.
- The formula description reflects the n_features intrinsic geometric inputs.
- Model is constructed with mult_arity to match the training-time architecture.
- plot_kan_overview() replaces pykan's model.plot() with a custom implementation
  that remains readable for wide networks.
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

# Whether to run each post-training step
PLOT    = True
FORMULA = True

# --------------------------------------------------

# Require the output folder to exist (created by train_gpu.py)
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
if not os.path.isdir(output_dir):
    raise FileNotFoundError(
        f"Output folder '{output_dir}' not found. "
        "Please run train_gpu.py first to train the model."
    )
os.chdir(output_dir)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")

precision = torch.float64
torch.set_default_dtype(precision)
eps = torch.finfo(precision).eps

# --------------------------------------------------

prm_file = os.path.join(output_dir, "train.prm")
if not os.path.exists(prm_file):
    raise FileNotFoundError(f"Parameter file '{prm_file}' not found in the output folder.")
prm = PrmParser().parse(prm_file)

# --------------------------------------------------

model_file = os.path.join(output_dir, "model.pth")
if not os.path.exists(model_file):
    print(f"Model file {model_file} not found. Please run train_gpu.py first.")
    exit(1)

print(f"Loading model from {model_file}")
ckpt = torch.load(model_file, map_location=device)

num_points         = ckpt['num_points']
dim                = ckpt['dim']
num_knots          = ckpt['num_knots']
width              = ckpt['width']
grid_intervals     = ckpt['grid_intervals']
spline_order       = ckpt['spline_order']
degree             = ckpt['degree']
window_size        = ckpt['window_size']
stride             = ckpt['stride']
quantile_sharpness = ckpt['quantile_sharpness']
mult_arity         = ckpt['mult_arity']
n_features         = ckpt['n_features']

model = KAN(
    width      = width,
    grid       = grid_intervals,
    k          = spline_order,
    mult_arity = mult_arity,
    seed       = 0,
    device     = device,
    auto_save  = False,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# --------------------------------------------------

# Reconstruct sliding-window geometry (needed for extract_windows and warm-up)
N_w = (num_points - window_size) // stride + 1

# --------------------------------------------------

# Load a small subset of the dataset to drive the warm-up forward pass

path = os.path.join(ROOT, prm.get("Dataset", "Path"))
print(f"Loading dataset from {path}")
dataset = BSplineDataset(path, num_knots, num_points)

# --------------------------------------------------

def extract_windows(pts_batch):
    """
    Extract rotation-invariant intrinsic features from sliding windows.

    Identical to train_gpu.py: chord lengths + turning angles.  The warm-up pass
    must feed the KAN the same input distribution it saw during training so that
    model.acts / model.spline_postacts are populated correctly for plotting and
    symbolic extraction.

    Parameters
    ----------
    pts_batch : torch.Tensor (B, num_points * dim)

    Returns
    -------
    torch.Tensor (B * N_w, n_features)
    """
    B      = pts_batch.shape[0]
    pts_3d = pts_batch.view(B, num_points, dim)

    bb_range     = pts_3d.max(dim=1).values - pts_3d.min(dim=1).values
    global_scale = bb_range.max(dim=1).values.clamp(min=1e-8)

    windows = pts_3d.unfold(1, window_size, stride)           # (B, N_w, dim, window_size)
    windows = windows.permute(0, 1, 3, 2).contiguous()        # (B, N_w, window_size, dim)

    chords = windows[:, :, 1:, :] - windows[:, :, :-1, :]    # (B, N_w, window_size-1, dim)
    chords = chords / global_scale.view(B, 1, 1, 1)

    chord_lengths = chords.norm(dim=-1)                        # (B, N_w, window_size-1)

    d_cur  = chords[:, :, :-1, :]
    d_next = chords[:, :,  1:, :]
    d_cur_n  = d_cur  / d_cur.norm( dim=-1, keepdim=True).clamp(min=1e-8)
    d_next_n = d_next / d_next.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cross  = d_cur_n[..., 0] * d_next_n[..., 1] - d_cur_n[..., 1] * d_next_n[..., 0]
    dot    = (d_cur_n * d_next_n).sum(dim=-1)
    angles = torch.atan2(cross, dot)                           # (B, N_w, window_size-2)

    features = torch.cat([chord_lengths, angles], dim=-1)
    return features.reshape(B * N_w, n_features)


# Warm-up forward pass with intrinsic window features.
# pykan requires batch >= 2 (torch.std uses Bessel's correction) and save_act=True to
# populate model.acts and model.spline_postacts before plotting or symbolic extraction.
# We temporarily re-enable save_act for the warm-up pass only, then restore it.
_warmup_batch = next(iter(DataLoader(dataset[:10], batch_size=2, shuffle=False)))
_warmup_pts   = _warmup_batch[0].to(device)    # (2, num_points*dim)
_warmup_input = extract_windows(_warmup_pts)   # (2*N_w, n_features)

model.save_act = True
with torch.no_grad():
    model(_warmup_input)
#model.save_act = False   # restore speed-mode state

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
                    color=color, lw=4,
                )
                ax.axis("off")
                fig.savefig(os.path.join(folder, f"sp_{l}_{i}_{j}.png"),
                            bbox_inches="tight", dpi=600)
                plt.close(fig)


def plot_kan_overview(
    model,
    folder   = "figures",
    beta     = 3,
    scale    = 0.5,
    metric   = 'backward',
    in_vars  = None,
    out_vars = None,
    title    = None,
):
    """
    KAN overview plot — adapted from pykan's MultKAN.plot().

    Layout
    ------
    - Figure width is scaled by f = sqrt(max_edges / 5) so thumbnails stay
      physically readable even for wide layers.
    - Per-layer thumbnail half-size: y1 = max(0.4 / max(N, 5), y2), where y2
      is the operator-symbol half-size.  Thumbnails and operator/mult symbols
      are all rendered as **square images in display space** (NFC width =
      NFC height × figheight/figwidth), so they remain circular regardless of
      the horizontal stretch.
    - Node scatter dots are sized to match the operator symbol diameter.
    - Lines stop 15 % of y1 short of thumbnail edges.
    - Spline PNGs saved at lw=4, dpi=600 for crisp curves.
    - Edge transparency: alpha = tanh(beta * attribution_score).
    - Z-order: edges sorted by attribution so high-importance thumbnails
      render on top.
    - Sum/mult node symbols loaded from source/imgs/*.png.

    Parameters
    ----------
    model    : KAN    Model forward-passed with save_act=True.
    folder   : str    Directory for sp_{l}_{i}_{j}.png images.
    beta     : float  Transparency coefficient: alpha = tanh(beta * score).
    scale    : float  Overall figure scale (same as pykan). Default: 0.5.
    metric   : str    'backward' | 'forward_n' | 'forward_u'.
    in_vars  : list[str] | None   Input node labels.
    out_vars : list[str] | None   Output node labels.
    title    : str | None         Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    os.makedirs(folder, exist_ok=True)

    # Operator symbol images
    _imgs = os.path.join(ROOT, 'source', 'imgs')
    _sum_img  = plt.imread(os.path.join(_imgs, 'sum_symbol.png'))
    _mult_img = plt.imread(os.path.join(_imgs, 'mult_symbol.png'))

    # ── 1. Write individual spline PNGs ──────────────────────────────────────
    save_spline_plots(model, folder=folder)

    # ── 2. Edge attribution scores → per-edge transparency ──────────────────
    if metric == 'backward':
        model.attribute()
        scores = model.edge_scores
    elif metric == 'forward_n':
        scores = model.acts_scale
    elif metric == 'forward_u':
        scores = model.edge_actscale
    else:
        raise ValueError(
            f"metric='{metric}' not recognised. "
            "Choose 'backward', 'forward_n', or 'forward_u'."
        )

    alpha = [np.tanh(beta * s.cpu().detach().numpy()) for s in scores]

    # ── 3. Layout constants ───────────────────────────────────────────────────
    width        = np.array(model.width)
    width_in     = np.array(model.width_in)
    width_out    = np.array(model.width_out)
    A            = 1
    y0           = 0.3   # data-coord height: input node → thumbnail centre
    z0           = 0.1   # data-coord height: thumbnail centre → next node
    neuron_depth = len(width)

    min_spacing = A / np.maximum(np.max(width_out), 5)
    max_neuron  = np.max(width_out)
    # y2: sum/mult node symbol half-size (global, based on widest output row)
    y2 = 0.15 / np.maximum(max_neuron, 5)

    num_weights = width_in[:-1] * width_out[1:]

    # Per-layer thumbnail half-size: natural pykan formula clamped so thumbnails
    # are never smaller than the node/operator symbols (y2).
    y1_per_layer = np.maximum(0.4 / np.maximum(num_weights, 5), y2)  # shape: (depth,)

    # Widen x proportionally to the busiest layer so thumbnails are physically
    # readable.  sqrt-scaling prevents extreme widening for very large networks.
    f = float(np.maximum(1.0, np.sqrt(float(np.max(num_weights)) / 5.0)))

    fig, ax = plt.subplots(
        figsize=(10 * scale * f, 10 * scale * (neuron_depth - 1) * (y0 + z0))
    )

    DC_to_FC  = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    _fig_h_pts = fig.get_size_inches()[1] * 72
    # Figure height/width ratio: used to keep image insets square in display space.
    # Without this correction, insets sized by data-x coords appear squished because
    # the x-axis is stretched by factor f.
    _aspect_r  = fig.get_size_inches()[1] / fig.get_size_inches()[0]

    # Operator symbol NFC half-dimensions (square in display)
    _op_h2_nfc = DC_to_NFC([0, y2])[1] - DC_to_NFC([0, 0])[1]
    _op_w2_nfc = _op_h2_nfc * _aspect_r

    # Node scatter dot: diameter = operator display height → visually matched
    node_dot_s = np.pi * (_op_h2_nfc * _fig_h_pts) ** 2

    # ── 4. Draw nodes (scatter) and connection lines ──────────────────────────
    for l in range(neuron_depth):
        n = width_in[l]
        for i in range(n):
            plt.scatter(1 / (2 * n) + i / n, l * (y0 + z0),
                        s=node_dot_s, color='black')

        # Lines: input nodes → thumbnail bottom  AND  thumbnail top → subnodes
        if l < neuron_depth - 1:
            y1    = y1_per_layer[l]
            gap   = 0.15 * y1          # gap between line tip and thumbnail edge
            n_next = width_out[l + 1]
            N = n * n_next
            for i in range(n):
                for j in range(n_next):
                    id_      = i * n_next + j
                    sym_mask = model.symbolic_fun[l].mask[j][i]
                    num_mask = model.act_fun[l].mask[i][j]
                    if sym_mask == 1. and num_mask > 0.:
                        color = 'purple'; alpha_mask = 1.
                    elif sym_mask == 1. and num_mask == 0.:
                        color = 'red';    alpha_mask = 1.
                    elif sym_mask == 0. and num_mask == 1.:
                        color = 'black';  alpha_mask = 1.
                    else:
                        color = 'white';  alpha_mask = 0.
                    plt.plot(
                        [1/(2*n)+i/n,              1/(2*N)+id_/N],
                        [l*(y0+z0),                l*(y0+z0)+y0/2-y1-gap],
                        color=color, lw=2*scale, alpha=alpha[l][j][i]*alpha_mask,
                    )
                    plt.plot(
                        [1/(2*N)+id_/N,            1/(2*n_next)+j/n_next],
                        [l*(y0+z0)+y0/2+y1+gap,   l*(y0+z0)+y0],
                        color=color, lw=2*scale, alpha=alpha[l][j][i]*alpha_mask,
                    )

        # Lines: subnodes → next-layer nodes  (pre-mult → post-mult aggregation)
        if l < neuron_depth - 1:
            n_in  = width_out[l + 1]
            n_out = width_in[l + 1]
            mult_id            = 0
            current_mult_arity = 0
            for i in range(n_in):
                if i < width[l + 1][0]:
                    j = i
                else:
                    if i == width[l + 1][0]:
                        ma = (model.mult_arity if isinstance(model.mult_arity, int)
                              else model.mult_arity[l + 1][mult_id])
                        current_mult_arity = ma
                    if current_mult_arity == 0:
                        mult_id += 1
                        ma = (model.mult_arity if isinstance(model.mult_arity, int)
                              else model.mult_arity[l + 1][mult_id])
                        current_mult_arity = ma
                    j = width[l + 1][0] + mult_id
                    current_mult_arity -= 1
                plt.plot(
                    [1/(2*n_in)+i/n_in, 1/(2*n_out)+j/n_out],
                    [l*(y0+z0)+y0,      (l+1)*(y0+z0)],
                    color='black', lw=2*scale,
                )

    plt.xlim(0, 1)
    plt.ylim(-0.1 * (y0 + z0), (neuron_depth - 1 + 0.1) * (y0 + z0))
    plt.axis('off')
    # Transparent background so thumbnail inset axes are always in front of lines
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.0)

    # ── 5. Embed spline thumbnails using DC→NFC coordinate transform ──────────
    for l in range(neuron_depth - 1):
        y1     = y1_per_layer[l]
        n      = width_in[l]
        n_next = width_out[l + 1]
        N      = n * n_next
        # Z-order: sort edges ascending by attribution so high-importance thumbnails
        # are rendered last and therefore appear on top of overlapping neighbours.
        pairs = sorted(
            [(i, j) for i in range(n) for j in range(n_next)],
            key=lambda ij: float(alpha[l][ij[1]][ij[0]])
        )
        # Thumbnail NFC half-dims: height from y1 data coords, width keeps square display
        _th_h2_nfc = DC_to_NFC([0, y1])[1] - DC_to_NFC([0, 0])[1]
        _th_w2_nfc = _th_h2_nfc * _aspect_r
        for i, j in pairs:
            id_      = i * n_next + j
            img_path = os.path.join(folder, f"sp_{l}_{i}_{j}.png")
            if not os.path.exists(img_path):
                continue
            im     = plt.imread(img_path)
            cx_nfc = DC_to_NFC([1/(2*N)+id_/N, 0])[0]
            cy_nfc = DC_to_NFC([0, l*(y0+z0)+y0/2])[1]
            left   = cx_nfc - _th_w2_nfc
            right  = cx_nfc + _th_w2_nfc
            bottom = cy_nfc - _th_h2_nfc
            up     = cy_nfc + _th_h2_nfc
            newax  = fig.add_axes([left, bottom, right - left, up - bottom])
            newax.imshow(im, alpha=alpha[l][j][i])
            newax.axis('off')

        # Sum-node symbols at the subnode row (square in display via _op_w2_nfc)
        N_sym = n_sym = width_out[l + 1]
        for j in range(n_sym):
            cx_nfc = DC_to_NFC([1/(2*N_sym)+j/N_sym, 0])[0]
            cy_nfc = DC_to_NFC([0, l*(y0+z0)+y0])[1]
            newax  = fig.add_axes([cx_nfc - _op_w2_nfc, cy_nfc - _op_h2_nfc,
                                   2*_op_w2_nfc, 2*_op_h2_nfc])
            newax.imshow(_sum_img)
            newax.axis('off')

        # Mult-node symbols at the next-layer node row
        N_mn  = width_in[l + 1]
        n_sum = width[l + 1][0]
        n_mul = width[l + 1][1]
        for j in range(n_mul):
            id_    = j + n_sum
            cx_nfc = DC_to_NFC([1/(2*N_mn)+id_/N_mn, 0])[0]
            cy_nfc = DC_to_NFC([0, (l+1)*(y0+z0)])[1]
            newax  = fig.add_axes([cx_nfc - _op_w2_nfc, cy_nfc - _op_h2_nfc,
                                   2*_op_w2_nfc, 2*_op_h2_nfc])
            newax.imshow(_mult_img)
            newax.axis('off')

    # ── 6. Optional labels and title ──────────────────────────────────────────
    main_ax = plt.gcf().get_axes()[0]
    fs = 40 * scale
    if in_vars is not None:
        n = width_in[0]
        for i, v in enumerate(in_vars[:n]):
            main_ax.text(1/(2*n)+i/n, -0.1, str(v),
                         fontsize=fs, ha='center', va='center')
    if out_vars is not None:
        n = width_in[-1]
        for i, v in enumerate(out_vars[:n]):
            main_ax.text(1/(2*n)+i/n, (y0+z0)*(neuron_depth-1)+0.15, str(v),
                         fontsize=fs, ha='center', va='center')
    if title is not None:
        main_ax.text(0.5, (y0+z0)*(neuron_depth-1)+0.3, title,
                     fontsize=fs, ha='center', va='center')

    return fig


def print_formula(model, lib, path):
    """
    Print to file the symbolic formula for each output dimension.

    The KAN4 formula takes n_features = 2*window_size - 3 intrinsic geometric inputs:
      x[0..window_size-2]  = scale-normalised chord lengths l_i  (dimensionless)
      x[window_size-1..n_features-1] = turning angles θ_i (radians, signed)
    With mult nodes, the formula may contain explicit product terms like l_i * θ_j,
    directly encoding the discrete curvature κ ≈ |θ| / (l · l').
    """
    m = copy.deepcopy(model)
    m.auto_symbolic(lib=lib)

    formulas, vars_ = m.symbolic_formula()
    with open(path, "w") as f:
        f.write(f"# KAN4 sliding-window formula\n")
        f.write(f"# Input: {n_features} intrinsic geometric features\n")
        f.write(f"#   x[0..{window_size-2}] = chord lengths l_0..l_{window_size-2} (scale-normalised)\n")
        f.write(f"#   x[{window_size-1}..{n_features-1}] = turning angles θ_0..θ_{window_size-3} (radians, signed)\n")
        f.write(f"# Output: scalar density score (high = more knots needed in this region)\n")
        f.write(f"# Architecture: width={width}, mult_arity={mult_arity}\n\n")
        for k, expr in enumerate(formulas):
            f.write(f"output[{k}] = {expr}\n")

# --------------------------------------------------

if FORMULA:
    libs = {
        "formula_full.txt":   ['x','x^2','x^3','1/x','1/x^2','1/x^3','sqrt','sin','cos','tan','tanh','exp','log','abs','sgn','0'],
        "formula_sparse.txt": ['x','x^2','x^3','1/x','1/x^2','sqrt','abs','sgn','0'],
        "formula_linear.txt": ['x'],
    }

    for filename, lib in libs.items():
        print(f"\nExtracting formula with library: {lib} ...")
        print_formula(model, lib, os.path.join(output_dir, filename))
else:
    print("Skipping formula extraction.")


# move after due to plotting shenanigans
if PLOT:
    print("Plotting splines ...")
    fig = plot_kan_overview(model, scale=0.5)
    fig.savefig("figures/model_plot.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Done.")
else:
    print("Skipping plotting.")
