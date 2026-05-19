"""
[task1/KAN5] Post-training analysis for the per-point KAN.

Loads the trained KAN5 model and performs:
- KAN overview plot (spline activation thumbnails, operator symbols,
  edge attributions) for Stage 1 via plot_kan_overview().
- Symbolic formula extraction for Stage 1 in multiple libraries.

The Stage 1 KAN has only 3 inputs for the default window_size=3:
    x[0] = l_0   (left chord length, scale-normalised)
    x[1] = l_1   (right chord length, scale-normalised)
    x[2] = θ_0   (signed turning angle, radians)

With one multiplication node, the formula should converge toward the discrete
curvature measure  κ ≈ |θ_0| / (l_0 · l_1)  or a power thereof — the
theoretically optimal equidistribution weight for B-spline knots.

Changes vs KAN4/post_train.py
------------------------------
- In-variable labels updated: ['l_0', 'l_1', 'theta_0'] (3 features instead of 7).
- Formula description reflects the minimal 3D input.
- Loads and reports the Stage 2 rebalancer if present (no symbolic extraction
  for Stage 2; its 4-input → 4-output structure is not symbolic-regression-friendly).
- Warm-up pass feeds the actual per-point window features matching training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kan import KAN

import numpy as np

from source.functions import *

import os
import copy

torch.manual_seed(0)

PLOT    = True
FORMULA = True

# --------------------------------------------------

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

num_points              = ckpt['num_points']
dim                     = ckpt['dim']
num_knots               = ckpt['num_knots']
width                   = ckpt['width']
rebalancer_width        = ckpt['rebalancer_width']
grid_intervals          = ckpt['grid_intervals']
spline_order            = ckpt['spline_order']
degree                  = ckpt['degree']
window_size             = ckpt['window_size']
stride                  = ckpt['stride']
quantile_sharpness      = ckpt['quantile_sharpness']
mult_arity              = ckpt['mult_arity']
n_features              = ckpt['n_features']
rebalancer_hidden_nodes = ckpt['rebalancer_hidden_nodes']
rebalancer_delta_scale  = ckpt['rebalancer_delta_scale']

# Reconstruct Stage 1 KAN
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

# Reconstruct Stage 2 rebalancer (if present)
if rebalancer_hidden_nodes > 0 and ckpt.get('rebalancer_state_dict') is not None:
    rebalancer_model = KAN(
        width     = rebalancer_width,
        grid      = grid_intervals,
        k         = spline_order,
        seed      = 1,
        device    = device,
        auto_save = False,
    )
    rebalancer_model.load_state_dict(ckpt['rebalancer_state_dict'])
    rebalancer_model.eval()
    print(f"Stage 2 rebalancer loaded: width={rebalancer_width}")
else:
    rebalancer_model = None
    print("No Stage 2 rebalancer (disabled for this run).")

# --------------------------------------------------

N_w = (num_points - window_size) // stride + 1

# --------------------------------------------------

path = os.path.join(ROOT, prm.get("Dataset", "Path"))
print(f"Loading dataset from {path}")
dataset = BSplineDataset(path, num_knots, num_points)

# --------------------------------------------------

def extract_windows(pts_batch):
    """
    Extract rotation-invariant intrinsic features from sliding windows.

    Identical to train_gpu.py.  Must match training exactly so that model.acts
    and model.spline_postacts are populated on the correct input distribution
    for meaningful plots and symbolic regression.

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

    windows = pts_3d.unfold(1, window_size, stride)
    windows = windows.permute(0, 1, 3, 2).contiguous()

    chords = windows[:, :, 1:, :] - windows[:, :, :-1, :]
    chords = chords / global_scale.view(B, 1, 1, 1)

    chord_lengths = chords.norm(dim=-1).clamp(min=1e-8)

    d_cur    = chords[:, :, :-1, :]
    d_next   = chords[:, :,  1:, :]
    d_cur_n  = d_cur  / d_cur.norm( dim=-1, keepdim=True).clamp(min=1e-8)
    d_next_n = d_next / d_next.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cross  = d_cur_n[..., 0] * d_next_n[..., 1] - d_cur_n[..., 1] * d_next_n[..., 0]
    dot    = (d_cur_n * d_next_n).sum(dim=-1)
    angles = torch.atan2(cross, dot)

    features = torch.cat([chord_lengths, angles.abs()], dim=-1)
    return features.reshape(B * N_w, n_features)


# Warm-up forward pass to populate model.acts / model.spline_postacts.
# pykan requires batch >= 2 (std uses Bessel's correction) and save_act=True.
_warmup_batch = next(iter(DataLoader(dataset[:10], batch_size=2, shuffle=False)))
_warmup_pts   = _warmup_batch[0].to(device)
_warmup_input = extract_windows(_warmup_pts)

model.save_act = True
with torch.no_grad():
    model(_warmup_input)

# --------------------------------------------------

def save_spline_plots(model, folder="figures"):
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
    Identical to KAN4/post_train.py.
    """
    os.makedirs(folder, exist_ok=True)

    _imgs = os.path.join(ROOT, 'source', 'imgs')
    _sum_img  = plt.imread(os.path.join(_imgs, 'sum_symbol.png'))
    _mult_img = plt.imread(os.path.join(_imgs, 'mult_symbol.png'))

    save_spline_plots(model, folder=folder)

    if metric == 'backward':
        model.attribute()
        scores = model.edge_scores
    elif metric == 'forward_n':
        scores = model.acts_scale
    elif metric == 'forward_u':
        scores = model.edge_actscale
    else:
        raise ValueError(f"metric='{metric}' not recognised.")

    alpha = [np.tanh(beta * s.cpu().detach().numpy()) for s in scores]

    width        = np.array(model.width)
    width_in     = np.array(model.width_in)
    width_out    = np.array(model.width_out)
    A            = 1
    y0           = 0.3
    z0           = 0.1
    neuron_depth = len(width)

    min_spacing = A / np.maximum(np.max(width_out), 5)
    max_neuron  = np.max(width_out)
    y2          = 0.15 / np.maximum(max_neuron, 5)

    num_weights  = width_in[:-1] * width_out[1:]
    y1_per_layer = np.maximum(0.4 / np.maximum(num_weights, 5), y2)
    f            = float(np.maximum(1.0, np.sqrt(float(np.max(num_weights)) / 5.0)))

    fig, ax = plt.subplots(
        figsize=(10 * scale * f, 10 * scale * (neuron_depth - 1) * (y0 + z0))
    )

    DC_to_FC  = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    _fig_h_pts = fig.get_size_inches()[1] * 72
    _aspect_r  = fig.get_size_inches()[1] / fig.get_size_inches()[0]

    _op_h2_nfc = DC_to_NFC([0, y2])[1] - DC_to_NFC([0, 0])[1]
    _op_w2_nfc = _op_h2_nfc * _aspect_r
    node_dot_s = np.pi * (_op_h2_nfc * _fig_h_pts) ** 2

    for l in range(neuron_depth):
        n = width_in[l]
        for i in range(n):
            plt.scatter(1 / (2 * n) + i / n, l * (y0 + z0), s=node_dot_s, color='black')

        if l < neuron_depth - 1:
            y1    = y1_per_layer[l]
            gap   = 0.15 * y1
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
                        [1/(2*n)+i/n,          1/(2*N)+id_/N],
                        [l*(y0+z0),            l*(y0+z0)+y0/2-y1-gap],
                        color=color, lw=2*scale, alpha=alpha[l][j][i]*alpha_mask,
                    )
                    plt.plot(
                        [1/(2*N)+id_/N,        1/(2*n_next)+j/n_next],
                        [l*(y0+z0)+y0/2+y1+gap, l*(y0+z0)+y0],
                        color=color, lw=2*scale, alpha=alpha[l][j][i]*alpha_mask,
                    )

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
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.0)

    for l in range(neuron_depth - 1):
        y1     = y1_per_layer[l]
        n      = width_in[l]
        n_next = width_out[l + 1]
        N      = n * n_next
        pairs  = sorted(
            [(i, j) for i in range(n) for j in range(n_next)],
            key=lambda ij: float(alpha[l][ij[1]][ij[0]])
        )
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

        N_sym = n_sym = width_out[l + 1]
        for j in range(n_sym):
            cx_nfc = DC_to_NFC([1/(2*N_sym)+j/N_sym, 0])[0]
            cy_nfc = DC_to_NFC([0, l*(y0+z0)+y0])[1]
            newax  = fig.add_axes([cx_nfc - _op_w2_nfc, cy_nfc - _op_h2_nfc,
                                   2*_op_w2_nfc, 2*_op_h2_nfc])
            newax.imshow(_sum_img)
            newax.axis('off')

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
    Print to file the symbolic formula for the Stage 1 KAN.

    For window_size=3 (default): 3 inputs = [l_0, l_1, θ_0].
    The formula encodes the learned per-point density weight:
      high score → more density → more knots in this region.
    The theoretically optimal weight is κ ≈ |θ_0| / (l_0 · l_1) or a power thereof.
    """
    m = copy.deepcopy(model)
    m.auto_symbolic(lib=lib)

    formulas, vars_ = m.symbolic_formula()
    with open(path, "w") as f:
        f.write(f"# KAN5 Stage 1 per-point density formula\n")
        f.write(f"# Window size: {window_size}  →  n_features = {n_features}\n")
        if window_size == 3:
            f.write(f"# Inputs:  x[0] = l_0  (left chord length, scale-normalised)\n")
            f.write(f"#          x[1] = l_1  (right chord length, scale-normalised)\n")
            f.write(f"#          x[2] = θ_0  (signed turning angle, radians)\n")
        else:
            f.write(f"# Inputs:  x[0..{window_size-2}] = chord lengths l_0..l_{window_size-2}\n")
            f.write(f"#          x[{window_size-1}..{n_features-1}] = turning angles θ_0..θ_{window_size-3}\n")
        f.write(f"# Output: scalar density score (high = more knots needed here)\n")
        f.write(f"# Architecture: width={width}, mult_arity={mult_arity}\n")
        f.write(f"# Theory: optimal weight ≈ κ^(1/3) where κ ≈ |θ_0| / (l_0 · l_1)\n\n")
        for k, expr in enumerate(formulas):
            f.write(f"output[{k}] = {expr}\n")


# --------------------------------------------------

if FORMULA:
    libs = {
        "formula_full.txt":   ['x', 'x^2', 'x^3', '1/x', '1/x^2', '1/x^3',
                                'sqrt', 'sin', 'cos', 'tan', 'tanh',
                                'exp', 'log', 'abs', 'sgn', '0'],
        "formula_sparse.txt": ['x', 'x^2', 'x^3', '1/x', '1/x^2', 'sqrt', 'abs', 'sgn', '0'],
        "formula_linear.txt": ['x'],
    }

    for filename, lib in libs.items():
        print(f"\nExtracting Stage 1 formula with library: {lib} ...")
        print_formula(model, lib, os.path.join(output_dir, filename))
else:
    print("Skipping formula extraction.")


# Stage 1 KAN overview plot.
# in_vars labels depend on window_size: for w=3 use [l_0, l_1, θ_0].
if window_size == 3:
    in_vars = ['l_0', 'l_1', 'theta_0']
else:
    chord_labels = [f'l_{i}' for i in range(window_size - 1)]
    angle_labels = [f'th_{i}' for i in range(window_size - 2)]
    in_vars = chord_labels + angle_labels

if PLOT:
    print("Plotting Stage 1 KAN ...")
    fig = plot_kan_overview(
        model,
        scale    = 0.5,
        in_vars  = in_vars,
        out_vars = ['density'],
        title    = "KAN5 Stage 1 (per-point density)",
    )
    fig.savefig("figures/model_plot.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Done.")
else:
    print("Skipping plotting.")
