import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

# Set seed for reproducibility
torch.manual_seed(0)


# Define global variables

INIT = "int"   # "0-int-1", "int", or "all"

PARAMS          = False
GRADIENTS       = False
LOSS            = False
KNOTS           = False
MATRIX_CTRLS    = False
LS_CONTROLS     = False
AT_LEAST_ONE = PARAMS or GRADIENTS or LOSS or KNOTS or MATRIX_CTRLS or LS_CONTROLS
if AT_LEAST_ONE:
    MAX_ITER = 2    # reduce iterations for debugging when printing detailed info
else:
    MAX_ITER = float('inf')

PLOT            = True
CURVE           = True
BASIS_TREE      = False
BASIS_FNS       = False

# --------------------------------------------------

# Load 2D points from text file
path = "/app/data/DNN-Solver/bspline-data/pts/spl_data00.txt"
points = np.loadtxt(path)

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


# Helper functions to convert between knots and intervals,
#   ensuring differentiability and proper dtype/device handling.

def knots_to_intervals(knots: torch.Tensor) -> torch.Tensor:
    # knots is assumed 1‑D; subtract shifted versions
    return knots[1:] - knots[:-1]

def intervals_to_knots(intervals: torch.Tensor) -> torch.Tensor:
    # prepend a 0 and do a cumulative sum
    zero = torch.zeros(1, dtype=intervals.dtype, device=intervals.device)
    return torch.cat((zero, torch.cumsum(intervals, dim=0)))


# Define the neural network model to approximate the mapping
#   from an initial knot vector to a final knot vector that minimizes the B-spline error.

class NN(nn.Module):
    def __init__(self, num_knots=5):    # class constructor with number of knots as argument
        super().__init__()              # call parent constructor
        self.num_knots = num_knots      # store number of knots as object variable for later use
        # the knot vector must be non-decreasing, so we predict intervals between knots and then convert back to knots
        num_intervals = num_knots - 1
        num_neurons = 64
        self.linear1 = nn.Linear(num_intervals, num_neurons)
        self.linear2 = nn.Linear(num_neurons, num_intervals + 2 if INIT == "int" else num_intervals)   # if "int", we predict two extra intervals for the fixed 0 and 1 at the ends

    def forward(self, x):
        x = knots_to_intervals(x)   # convert knots to intervals for the network to predict
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=0)     # ensure intervals are positive and sum to 1
        x = intervals_to_knots(x)   # convert back to a non-decreasing vector of knots
        return x

# number of knots
if INIT == "0-int-1" or INIT == "int":
    num_knots = 5
elif INIT == "all":
    num_knots = 11  # num_knots + 2*degree = 5 + 2*3 = 11
else:
    raise ValueError(f"Unknown INIT value: {INIT}")

model = NN(num_knots).to(device)    # create model instance and move to device

# print architecture and number of parameters
print("\nModel architecture:")
print(model)
print("\nModel parameters:")
for name, module in model.named_modules():
    if module == model:
        name = "Total"
    print(f"{name}: {sum(param.numel() for param in module.parameters()):>,.0f}")
print()


if GRADIENTS:
    # Enable anomaly detection to catch NaN/Inf sources
    torch.autograd.set_detect_anomaly(True)

    # Register hooks to trace gradients through the network
    def make_grad_hook(name):
        def hook(grad):
            if grad is not None:
                grad_norm = grad.norm().item()
                grad_min = grad.min().item()
                grad_max = grad.max().item()
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
                print(f"  {name} gradient: norm={grad_norm:.6e}, min={grad_min:.6e}, max={grad_max:.6e}, NaN={has_nan}, Inf={has_inf}")
        return hook

    # Attach gradient hooks to all model parameters
    for name, param in model.named_parameters():
        param.register_hook(make_grad_hook(name))

    # Register forward hooks to inspect layer outputs
    def make_forward_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                print(f"  Forward {name}: shape={output.shape}, min={output.min():.6e}, max={output.max():.6e}, has_NaN={torch.isnan(output).any().item()}")
        return hook

    for name, module in model.named_modules():
        if hasattr(module, 'register_forward_hook') and name != '':
            module.register_forward_hook(make_forward_hook(name))

# --------------------------------------------------

# Parameterization methods for the input points to create the t_grid for B-spline evaluation.
# Work with torch tensors and ensure the output is on the same device and dtype as the input points.

# uniform parameterization: simply divide the interval [0, 1] into equal parts
def uniform_params(points: torch.Tensor) -> torch.Tensor:
    # create a linearly spaced vector of length N (number of points) from 0 to 1
    return torch.linspace(0., 1., len(points), 
                          dtype=points.dtype, device=points.device)

# chord length parameterization: proportional to the distance between points
def chord_length_params(points: torch.Tensor) -> torch.Tensor:
    distances = torch.sqrt(torch.sum((points[1:] - points[:-1]) ** 2, dim=1))   # compute distances between consecutive points
    cumulative_distances = torch.cumsum(distances, dim=0)                       # cumulative sum of distances to get the parameter values before normalization
    total_distance = cumulative_distances[-1]                                   # total length of the curve (last value of cumulative distances)
    zero = torch.zeros(1, dtype=points.dtype, device=points.device)             # create a zero tensor to prepend to the cumulative distances
    return torch.cat((zero, cumulative_distances / total_distance))             # prepend 0 and normalize to [0, 1]

# centripetal parameterization: proportional to the square root of the distance between points
def centripetal_params(points: torch.Tensor) -> torch.Tensor:
    distances = torch.sqrt(torch.sum((points[1:] - points[:-1]) ** 2, dim=1))
    cumulative_distances = torch.cumsum(torch.sqrt(distances), dim=0)   # same as chord length but with sqrt of distances
    total_distance = cumulative_distances[-1]
    zero = torch.zeros(1, dtype=points.dtype, device=points.device)
    return torch.cat((zero, cumulative_distances / total_distance))

# Choose the appropriate parameterization method based on the input argument
def make_grid(points, method="uniform"):
    if method == "uniform":
        return uniform_params(points)
    elif method == "chord_length":
        return chord_length_params(points)
    elif method == "centripetal":
        return centripetal_params(points)
    else:
        raise ValueError(f"Unknown parameterization method: {method}")


# Assemble matrix of B-spline basis functions evaluated at the parameter grid for the given knots and degree.
def bspline_basis(t_grid, knots, degree, soft=False, k=1000.0):
    """
    Vectorized Cox--de Boor B-spline basis (autograd-friendly).

    Inputs
    ------
    t_grid : (N,) tensor (float), points where to evaluate, in same dtype/device as knots
    knots  : (K,) tensor (non-decreasing)
    degree : int >= 0
    soft   : if False -> hard indicator for degree 0 (exact, non-differentiable).
             if True  -> soft indicator using sigmoid(k * (x - left))*sigmoid(k * (right - x))
    k      : steepness for sigmoid when soft=True (larger k -> closer to hard step)

    Returns
    -------
    B : (N, M) where M = K - degree - 1
    """
    N = t_grid.shape[0]     # number of evaluation points
    K = knots.shape[0]      # number of knots
    m = K - 1               # number of intervals between knots, which is also the number of degree-0 basis functions
    if m <= 0:
        return torch.zeros((N, 0), dtype=t_grid.dtype, device=t_grid.device)

    # lefts: knots[0..K-2], rights: knots[1..K-1]
    lefts = knots[:-1]          # shape (m,)
    rights = knots[1:]          # shape (m,)
    # expand t_grid to (N,1) for broadcasting against (m,) intervals
    t_expand = t_grid.unsqueeze(1)   # (N,1)

    # degree-0 basis (vectorized)
    # compute indicator functions for each interval [lefts[j], rights[j]) for j=0..m-1

    # hard mask: intervals [left, right) except last interval which includes right endpoint
    mask = (t_expand >= lefts) & (t_expand < rights)    # (N,m) boolean
    
    # find the last non-degenerate interval (where lefts[j] < rights[j])
    last_nondegen_idx = -1
    for jj in range(len(lefts) - 1, -1, -1):    # loop from the end
        if lefts[jj] < rights[jj]:
            last_nondegen_idx = jj
            break
    
    # include right endpoint for the last non-degenerate interval
    if last_nondegen_idx >= 0:
        mask_last = t_grid == rights[last_nondegen_idx] # (N,) boolean
        mask = mask.clone()                             # create a copy to avoid in-place modification which may cause issues with autograd
        mask[:, last_nondegen_idx] = mask[:, last_nondegen_idx] | mask_last     # logical OR between mask (<) and mask_last (==) for the last interval
    
    hard = mask.to(dtype=knots.dtype)      # convert boolean to float

    if not soft:
        Bprev = hard
    else:
        # soft mask: use product of two sigmoids -> smooth approximation of indicator
        # torch.sigmoid : [-inf,+inf] -> [0,1],  0 |-> 1/2,  large positive -> ~1,  large negative -> ~0
        left_term  = torch.sigmoid(k * (t_expand - lefts))      # (N,m)
        right_term = torch.sigmoid(k * (rights - t_expand))     # (N,m)
        soft_vals = (left_term * right_term)
        
        # zero out zero-width intervals — they should contribute nothing,
        # but sigmoid(0)*sigmoid(0)=0.25 would otherwise corrupt the recursion
        non_degenerate = (rights > lefts).unsqueeze(0)          # (1, m) bool
        soft_vals = soft_vals * non_degenerate
        hard      = hard * non_degenerate

        # Straight-Through Estimator (STE):
        #   in the forward pass this equals hard exactly;
        #   in the backward pass autograd sees only soft_vals (hard does not depend on knots, so it has zero gradients)
        Bprev = hard + (soft_vals - soft_vals.detach())

    # recursion for p = 1..degree
    # at recursion p we should produce shape (N, K-p-1)
    for p in range(1, degree + 1):
        M = K - p - 1       # number of basis functions at degree p
        if M <= 0:
            return torch.zeros((N, 0), dtype=t_grid.dtype, device=t_grid.device)
        
        # slices for numerators (with broadcasting):
        # num1_j = t_grid - knots[j]      for j=0..M-1  =>  t_grid[:,None] - knots[0:M]
        # num2_j = knots[j+p+1] - t_grid  for j=0..M-1  =>  knots[p+1 : M+p+1] - t_grid[:,None]
        num1 = t_expand - knots[0 : M]                          # (N, M)
        num2 = knots[p + 1 : p + 1 + M] - t_expand              # (N, M)

        # slices for denominators:
        # denom1_j = knots[j+p]   - knots[j]    for j=0..M-1  =>  knots[p : M+p] - knots[0:M]
        # denom2_j = knots[j+p+1] - knots[j+1]  for j=0..M-1  =>  knots[p+1 : M+p+1] - knots[1:M+1]
        denom1 = knots[p : p + M] - knots[0 : M]                # (M,)
        denom2 = knots[p + 1 : p + 1 + M] - knots[1 : M + 1]    # (M,)

        # expand denom1/2 for broadcasting
        denom1_expand = denom1.unsqueeze(0).expand_as(num1)     # (N, M)
        denom2_expand = denom2.unsqueeze(0).expand_as(num2)     # (N, M)

        # convention: coefficients are zero if the corresponding denominator is zero, which effectively removes that term from the sum
        # safe division: compute coef = num / denom where denom != 0, else 0
        # use masking to avoid dividing by zero (safe for gradients)
        mask1 = denom1_expand != 0
        coef1 = torch.zeros_like(num1)
        coef1[mask1] = num1[mask1] / denom1_expand[mask1]
        
        mask2 = denom2_expand != 0
        coef2 = torch.zeros_like(num2)
        coef2[mask2] = num2[mask2] / denom2_expand[mask2]

        # slices for basis matrix:
        # B_j^{p} = coef1_j * B_j^{p-1} + coef2_j * B_{j+1}^{p-1}  for j=0..M-1
        # Bprev has shape (N, previous_M = M+1)
        left_term  = coef1 * Bprev[:, 0 : M    ]    # (N, M)
        right_term = coef2 * Bprev[:, 1 : M + 1]    # (N, M)
        Bp = left_term + right_term                 # (N, M)
        Bprev = Bp                                  # allocate new tensor each degree
    
    return Bprev   # shape (N, K-degree-1)

# # Given the basis matrix and target points, solve for control points using least squares.
# # Uses Tikhonov regularization to ensure numerical stability (if ill-conditioned) and differentiability (if singular).
# def solve_control_points(B, X, reg=1e-6):
#     # B: (N, M), X: (N, d)
#     # solve min_C ||B @ C - X||^2 + reg * ||C||^2 equivalent to the normal equations (B^T B + reg*I) C = B^T X
#     Bt = B.transpose(0,1)               # (M, N)
#     G = Bt @ B                          # (M, M)
#     M = G.shape[0]
#     Greg = G + torch.eye(M, dtype=B.dtype, device=B.device) * reg
#     rhs = Bt @ X                        # (M, d)
#     C = torch.linalg.solve(Greg, rhs)   # (M, d)
#     return C

# Enhanced version of solve_control_points with diagnostics for singularity and condition number.
def solve_control_points(B, X, reg=1e-6):
    # DEBUG: diagnose singularity
    if LS_CONTROLS:
        print(f"\n--- Basis Matrix Diagnostics ---")
        print(f"B shape: {B.shape}")
        print(f"B min: {B.min():.2e}, B max: {B.max():.2e}")
        print(f"B column norms: min={B.norm(dim=0).min():.2e}, max={B.norm(dim=0).max():.2e}")
    
    Bt = B.transpose(0,1)
    G = Bt @ B
    if LS_CONTROLS:
        print(f"G = B^T B min: {G.min():.2e}, G max: {G.max():.2e}")
        print(f"G condition number: {torch.linalg.cond(G):.2e}")
    
    # Compute rank of B (using SVD)
    U, S, Vh = torch.svd(B)
    effective_rank = (S > 1e-10).sum().item()
    if LS_CONTROLS:
        print(f"B effective rank: {effective_rank}/{B.shape[1]}")
        print(f"B singular values: min={S.min():.2e}, max={S.max():.2e}, ratio={S.max()/S.min():.2e}")
    
    # Try solve with increased regularization
    M = G.shape[0]
    Greg = G + torch.eye(M, dtype=B.dtype, device=B.device) * reg
    if LS_CONTROLS:
        print(f"Regularization: {reg:.2e}")
        print(f"Condition of (G + reg*I): {torch.linalg.cond(Greg):.2e}")
    
    rhs = Bt @ X
    C = torch.linalg.solve(Greg, rhs)
    return C


# Training loop to optimize the neural network parameters to minimize the B-spline fitting loss.

def train(model, points, degree, param="uniform",
          max_iter=1000, tol=1e-6, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)     # method to update model parameters based on computed gradients
    losses = []     # vector to store loss value after each iteration
    
    # convert from numpy to torch.Tensor, cast to torch.float32 and move to device
    points = torch.from_numpy(points).float().to(device)
    # compute parametrization points corresponding to data points
    t_grid = make_grid(points, method=param)
    
    # loop until reaching maximum number of iterations or the error is below the specified tolerance
    for iter in range( min(max_iter, MAX_ITER) ):
        if AT_LEAST_ONE:
            print(f"\n--------------------\n\nIteration {iter:>5n}\n")   ###
        # print model parameters (weights of linear layers) for debugging
        if PARAMS:
            for name, param in model.named_parameters():    ###
                print(f"Parameter {name}:\n{param.data}")   ###

        # initial guess for knots (uniformly spaced)
        if INIT == "0-int-1":
            initial_knots = torch.linspace(0., 1., model.num_knots, 
                                           dtype=points.dtype, device=points.device)
        elif INIT == "int":
            # add two more knots
            initial_knots = torch.linspace(0., 1., model.num_knots + 2, 
                                           dtype=points.dtype, device=points.device)
            # remove the first and last knot for the model input (since they are fixed at 0 and 1)
            initial_knots = initial_knots[1:-1]
        elif INIT == "all":
            # internal knots + first 0 and last 1
            initial_knots = torch.linspace(0., 1., model.num_knots - 2*degree, 
                                           dtype=points.dtype, device=points.device)
            # padding to achieve p+1 multiplicity at the ends
            initial_knots = torch.cat((
                                torch.zeros(degree, dtype=initial_knots.dtype, device=initial_knots.device), 
                                initial_knots, 
                                torch.ones( degree, dtype=initial_knots.dtype, device=initial_knots.device)
                            ))
        if KNOTS:
            print(f"\nInitial knots:\n{initial_knots}")   ###
        
        # predict final knots from the model given the initial knots (forward pass)
        if GRADIENTS:
            print("\nForward pass:")                      ###
        pred_knots = model(initial_knots)
        if KNOTS:
            print(f"\nPredicted knots:\n{pred_knots}")    ###
        
        if INIT == "0-int-1" or INIT == "int":
            # pad the internal knots to open/clamped knots
            # to avoid numerical errors from the cumsum + softmax, we throw away the first and last predicted knots and replace them with exact 0 and 1
            full_knots = torch.cat((
                            torch.zeros(degree + 1, dtype=pred_knots.dtype, device=pred_knots.device),
                            pred_knots[1:-1],      # interior knots only, guaranteed in (0, 1)
                            torch.ones( degree + 1, dtype=pred_knots.dtype, device=pred_knots.device)
                        ))
        elif INIT == "all":
            # since the model already outputs the full knot vector with multiplicity due to the way we set up the input and softmax, we can use pred_knots directly as full_knots
            full_knots = pred_knots
        if KNOTS:
            print(f"\nFull knots:\n{full_knots}")         ###
        
        # compute the B-spline basis matrix evaluated at the parameter grid for the predicted knots and given degree
        basis_matrix = bspline_basis(t_grid, full_knots, degree, soft=False, k=1000.0)
        if MATRIX_CTRLS:
            print(f"\nBasis matrix:\n{basis_matrix}")     ###
        # solve for control points that best fit the data points given the basis matrix (least squares)
        controls = solve_control_points(basis_matrix, points, reg=1e-6)
        if MATRIX_CTRLS:
            print(f"\nControls:\n{controls}")             ###

        # compute loss as sum of distances (norm2) between data points and points on the B-spline curve
        loss = torch.sum((points - basis_matrix @ controls) ** 2)
        losses.append(loss.item())
        if LOSS:
            print(f"\nLoss value: {loss.item():.6e}")     ###
        
        if GRADIENTS:
            print("\nBackward pass:")   ###
        loss.backward()         # compute gradients of the loss with respect to model parameters using backpropagation
        
        ###
        if GRADIENTS:
            print("\nGradient summary after backward:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_min = param.grad.min().item()
                    grad_max = param.grad.max().item()
                    has_nan = torch.isnan(param.grad).any().item()
                    has_inf = torch.isinf(param.grad).any().item()
                    print(f"  {name}: norm={grad_norm:.6e}, min={grad_min:.6e}, max={grad_max:.6e}, NaN={has_nan}, Inf={has_inf}")
                else:
                    print(f"  {name}: No gradient!")
        
        optimizer.step()        # update model parameters based on computed gradients
        optimizer.zero_grad()   # reset gradients to zero for the next iteration
        
        if not LOSS:
            # print loss every 100 iterations
            if iter % 100 == 0:
                print(f"Iteration {iter:>5n},    Loss: {loss.item():>12,.6f}")
        
        # check for convergence: if the loss is below the specified tolerance, stop training
        if iter > 0 and losses[-1] < tol:
            print(f"Converged at iteration {iter}")
            break
    
    if iter == max_iter - 1:
        print("Reached maximum iterations without convergence.")

    return (
        losses,
        # stop tracking gradients for final knots and convert to numpy array for output
        full_knots.detach().cpu().numpy(),
        controls.detach().cpu().numpy(),
    )

# launch training and print final knots
degree = 3
losses, final_knots, controls = train(model, points, degree, param="uniform")
print("\nFinal knots:\n", final_knots)
final_err = losses[-1]
print(f"\nFinal error: {final_err:.6f}")

# --------------------------------------------------

import matplotlib.pyplot as plt
import os

if PLOT:
    # plot losses (log scale)
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # create output folder if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    # save plot to external file
    plt.savefig(os.path.join(output_dir, "training_loss.png"))


# Recursive implementation of the Cox--de Boor formula for evaluating a single B-spline basis function at a given parameter t.
def bspline_basis_eval(j, p, t, knots):
    if p == 0:
        # 0-degree basis functions are indicator functions for the intervals defined by the knots
        # support = [left, right) except the last non-degenerate interval which includes the right endpoint

        # find last non-degenerate interval
        last_nondegen_idx = -1
        for jj in range(len(knots) - 2, -1, -1):    # loop from the end
            if knots[jj] < knots[jj + 1]:
                last_nondegen_idx = jj
                break
        
        # include right endpoint of last non-degenerate interval
        if j == last_nondegen_idx and t == knots[j + 1]:
            return 1.0
        else:
            return 1.0 if knots[j] <= t < knots[j + 1] else 0.0
    else:
        coef1 = (t - knots[j]) / (knots[j + p] - knots[j]) if knots[j + p] != knots[j] else 0
        coef2 = (knots[j + p + 1] - t) / (knots[j + p + 1] - knots[j + 1]) if knots[j + p + 1] != knots[j + 1] else 0
        return coef1 * bspline_basis_eval(j, p - 1, t, knots) + coef2 * bspline_basis_eval(j + 1, p - 1, t, knots)

# Evaluate the B-spline curve at a given parameter t using the control points and the basis functions.
def bspline_eval(t, knots, degree, controls):
    # S(t) = sum_j B_j^p(t) * control_j
    S = 0
    for j in range(len(controls)):
        B_j_p = bspline_basis_eval(j, degree, t, knots)
        S += B_j_p * controls[j]
    return S


# Plot the original data points, the fitted B-spline curve, and the knot points for visualization.

t_grid = np.linspace(0., 1., 1000)  # dense grid for smooth curve
curve_points = np.array([bspline_eval(t, final_knots, degree, controls) for t in t_grid])
knot_points  = np.array([bspline_eval(u, final_knots, degree, controls) for u in final_knots])
if CURVE:
    print("\nCurve points:\n", curve_points)    ###
    print("\nControls:\n", controls)            ###

if PLOT:
    plt.figure(figsize=(10, 8))
    plt.plot(points[:, 0], points[:, 1], 
                marker='o', linestyle='none', color='gray', markersize=4, label='Data Points')
    plt.plot(curve_points[:, 0], curve_points[:, 1], 
                linestyle='-', color='blue', label='B-spline Curve')
    plt.plot(knot_points[:, 0], knot_points[:, 1], 
                marker='^', linestyle='none', color='green', markersize=10, label='Knot Points')

    # make room on the right for legend/text
    plt.subplots_adjust(right=0.75)   # leave 25% width on the right for annotations
    # place legend in the reserved right margin
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    # add a red rectangle in the lower right corner to display the final error
    plt.text(1.03, 0.02, f"Error: {final_err:.2e}", 
                transform=plt.gca().transAxes,  # convert to percentage coordinates of the axes
                fontsize=12, color='red', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))

    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

    # save plot to external file
    plt.savefig(os.path.join(output_dir, "curve_fit.png"))


# Helper function to print the relevant B-spline basis functions in a tree structure.
def bspline_basis_tree(j, p, knots, indent=""):
    print(f"{indent}B_{j}^{p}")
    if p == 0:
        return
    else:
        bspline_basis_tree(j    , p - 1, knots, indent + "  ")
        bspline_basis_tree(j + 1, p - 1, knots, indent + "  ")
        return

if BASIS_TREE:
    print("\nB-spline basis functions tree:")
    for j in range(len(final_knots) - degree - 1):
        bspline_basis_tree(j, degree, final_knots)

# Evaluate and print the B-spline basis functions at specific parameter values for verification.

if BASIS_FNS:
    print("\n\nB-spline basis functions:")
    for p in range(degree + 1):
        print(f"\nB-spline basis functions of degree {p}:")
        n = len(final_knots) - p - 1
        for j in range(n):
            for t_test in [0.0, 0.5, 1.0]:
                val = bspline_basis_eval(j, p, t_test, final_knots)
                print(f"B_{j}^{p}({t_test}) = {val:0.3f};    ", end="")
            print()

    print("\n\nVectorized version (hard):")
    for p in range(degree + 1):
        print(f"\nB-spline basis functions of degree {p}:")
        n = len(final_knots) - p - 1
        t_grid = torch.Tensor([0.0, 0.5, 1.0])
        mat = bspline_basis(t_grid, torch.from_numpy(final_knots), p, soft=False)
        print(mat.transpose(0,1))

    print("\n\nVectorized version (soft):")
    for p in range(degree + 1):
        print(f"\nB-spline basis functions of degree {p}:")
        n = len(final_knots) - p - 1
        t_grid = torch.Tensor([0.0, 0.5, 1.0])
        mat = bspline_basis(t_grid, torch.from_numpy(final_knots), p, soft=True, k=1000.0)
        print(mat.transpose(0,1))
