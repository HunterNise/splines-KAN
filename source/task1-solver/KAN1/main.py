"""
[task1-solver/KAN1] KAN-based B-spline knot optimizer — KAN version of NN1.

Replaces the MLP solver with a Kolmogorov-Arnold Network (KAN) while keeping the same
single-sample, single-knot-count task as NN1. Changes vs NN1/main.py:
- Model: KAN([num_intervals, num_hidden, num_intervals]) with learnable cubic B-spline
  activations on each edge (grid=5, k=3) instead of fixed ReLU activations on nodes.
- KAN output is in interval space; softmax is applied manually after the forward pass.
- Visualization: plots the KAN graph before training, after training, and after pruning.
- Model pruning: model.prune() removes low-contribution nodes post-training; the pruned
  model is re-evaluated and its architecture saved.
- Requires a warm-up forward pass (reinit, batch≥2) to initialize KAN spline coefficients
  before plotting or training.
"""


import torch
import torch.nn.functional as F

from kan import KAN

import numpy as np

from source.functions import *


# Set seed for reproducibility
torch.manual_seed(0)

import os
# create output folder if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)
# Change the working directory to the current file's directory to ensure that the figures are saved in the correct location
os.chdir(output_dir)

# --------------------------------------------------

# Load 2D points from text file
path = ROOT + "/data/DNN-Solver/bspline-data/pts/spl_data10.txt"
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

# use double precision for better numerical stability
precision = torch.float64
torch.set_default_dtype(precision)
eps = torch.finfo(precision).eps    # machine epsilon for the chosen precision, used as convergence tolerance lower bound

# --------------------------------------------------

# Hyperparameters for the KAN model and B-spline fitting
num_knots = 6           # number of internal knots (without clamping repetitions)
num_hidden = 8          # number of hidden neurons in the KAN hidden layer
degree = 3              # degree of the B-spline curve
num_intervals = num_knots - 1   # KAN input/output dimension (intervals between knots)

# Define a KAN (Kolmogorov-Arnold Network) to approximate the mapping from an initial
# knot interval vector to an optimized one that minimizes the B-spline fitting error.
#
# Architecture: [num_intervals] -> [num_hidden] -> [num_intervals]
#   - Input:  num_intervals values (intervals of a uniform initial knot vector)
#   - Output: num_intervals values (predicted intervals, normalized via softmax to sum to 1)
#
# Unlike a vanilla NN, KAN places learnable univariate spline functions on each edge
# (connection) rather than fixed activation functions on nodes, giving it more expressive
# power per parameter for smooth function approximation.
model = KAN(
    width   = [num_intervals, num_hidden, num_intervals],   # layer sizes (input, hidden, output)
    grid    = 5,                                            # number of grid intervals per spline activation
    k       = 3,                                            # spline order (cubic B-splines on edges)
    seed    = 0,                                            # random seed for reproducibility
    device  = device,
    auto_save = False,                                      # disable automatic checkpoint saving
)

# print architecture and number of parameters
print("\nModel architecture:")
print(model)
print(f"\nTotal trainable parameters: {sum(param.numel() for param in model.parameters()):>,.0f}")
print()


def reinit(model):
    initial_knots = torch.linspace(0., 1., num_knots, dtype=precision, device=device)
    initial_intervals = knots_to_intervals(initial_knots)

    with torch.no_grad():
        # batch >= 2 required so torch.std(x, dim=0) doesn't produce NaN
        # (Bessel's correction divides by n-1, which is 0 for batch=1)
        batch_input = initial_intervals.unsqueeze(0).expand(64, -1).clone()
        batch_input = batch_input + torch.randn_like(batch_input) * 1e-4  # tiny noise so values differ
        model(batch_input)

reinit(model)   # forward pass with batch input to initialize the model and cache spline coefficients (necessary for plotting)
model.plot(beta=100, scale=2.0)
plt.savefig(os.path.join(output_dir, "figures/model_plot0.png")); plt.close()

# --------------------------------------------------

# Training loop to optimize the neural network parameters to minimize the B-spline fitting loss.

def train(model, points, method="uniform",
          max_iter=1000, tol=1e-6, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)     # method to update model parameters based on computed gradients
    losses = []     # vector to store loss value after each iteration
    
    # convert from numpy.ndarray to torch.Tensor, cast to float (with desired precision) and move to device
    points = torch.from_numpy(points).to(precision).to(device)
    # compute parametrization points corresponding to data points
    t_grid = make_grid(points, method=method)
    
    # loop until reaching maximum number of iterations or the error is below the specified tolerance
    for iter in range(max_iter):
        # initial guess for knots (uniformly spaced)
        initial_knots = torch.linspace(0., 1., num_knots, 
                                       dtype=points.dtype, device=points.device)

        # Convert knots to intervals so the KAN operates in unconstrained interval space,
        # shape (num_intervals,)
        initial_intervals = knots_to_intervals(initial_knots)

        # KAN expects 2D input (batch, features); add a batch dimension -> (1, num_intervals)
        kan_input = initial_intervals.unsqueeze(0)

        # Forward pass through the KAN: shape (1, num_intervals)
        pred_intervals_raw = model(kan_input)

        # Remove the batch dimension and apply softmax so the predicted intervals are
        # positive and sum to 1, making them a valid probability simplex -> (num_intervals,)
        pred_intervals = F.softmax(pred_intervals_raw.squeeze(0), dim=0)

        # Convert the normalized intervals back to a non-decreasing knot vector,
        # shape (num_knots,)
        pred_knots = intervals_to_knots(pred_intervals)

        # pad the internal knots to open/clamped knots
        # to avoid numerical errors from the cumsum + softmax, we throw away the first and last predicted knots and replace them with exact 0 and 1
        full_knots = torch.cat((
                        torch.zeros(degree + 1, dtype=pred_knots.dtype, device=pred_knots.device),
                        pred_knots[1:-1],      # interior knots only, guaranteed in (0, 1)
                        torch.ones( degree + 1, dtype=pred_knots.dtype, device=pred_knots.device)
                     ))
        
        # compute the B-spline basis matrix evaluated at the parameter grid for the predicted knots and given degree
        basis_matrix = bspline_basis_matrix(t_grid, full_knots, degree, soft=False, k=1000.0)
        # solve for control points that best fit the data points given the basis matrix (least squares)
        controls = solve_control_points(basis_matrix, points, reg=1e-6)

        # compute loss as sum of distances (norm2) between data points and points on the B-spline curve
        loss = torch.sum((points - basis_matrix @ controls) ** 2)
        losses.append(loss.item())
        
        loss.backward()         # compute gradients of the loss with respect to model parameters using backpropagation
        optimizer.step()        # update model parameters based on computed gradients
        optimizer.zero_grad()   # reset gradients to zero for the next iteration
        
        # print loss every 100 iterations
        if iter % 100 == 0:
            print(f"Iteration {iter:>5n},    Loss: {loss.item():>12,.16f}")

            for i, layer in enumerate(model.act_fun):
                c = layer.coef.data
                print(f"Layer {i}: max={c.abs().max():.4f}, has_nan={torch.isnan(c).any().item()}, has_inf={torch.isinf(c).any().item()}")
            
            for i, acts in enumerate(model.acts):
                print(f"acts[{i}]: max={acts.abs().max():.4f}, has_nan={torch.isnan(acts).any()}, shape={acts.shape}")
            
            print()
        
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

# launch training and print final knots and error
losses, final_knots, controls = train(model, points, method="uniform", 
                                      max_iter=1500, tol=eps, lr=1e-3)
print(f"\nFinal knots:\n{final_knots}")
final_err = losses[-1]
print(f"\nFinal error: {final_err:.16f}")


# plot model to file
reinit(model)   # re-initialize the model to reset the activation cache for plotting
model.plot(beta=100, scale=2.0)
plt.savefig(os.path.join(output_dir, "figures/model_plot.png")); plt.close()

# prune() internally creates a model with auto_save=True (hardcoded in prune_node),
# so the checkpoint directory and history file must exist beforehand.
os.makedirs(model.ckpt_path, exist_ok=True)
open(os.path.join(model.ckpt_path, 'history.txt'), 'a').close()

model = model.prune()
model.auto_save = False     # disable auto_save on the returned pruned model

reinit(model)               # repopulate model.acts for plotting
model.plot(beta=100, scale=2.0)
plt.savefig(os.path.join(output_dir, "figures/model_plot_pruned.png")); plt.close()

# --------------------------------------------------

# print final results to file
results = os.path.join(output_dir, "results.txt")
with open(results, "w") as f:
    f.write(f"Final error: {final_err:.16f}\n\n")
    f.write(f"Degree: {degree}\n\n")
    f.write(f"Knots:\n{final_knots}\n\n")
    f.write(f"Controls:\n{controls}\n\n")

# plot the training loss curve and the final curve fit to file
plot_loss(losses, log=True, 
            path = output_dir, name = "loss_curve.png")
plot_curve_fit(points, final_knots, degree, controls, final_err, 
            path = output_dir, name = "curve_fit.png")
