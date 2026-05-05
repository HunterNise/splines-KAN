import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from source.functions import *


# Set seed for reproducibility
torch.manual_seed(0)

# --------------------------------------------------

# Load 2D points from text file
path = ROOT + "/data/DNN-Solver/bspline-data/pts/spl_data10.txt"
points = np.loadtxt(path)

num_points, dim = points.shape

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


# Define the neural network model to approximate the mapping
#   from input data points to a final knot vector that minimizes the B-spline error.

class NN(nn.Module):
    # class contructor to initialize the neural network architecture and parameters
    def __init__(self, num_knots=5, num_neurons=128, degree=3):
        super().__init__()              # call parent constructor
        
        self.num_knots = num_knots      # store number of knots as object variable for later use
        self.degree = degree            # store degree of B-spline as object variable for later use
        
        # the knot vector must be non-decreasing, so we predict intervals between knots and then convert back to knots
        num_intervals = num_knots - 1
        
        self.stack1 = nn.Sequential(
            nn.Linear(num_points * dim, num_neurons),   # input layer: takes flattened data points as input
            nn.ReLU(),
            nn.Linear(num_neurons, num_intervals),
            nn.Softmax(dim=0)       # ensure intervals are positive and sum to 1
        )
        self.stack2 = nn.Sequential(
            nn.Linear(num_intervals, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_intervals),
            nn.Softmax(dim=0)       # ensure intervals are positive and sum to 1
        )

    def forward(self, x):
        x = self.stack1(x)          # pass through first stack of layers
        x = self.stack2(x)          # pass through second stack of layers
        x = intervals_to_knots(x)   # convert back to a non-decreasing vector of knots
        return x

num_knots = 6                       # number of knots (without repetitions/clamping)
num_neurons = 128                   # number of neurons in each hidden layer
degree = 3                          # degree of the B-spline curve
# create model instance and move to device
model = NN(num_knots, num_neurons, degree).to(device)

# print architecture and number of parameters
print("\nModel architecture:")
print(model)
print("\nModel parameters:")
for name, module in model.named_modules():
    if module == model:
        name = "Total"
    print(f"{name}: {sum(param.numel() for param in module.parameters()):>,.0f}")
print()

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
    degree = model.degree
    
    # loop until reaching maximum number of iterations or the error is below the specified tolerance
    for iter in range(max_iter):
        # predict final knots from the model given the input data points
        pred_knots = model(points.flatten())   # flatten the input points to a 1D vector for the model
        
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
                                      max_iter=2000, tol=eps, lr=1e-3)
print(f"\nFinal knots:\n{final_knots}")
final_err = losses[-1]
print(f"\nFinal error: {final_err:.16f}")

# --------------------------------------------------

# Gradient saliency: input attribution to quantify how much each data point
# drives the placement of the learned interior knots.
#
# We compute the Jacobian  J[j, i] = || ∂t_j / ∂x_i ||_2
#   where t_j is the j-th interior (free) knot and x_i is the i-th input point.
# Each J[j, i] entry is the L2 norm of the gradient over the coordinate dimensions of x_i.
# This tells us: "how sensitive is knot t_j to a small perturbation of point x_i?"
#
# Per-point saliency aggregates J over all interior knots (mean), giving a single
# scalar per input point that reflects its overall influence on knot placement.

model.eval()    # switch to eval mode (disables dropout etc., if any)

# wrap flattened input points as a leaf tensor that tracks gradients
points_tensor = torch.from_numpy(points).to(precision).to(device)
x_sal = points_tensor.flatten().detach().requires_grad_(True)   # shape: (num_points * dim,)

# forward pass — identical structure to the training loop
pred_knots_sal = model(x_sal)
full_knots_sal = torch.cat((
    torch.zeros(degree + 1, dtype=pred_knots_sal.dtype, device=pred_knots_sal.device),
    pred_knots_sal[1:-1],   # interior predicted knots (first and last are clamped to 0/1)
    torch.ones( degree + 1, dtype=pred_knots_sal.dtype, device=pred_knots_sal.device)
))

# extract only the free (interior) knots: these are the ones the network actually learned;
# the clamped endpoints (degree+1 zeros and ones) are fixed constants and carry no gradient
interior_knots = full_knots_sal[degree + 1 : -(degree + 1)]    # shape: (num_free,)
num_free = interior_knots.shape[0]
#print(f"\nComputing gradient saliency for {num_free} interior knot(s) over {num_points} points...")

# Jacobian tensor: J[j, i, d] = ∂t_j / ∂x_{i,d}  (before taking the norm)
jacobian = torch.zeros(num_free, num_points, dim, dtype=precision)

for j in range(num_free):
    # zero out accumulated gradients from the previous knot's backward pass
    if x_sal.grad is not None:
        x_sal.grad.zero_()
    # differentiate interior knot t_j w.r.t. all input coordinates;
    # retain_graph keeps the computation graph alive for subsequent backward calls
    interior_knots[j].backward(retain_graph=(j < num_free - 1))
    jacobian[j] = x_sal.grad.view(num_points, dim).abs()   # take absolute value; we care about magnitude

# per-knot per-point saliency: collapse the coordinate dimension via L2 norm
# shape: (num_free, num_points)  — rows are knots, columns are input points
knot_jacobian = jacobian.norm(dim=-1).detach().cpu().numpy()

# aggregate over all interior knots to get a single importance score per input point
# shape: (num_points,)
point_saliency = knot_jacobian.mean(axis=0)

#print(f"Point saliency:\n{point_saliency}")

model.train()   # restore training mode

# --------------------------------------------------

# create output folder if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)


# print final results to file
results = os.path.join(output_dir, "results.txt")
with open(results, "w") as f:
    f.write(f"Final error: {final_err:.16f}\n\n")
    f.write(f"Degree: {degree}\n\n")
    f.write(f"Knots:\n{final_knots}\n\n")
    f.write(f"Controls:\n{controls}\n\n")
    f.write(f"Point saliency (mean |∂tⱼ/∂xᵢ| over interior knots):\n{point_saliency}\n\n")
    f.write(f"Knot-point Jacobian (num_free_knots × num_points):\n{knot_jacobian}\n\n")

# plot the training loss curve and the final curve fit with saliency attribution to file
plot_loss(losses, log=True, 
            path = output_dir, name = "loss_curve.png")
plot_curve_fit_saliency(points, final_knots, degree, controls, final_err,
            point_saliency, knot_jacobian,
            path = output_dir, name = "curve_fit.png")
