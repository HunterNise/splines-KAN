import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from tasks.functions import *

from tqdm import tqdm


# Set seed for reproducibility
torch.manual_seed(0)

# --------------------------------------------------

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
#   from an initial knot vector to a final knot vector that minimizes the B-spline error.

class NN(nn.Module):
    # class contructor to initialize the neural network architecture and parameters
    def __init__(self, num_knots=5, num_neurons=128, degree=3):
        super().__init__()              # call parent constructor
        
        self.num_knots = num_knots      # store number of knots as object variable for later use
        self.degree = degree            # store degree of B-spline as object variable for later use
        
        # the knot vector must be non-decreasing, so we predict intervals between knots and then convert back to knots
        num_intervals = num_knots - 1
        
        self.stack1 = nn.Sequential(
            nn.Linear(num_intervals, num_neurons),
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
        x = knots_to_intervals(x)   # convert knots to intervals for the network to predict
        x = self.stack1(x)          # pass through first stack of layers
        x = self.stack2(x)          # pass through second stack of layers
        x = intervals_to_knots(x)   # convert back to a non-decreasing vector of knots
        return x

num_knots = 5                       # number of knots (without repetitions/clamping)
num_neurons = 128                   # number of neurons in each hidden layer
degree = 3                          # degree of the B-spline curve
# create model instance and move to device
model = NN(num_knots, num_neurons, degree).to(device)

# --------------------------------------------------

# Training loop to optimize the neural network parameters to minimize the B-spline fitting loss.

def train(model, points, param="uniform",
          max_iter=1000, tol=1e-6, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)     # method to update model parameters based on computed gradients
    losses = []     # vector to store loss value after each iteration
    
    # convert from numpy.ndarray to torch.Tensor, cast to float (with desired precision) and move to device
    points = torch.from_numpy(points).to(precision).to(device)
    # compute parametrization points corresponding to data points
    t_grid = make_grid(points, method=param)
    degree = model.degree
    
    # loop until reaching maximum number of iterations or the error is below the specified tolerance
    for iter in range(max_iter):
    #for iter in tqdm(range(max_iter), ncols=100):
        # initial guess for knots (uniformly spaced)
        initial_knots = torch.linspace(0., 1., model.num_knots, 
                                       dtype=points.dtype, device=points.device)
        
        # predict final knots from the model given the initial knots (forward pass)
        pred_knots = model(initial_knots)
        
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
        
    #     # print loss every 100 iterations
    #     if iter % 100 == 0:
    #         print(f"Iteration {iter:>5n},    Loss: {loss.item():>12,.10f}")
        
    #     # check for convergence: if the loss is below the specified tolerance, stop training
    #     if iter > 0 and losses[-1] < tol:
    #         print(f"Converged at iteration {iter}")
    #         break
    
    # if iter == max_iter - 1:
    #     print("Reached maximum iterations without convergence.")

    return (
        losses,
        # stop tracking gradients for final knots and convert to numpy array for output
        full_knots.detach().cpu().numpy(),
        controls.detach().cpu().numpy(),
    )

# --------------------------------------------------

# directory containing the input data files for training
dir = "/app/data/DNN-Solver/bspline-data/pts/"

for param in ["uniform", "chord_length", "centripetal"]:
    # create output folder if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "outputs" + f"-{param}")
    os.makedirs(output_dir, exist_ok=True)

    # print architecture and number of parameters to file
    model_log = os.path.join(output_dir, "model.txt")
    with open(model_log, "w") as f:
        f.write("Model architecture:\n")
        f.write(str(model) + "\n\n")
        
        f.write("Model parameters:\n")
        for name, module in model.named_modules():
            if module == model:
                name = "Total"
            f.write(f"{name}: {sum(param.numel() for param in module.parameters()):>,.0f}\n")

    for file in tqdm(sorted(os.scandir(dir), key=lambda f: f.name), ncols=100, desc="Processing files"):
        # load points from text file
        path = os.path.join(dir, file)
        name, _ = os.path.splitext(file.name)
        points = np.loadtxt(path)

        # launch training
        #print(f"\nTraining on {name}:\n")
        losses, final_knots, controls = train(model, points, param=param, 
                                                max_iter=2000, tol=eps, lr=1e-3)
        final_err = losses[-1]

        # print final results to file
        results = os.path.join(output_dir, name + "-results.txt")
        with open(results, "w") as f:
            f.write(f"Final error: {final_err:.16f}\n\n")
            f.write(f"Degree: {degree}\n\n")
            f.write(f"Knots:\n{final_knots}\n\n")
            f.write(f"Controls:\n{controls}\n\n")

        # plot the training loss curve and the final curve fit to file
        plot_loss(losses, log=True, 
                    path = output_dir, name = name + "-loss_curve.png")
        plot_curve_fit(points, final_knots, degree, controls, final_err, 
                    path = output_dir, name = name + "-curve_fit.png")
