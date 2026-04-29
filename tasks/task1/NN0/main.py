import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

from tasks.functions import *


import os

from tqdm import tqdm


# Set seed for reproducibility
torch.manual_seed(0)

# --------------------------------------------------

# create output folder if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)
# Change the working directory to the current file's directory to ensure that the figures are saved in the correct location
os.chdir(output_dir)

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

# --------------------------------------------------

# Load 2D points from text file

num_knots = 6       # number of knots (without repetitions/clamping)
degree = 3          # degree of the B-spline curve
param = "uniform"   # method to compute parameter values corresponding to data points


path = "/app/data/DNN-Solver/bspline-data"

class BSplineDataset(Dataset):
    def __init__(self, indices, path, degree=3):
        self.samples = []
        for i in indices:
            pts   = np.loadtxt(f"{path}/pts/spl_data{i:02d}.txt")   # (100, 2)
            knots = np.loadtxt(f"{path}/knot/{i:02d}_knot.txt")     # variable length
            interior = knots[degree+1 : -(degree+1)]                # strictly interior knots in (0,1)
            
            pts_t   = torch.from_numpy(pts).to(precision)
            label_t = torch.from_numpy(interior).to(precision)
            self.samples.append((pts_t.flatten(), label_t))
        
        self.pts_shape = pts.shape
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


idx = (num_knots - 5) * 10
dataset = BSplineDataset(range(idx, idx + 10), path, degree=degree)
num_points, dim = dataset.pts_shape

n_train = int(0.7 * len(dataset))
n_test  = len(dataset) - n_train
train_set, test_set = random_split(dataset, [n_train, n_test],
                                   generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False)

# --------------------------------------------------

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
            nn.Softmax(dim=1)       # ensure intervals are positive and sum to 1; dim=1 applies softmax across the correct dimension for batch processing
        )
        self.stack2 = nn.Sequential(
            nn.Linear(num_intervals, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_intervals),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.stack1(x)          # pass through first stack of layers
        x = self.stack2(x)          # pass through second stack of layers
        # intervals_to_knots only handles 1D; prepend zero column and cumsum manually
        zero = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        x = torch.cumsum(torch.cat((zero, x), dim=1), dim=1)  # (batch, num_knots)
        return x


num_neurons = 128                   # number of neurons in each hidden layer
# create model instance and move to device
model = NN(num_knots, num_neurons, degree).to(device)

# print architecture and number of parameters to file
summary_path = os.path.join(output_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write("\nModel architecture:\n")
    f.write(str(model) + "\n\n")
    
    f.write(f"\nInput dimension:  {num_knots - 1}  (intervals between {num_knots} knots)")
    f.write(f"\nOutput dimension: {num_knots - 1}  (predicted intervals, softmax-normalized)")
    f.write(f"\nB-spline degree:  {degree}")
    
    f.write("\n\nLayer shapes (weight, bias):\n")
    for name, param in model.named_parameters():
        f.write(f"  {name}: {list(param.shape)}\n")
    
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write(f"\nTotal parameters:     {total:>,.0f}")
    f.write(f"\nTrainable parameters: {trainable:>,.0f}")
    f.write(f"\nFrozen parameters:    {total - trainable:>,.0f}")

# --------------------------------------------------

# Training loop to optimize the neural network parameters to minimize the B-spline fitting loss.

def train(model, train_loader, param="uniform",
          num_epochs=100, tol=1e-6, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)     # method to update model parameters based on computed gradients
    train_losses = []     # list to store loss values after each epoch
    
    degree = model.degree
    
    # loop until reaching maximum number of epochs or the error is below the specified tolerance
    for epoch in range(num_epochs):
        batch_losses = []
        # loop through batches of data points from the training set
        # labels ignored; physics inspired loss only depends on the input data points and predicted knots, not the true knots
        for pts_batch, _ in tqdm(train_loader, ncols=100, desc=f"Epoch {epoch}"):
            pts_batch = pts_batch.to(device)        # (batch, num_points*dim)
        
            # predict final knots from the model given the input data points
            pred_knots = model(pts_batch)           # (batch, num_knots)
        
            batch_loss = torch.tensor(0.0, dtype=precision, device=device)
            # loop through each sample in the batch
            for j in range(pts_batch.shape[0]):
                points = pts_batch[j].reshape(num_points, dim)
                t_grid = make_grid(points, method=param)

                # pad the internal knots to open/clamped knots
                # to avoid numerical errors from the cumsum + softmax, we throw away the first and last predicted knots and replace them with exact 0 and 1
                full_knots = torch.cat((
                    torch.zeros(degree + 1, dtype=pred_knots.dtype, device=pred_knots.device),
                    pred_knots[j, 1:-1],      # interior knots only, guaranteed in (0, 1)
                    torch.ones( degree + 1, dtype=pred_knots.dtype, device=pred_knots.device)
                ))
        
                # compute the B-spline basis matrix evaluated at the parameter grid for the predicted knots and given degree
                basis_matrix = bspline_basis_matrix(t_grid, full_knots, degree, soft=False, k=1000.0)
                # solve for control points that best fit the data points given the basis matrix (least squares)
                controls = solve_control_points(basis_matrix, points, reg=1e-6)

                # compute loss as sum of distances (norm2) between data points and points on the B-spline curve
                loss = torch.sum((points - basis_matrix @ controls) ** 2)
                batch_loss += loss
        
            batch_loss = batch_loss / pts_batch.shape[0]   # mean over batch
            batch_losses.append(batch_loss.item())
            
            batch_loss.backward()       # compute gradients of the batch loss with respect to model parameters using backpropagation
            optimizer.step()            # update model parameters based on computed gradients
            optimizer.zero_grad()       # reset gradients to zero for the next batch
        
        # compute loss statistics across all batches for the epoch
        mean_loss = np.mean(batch_losses)
        max_loss = np.max(batch_losses)
        min_loss = np.min(batch_losses)
        std_loss = np.std(batch_losses)
        train_losses.append((mean_loss, max_loss, min_loss, std_loss))

        print(f"Mean: {mean_loss:>4,.8f},  Max: {max_loss:>4,.8f},  Min: {min_loss:>4,.8f},  Std: {std_loss:>4,.8f}\n")
        
        # check for convergence: if the loss is below the specified tolerance, stop training
        if epoch > 0 and mean_loss < tol:
            print(f"Converged at epoch {epoch}")
            break
    
    if epoch == num_epochs - 1:
        print("Reached maximum epochs without convergence.")

    return train_losses


model_file = os.path.join(output_dir, "model.pth")
losses_file = os.path.join(output_dir, "losses.npy")

# if a trained model file already exists, load it ...
if os.path.exists(model_file):
    print(f"Loading model from {model_file}")
    # load saved model parameters from file
    model.load_state_dict(torch.load(model_file, map_location=device))
    # load saved training losses from file
    train_losses = np.load(losses_file, allow_pickle=True)
# ... otherwise, train a new model and save it to file
else:
    # launch training
    train_losses = train(model, train_loader, param="uniform", 
                        num_epochs=100, tol=eps, lr=1e-3)
    # save trained model to file
    #torch.save(model.state_dict(), model_file)
    # save losses to file
    np.save(losses_file, np.array(train_losses, dtype=np.float64))



# --------------------------------------------------

# # print final results to file
# results_path = os.path.join(output_dir, "results.txt")
# with open(results_path, "w") as f:
#     f.write(f"Final error: {final_err:.16f}\n\n")
#     f.write(f"Degree: {degree}\n\n")
#     f.write(f"Knots:\n{final_knots}\n\n")
#     f.write(f"Controls:\n{controls}\n\n")

# # plot the training loss curve and the final curve fit to file
# plot_loss(losses, log=True, 
#             path = output_dir, name = "loss_curve.png")
# plot_curve_fit(points, final_knots, degree, controls, final_err, 
#             path = output_dir, name = "curve_fit.png")
