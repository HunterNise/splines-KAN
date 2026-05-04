import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

from source.functions import *


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

# Load points from text file

path = "/app/data/SplinegenDataset/2d_train.npz"

num_knots = 6       # number of knots (without repetitions/clamping)
num_points = 100    # number of data points sampled from the B-spline curve


class BSplineDataset(Dataset):
    def __init__(self, path, num_knots, num_points=100):
        self.samples = []

        with np.load(path) as data:
            knots       = data['knots']
            ctrl_pts    = data['ctrl_pts']
            degree      = data['degree']

        self.degree = int(degree)
        self.dim = ctrl_pts.shape[2]

        
        # filter curves whose knot count matches num_knots
        
        d = self.degree
        # formula derived from the padded representation:
        #   count_nonzero counts the (num_knots-2) interior knots plus (d+1) trailing ones
        knot_count = np.count_nonzero(knots, axis=1) + (d + 1) - 2 * d
        mask = knot_count == num_knots      # bool (N_ds,)

        knots    = knots[mask]              # (N, max_knots_padded)
        ctrl_pts = ctrl_pts[mask]           # (N, max_ctrl, dim)

        # actual (unpadded) sizes for this num_knots
        full_knot_len   = num_knots + 2 * d         # full clamped knot vector length
        n_ctrl          = num_knots + d - 1         # number of control points

        
        # resample curves at num_points uniform parameter values in [0, 1]
        
        t_grid = torch.linspace(0.0, 1.0, num_points, dtype=torch.float64)

        # loop over samples
        for i in range(len(ctrl_pts)):
            # discard trailing zeros from the padded representation and convert to torch tensors
            full_knots  = torch.tensor(knots[i, :full_knot_len], dtype=torch.float64)
            ctrls       = torch.tensor(ctrl_pts[i, :n_ctrl],     dtype=torch.float64)   # (n_ctrl, dim)

            # evaluate B-spline at num_points parameter values
            B   = bspline_basis_matrix(t_grid, full_knots, d, soft=False)       # (num_points, n_ctrl)
            pts = B @ ctrls                                                     # (num_points, dim)

            # interior knots as labels (exclude the d+1 leading zeros and d+1 trailing ones)
            interior_knots = full_knots[d + 1 : -(d + 1)]       # (num_knots - 2,)

            self.samples.append((pts.reshape(-1), interior_knots, t_grid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


print(f"Loading dataset ...")
dataset = BSplineDataset(path, num_knots, num_points)
degree  = dataset.degree        # degree of the B-spline curve
dim     = dataset.dim           # dimension of the data points (2 for 2D, 3 for 3D)

n_train = int(0.8 * len(dataset))
n_test  = len(dataset) - n_train
train_set, test_set = random_split(dataset, [n_train, n_test],
                                   generator=torch.Generator().manual_seed(0))

batch_size   = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

# --------------------------------------------------

# Define the neural network model to approximate the mapping
#   from input data points to a final knot vector that minimizes the B-spline error.

from .model import NN


num_neurons = 256                   # number of neurons in each hidden layer
dropout     = 0.1                   # dropout probability applied after each hidden ReLU
# create model instance and move to device
model = NN(num_points, dim, num_knots, num_neurons, degree, dropout).to(device)

# print architecture and number of parameters to file
summary_path = os.path.join(output_dir, "summary.txt")
print(f"Saving model summary ...")
with open(summary_path, "w") as f:
    f.write("\nModel architecture:\n")
    f.write(str(model) + "\n\n")
    
    f.write(f"\nInput dimension:  {model.num_points * model.dim} = {model.num_points} * {model.dim}  (flattened data points)")
    f.write(f"\nOutput dimension: {model.num_knots} (knots without clamping)")
    f.write(f"\nB-spline degree:  {model.degree}")
    
    f.write("\n\nLayer shapes (weight, bias):\n")
    for name, param in model.named_parameters():
        f.write(f"  {name}: {list(param.shape)}\n")
    
    total     = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    f.write(f"\nTotal parameters:     {total:>,.0f}")
    f.write(f"\nTrainable parameters: {trainable:>,.0f}")
    f.write(f"\nFrozen parameters:    {total - trainable:>,.0f}")

# --------------------------------------------------

# Training loop to optimize the neural network parameters to minimize the B-spline fitting loss.

def compute_epoch_loss(model, loader, beta):
    """Evaluate mean/max/min/std loss over a data loader without gradient updates."""
    batch_losses = []
    degree = model.degree
    model.eval()
    with torch.no_grad():
        for points_batch, labels_batch, params_batch in loader:
            points_batch = points_batch.to(device)
            labels_batch = labels_batch.to(device)
            params_batch = params_batch.to(device)
            
            pred_knots = model(points_batch)   # (batch, num_knots)

            batch_loss = torch.tensor(0.0, dtype=precision, device=device)
            batch_len = points_batch.shape[0]
            for j in range(batch_len):
                points = points_batch[j].reshape(num_points, dim)
                t_grid = params_batch[j]

                full_knots = torch.cat((
                    torch.zeros(degree + 1, dtype=pred_knots.dtype, device=pred_knots.device),
                    pred_knots[j, 1:-1],
                    torch.ones( degree + 1, dtype=pred_knots.dtype, device=pred_knots.device)
                ))

                basis_matrix = bspline_basis_matrix(t_grid, full_knots, degree, soft=False, k=1000.0)
                controls     = solve_control_points(basis_matrix, points, reg=1e-6)

                physics_loss    = torch.sum((points - basis_matrix @ controls) ** 2)
                supervised_loss = F.mse_loss(pred_knots[j, 1:-1], labels_batch[j])
                batch_loss     += physics_loss + beta * supervised_loss

            batch_losses.append((batch_loss / batch_len).item())
    model.train()
    return batch_losses


def train(model, train_loader, test_loader,
          num_epochs=100, tol=1e-6, lr=1e-3, beta=0.0, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    train_losses = []
    test_losses  = []

    num_points = model.num_points
    dim        = model.dim
    degree     = model.degree

    best_test_mean  = float("inf")
    best_state      = None
    patience_counter = 0

    # loop until reaching maximum number of epochs or early stopping
    for epoch in range(num_epochs):
        batch_losses = []
        model.train()
        # loop through batches of data points from the training set
        for points_batch, labels_batch, params_batch in tqdm(train_loader, ncols=100, desc=f"Epoch {epoch}"):
            points_batch = points_batch.to(device)         # (batch, num_points*dim)
            labels_batch = labels_batch.to(device)
            params_batch = params_batch.to(device)

            # predict final knots from the model given the input data points
            pred_knots = model(points_batch)               # (batch, num_knots)

            batch_loss = torch.tensor(0.0, dtype=precision, device=device)
            # loop through each sample in the batch
            batch_len = points_batch.shape[0]
            for j in range(batch_len):
                points = points_batch[j].reshape(num_points, dim)
                t_grid = params_batch[j]

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

                # compute physics inspired loss as sum of distances (norm2) between data points and points on the B-spline curve
                physics_loss = torch.sum((points - basis_matrix @ controls) ** 2)
                # MSE loss between predicted interior knots and true interior knots
                supervised_loss = F.mse_loss(pred_knots[j, 1:-1], labels_batch[j])
                # total loss for the sample is a combination of physics loss and supervised loss, weighted by a hyperparameter beta to balance the two components
                batch_loss += physics_loss + beta * supervised_loss

            batch_loss = batch_loss / batch_len   # mean over batch
            batch_losses.append(batch_loss.item())

            batch_loss.backward()       # compute gradients of the batch loss with respect to model parameters using backpropagation
            optimizer.step()            # update model parameters based on computed gradients
            optimizer.zero_grad()       # reset gradients to zero for the next batch

        # compute train loss statistics across all batches for the epoch
        train_losses.append((
            np.mean(batch_losses), np.max(batch_losses),
            np.min(batch_losses),  np.std(batch_losses)
        ))

        # evaluate on test set
        test_batch_losses = compute_epoch_loss(model, test_loader, beta)
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

        scheduler.step(test_mean)   # adjust lr based on test loss

        # early stopping: save best model and track patience
        if test_mean < best_test_mean - tol:
            best_test_mean   = test_mean
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)   # restore best weights

    return train_losses, test_losses


model_file        = os.path.join(output_dir, "model.pth")
train_losses_file = os.path.join(output_dir, "train_losses.npy")
test_losses_file  = os.path.join(output_dir, "test_losses.npy")
training_file     = os.path.join(output_dir, "training_info.txt")

# train the model and save it to file

num_epochs  = 300       # maximum number of epochs to train for
lr          = 1e-3      # learning rate
beta        = 0.5       # supervised loss weight; set to 0.0 to train with physics loss only
patience    = 15        # early stopping patience

train_losses, test_losses = train(
    model, train_loader, test_loader,
    num_epochs=num_epochs, tol=eps, lr=lr, beta=beta, patience=patience
)

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
    f.write(f"  Number of samples: {len(dataset)}\n")
    f.write(f"  Number of training samples: {len(train_set)}\n")
    f.write(f"  Number of test samples: {len(test_set)}\n")

    f.write("\nTraining configuration:\n")
    f.write(f"  Dropout: {dropout}\n")
    f.write(f"  Batch size: {batch_size}\n")
    f.write(f"  Number of epochs: {num_epochs}\n")
    f.write(f"  Learning rate: {lr:.e}\n")
    f.write(f"  Beta (supervised loss weight): {beta}\n")
    f.write(f"  Patience (early stopping): {patience} epochs\n")

    f.write("\nTraining information and final results:\n")
    f.write(f"  Training completed in {len(train_losses)} epochs.\n")
    f.write(f"  Final train loss: {train_losses[-1][0]:>.6f}\n")
    f.write(f"  Final test loss:  {test_losses[-1][0]:>.6f}\n")


# plot training and test loss curves
plot_train_losses(train_losses, log=True,
                  path=output_dir, name="train_losses.png")
plot_train_losses(test_losses,  log=True,
                  path=output_dir, name="test_losses.png")

# combined plot: train vs test mean loss
import matplotlib.pyplot as plt
epochs = np.arange(len(train_losses))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, np.array(train_losses)[:, 0], label="Train mean", color="steelblue", linewidth=2)
ax.plot(epochs, np.array(test_losses)[:, 0],  label="Test mean",  color="tomato",    linewidth=2)
ax.set_xlabel("Epoch")
ax.set_yscale("log")
ax.set_ylabel("Loss (log scale)")
ax.set_title("Train vs Test Loss")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "train_vs_test_losses.png"))
plt.close(fig)
