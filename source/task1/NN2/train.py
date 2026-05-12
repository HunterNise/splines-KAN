"""
[task1/NN2] Dataset-level B-spline knot predictor — parameter file, dropout, checkpointing.

Productionizes the NN1 training pipeline with better software engineering. Changes vs task1/NN1:
- All hyperparameters read from train.prm (deal.II parameter file convention: subsection/set).
- Model architecture factored out to model.py; both model.py and train.prm are copied to
  outputs/ at the start so each run is fully reproducible from its output folder alone.
- Dropout regularization added after each hidden ReLU (probability read from parameter file).
- Checkpoint/resume: training state (epoch, optimizer, scheduler, losses, patience counter,
  best weights) written to checkpoint.pth every checkpoint_interval epochs; automatically
  reloaded if training is restarted after interruption; checkpoint deleted on clean completion.
- Dataset yields a precomputed t_grid as a third element alongside pts and interior_knots.
"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

from source.functions import *


import os
import shutil
import importlib.util

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

# Parse parameter file

prm_file = os.path.join(os.path.dirname(__file__), "train.prm")
prm      = PrmParser().parse(prm_file)

# save a copy of the parameter file in the output folder
shutil.copy2(prm_file, os.path.join(output_dir, "train.prm"))

# save a copy of the model architecture file in the output folder
arch_file = os.path.join(os.path.dirname(__file__), "model.py")
shutil.copy2(arch_file, os.path.join(output_dir, "model.py"))

# --------------------------------------------------

# Load points from text file

path        = prm.get("Dataset", "Path")                    # path to the training dataset
full_path   = os.path.join(ROOT, path)                      # prepend ROOT to get absolute path

num_knots   = prm.get_int("Dataset", "Number of knots")     # number of knots (without repetitions/clamping)
num_points  = prm.get_int("Dataset", "Number of points")    # number of data points sampled from the B-spline curve


print("Loading dataset ...")
dataset = BSplineDataset(full_path, num_knots, num_points)
degree  = dataset.degree        # degree of the B-spline curve
dim     = dataset.dim           # dimension of the data points (2 for 2D, 3 for 3D)

n_train = int(0.8 * len(dataset))
n_test  = len(dataset) - n_train
train_set, test_set = random_split(dataset, [n_train, n_test],
                                   generator=torch.Generator().manual_seed(0))

batch_size   = prm.get_int("Training", "Batch size")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

# --------------------------------------------------

# Define the neural network model to approximate the mapping
#   from input data points to a final knot vector that minimizes the B-spline error.

# load class from file model.py in the same directory
_spec = importlib.util.spec_from_file_location("model", os.path.join(os.path.dirname(__file__), "model.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
NN = _mod.NN


num_neurons = prm.get_int("Model", "Number of neurons")     # number of neurons in each hidden layer
dropout     = prm.get_float("Model", "Dropout")             # dropout probability applied after each hidden ReLU

# create model instance and move to device
model = NN(num_points, dim, num_knots, num_neurons, degree, dropout).to(device)


# print architecture and number of parameters to file
summary_path = os.path.join(output_dir, "summary.txt")
print("Saving model summary ...")
with open(summary_path, "w") as f:
    f.write("\nModel architecture:\n")
    f.write(str(model) + "\n\n")
    
    f.write(f"\nInput dimension:  {model.num_points * model.dim} = {model.num_points} * {model.dim}  (flattened data points)")
    f.write(f"\nOutput dimension: {model.num_knots - 1} (intervals without clamping)")
    f.write(f"\nB-spline degree:  {model.degree}")
    
    f.write("\n\nLayer shapes (weight, bias):\n")
    for name, param in model.named_parameters():
        f.write(f"  {name:<15} : {list(param.shape)}\n")
    
    total     = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    f.write(f"\nTotal parameters:     {total:>9,d}")
    f.write(f"\nTrainable parameters: {trainable:>9,d}")
    f.write(f"\nFrozen parameters:    {total - trainable:>9,d}")

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
            
            pred_knots = model(points_batch)    # (batch, num_knots)

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
          num_epochs=100, tol=1e-6, lr=1e-3, beta=0.0, patience=10,
          checkpoint_file=None, checkpoint_interval=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    train_losses = []
    test_losses  = []

    num_points = model.num_points
    dim        = model.dim
    degree     = model.degree

    best_test_mean   = float("inf")
    best_state       = None
    patience_counter = 0
    start_epoch      = 0

    # resume from checkpoint if one exists (unfinished previous run)
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
        print(f"  Resumed at epoch {start_epoch} (best test mean so far: {best_test_mean:.6f})")

    # loop until reaching maximum number of epochs or early stopping
    for epoch in range(start_epoch, num_epochs):
        batch_losses = []
        model.train()
        # loop through batches of data points from the training set
        for points_batch, labels_batch, params_batch in tqdm(train_loader, ncols=100, desc=f"Epoch {epoch}"):
            points_batch = points_batch.to(device)          # (batch, num_points*dim)
            labels_batch = labels_batch.to(device)          # (batch, num_knots-2)
            params_batch = params_batch.to(device)          # (batch, num_points)

            # predict final knots from the model given the input data points
            pred_knots = model(points_batch)                # (batch, num_knots)

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

            batch_loss = batch_loss / batch_len     # mean over batch
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

        # save checkpoint so training can be resumed if interrupted
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
            }, checkpoint_file)

    if best_state is not None:
        model.load_state_dict(best_state)   # restore best weights

    # remove checkpoint on clean completion — the final model.pth takes over
    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return train_losses, test_losses

# --------------------------------------------------

model_file        = os.path.join(output_dir, "model.pth")
train_losses_file = os.path.join(output_dir, "train_losses.npy")
test_losses_file  = os.path.join(output_dir, "test_losses.npy")
training_file     = os.path.join(output_dir, "training_info.txt")
checkpoint_file   = os.path.join(output_dir, "checkpoint.pth")          # temporary; deleted on clean completion


num_epochs          = prm.get_int("Training", "Number of epochs")       # maximum number of epochs to train for
lr                  = prm.get_float("Training", "Learning rate")        # learning rate
beta                = prm.get_float("Training", "Beta")                 # supervised loss weight; set to 0.0 to train with physics loss only
patience            = prm.get_int("Training", "Patience")               # early stopping patience
checkpoint_interval = prm.get_int("Training", "Checkpoint interval")    # save a checkpoint to disk every this many epochs (1 = every epoch, 0 = disable)

# train the model and get training/test loss curves
train_losses, test_losses = train(
    model, train_loader, test_loader,
    num_epochs=num_epochs, tol=eps, lr=lr, beta=beta, patience=patience,
    checkpoint_file=checkpoint_file, checkpoint_interval=checkpoint_interval
)


# save model and training information to files for later evaluation and plotting
torch.save({
    'model_state_dict'  : model.state_dict(),
    'num_points'        : num_points,
    'dim'               : dim,
    'num_knots'         : num_knots,
    'num_neurons'       : num_neurons,
    'degree'            : degree,
    'dropout'           : dropout,
}, model_file)

# save training and test losses as numpy arrays for later plotting
np.save(train_losses_file, np.array(train_losses, dtype=np.float64))
np.save(test_losses_file,  np.array(test_losses,  dtype=np.float64))

# save training information and final results to a text file
with open(training_file, "w") as f:
    f.write("\nDataset information:\n")
    f.write(f"  Path: {path}\n")
    f.write(f"  {num_points} points {dim}D per curve, {num_knots} knots, degree {degree}\n")
    f.write(f"  Number of (total) samples:  {len(dataset):>7,d}\n")
    f.write(f"  Number of training samples: {n_train:>7,d}\n")
    f.write(f"  Number of test samples:     {n_test:>7,d}\n")

    f.write("\nModel hyperparameters:\n")
    f.write(f"  Dropout: {dropout}\n")

    f.write("\nTraining configuration:\n")
    f.write(f"  Batch size: {batch_size}\n")
    f.write(f"  Number of epochs: {num_epochs}\n")
    f.write(f"  Learning rate: {lr}\n")
    f.write(f"  Beta (supervised loss weight): {beta}\n")
    f.write(f"  Patience (early stopping): {patience} epochs\n")

    f.write("\nTraining information and final results:\n")
    f.write(f"  Training completed in {len(train_losses)} epochs.\n")
    f.write(f"  Final train loss: {train_losses[-1][0]:>.6f}\n")
    f.write(f"  Final test loss:  {test_losses[-1][0]:>.6f}\n")
    f.write(f"  Best test loss:   {min(test_losses, key=lambda x: x[0])[0]:>.6f}\n")


# plot training and test loss curves
plot_train_losses(train_losses, log=True,
                  path=output_dir, name="train_losses.png")
plot_train_losses(test_losses,  log=True,
                  path=output_dir, name="test_losses.png")

# combined plot: train vs test mean loss
plot_train_and_test_losses(train_losses, test_losses, log=True,
                           path=output_dir, name="train_vs_test_losses.png")
