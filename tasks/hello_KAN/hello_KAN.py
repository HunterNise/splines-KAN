# https://kindxiaoming.github.io/pykan/intro.html


import os
# Change the working directory to the current file's directory to ensure that the figures are saved in the correct location
output_dir = os.path.join(os.path.dirname(__file__), "outputs-3")
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)

import sys
# Redirect all stdout and stderr to output.txt
_log = open("output.txt", "w")
sys.stdout = _log
sys.stderr = _log

import matplotlib.pyplot as plt

# import random
# import numpy as np
# import torch
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)


# Initialize KAN

from kan import *
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)

# Create dataset

# f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)
dataset['train_input'].shape, dataset['train_label'].shape

# Plot KAN at initialization

model(dataset['train_input']);  # forward pass to initialize the model and cache spline coefficients (necessary for plotting)
model.plot(beta=100)
plt.savefig("figures/model_plot0.png"); plt.close()

# Train KAN with sparsity regularization

model.fit(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);

# Plot trained KAN

model.plot()
plt.savefig("figures/model_plot1.png"); plt.close()

# Prune KAN and replot (keep the original shape)

model.prune()
model.plot()
plt.savefig("figures/model_plot2.png"); plt.close()

# Prune KAN and replot (get a smaller shape)

model = model.prune()           # re-assign the pruned model to get a smaller shape
model(dataset['train_input'])   # the new smaller model has never done a forward pass, so its activation cache is empty
model.plot()
plt.savefig("figures/model_plot3.png"); plt.close()

# Continue training and replot

model.fit(dataset, opt="LBFGS", steps=50);
model.plot()
plt.savefig("figures/model_plot4.png"); plt.close()

# Automatically or manually set activation functions to be symbolic

mode = "auto" # "manual"
if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin');
    model.fix_symbolic(0,1,0,'x^2');
    model.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)

# Continue training to almost machine precision

# grid updates should not run after edges have been fixed to symbolic functions
# singularity_avoiding prevent nan overflow
model.fit(dataset, opt="LBFGS", steps=50, 
            update_grid=False, singularity_avoiding=True);

# Obtain the symbolic formula

print("Symbolic formula:", model.symbolic_formula()[0][0])
