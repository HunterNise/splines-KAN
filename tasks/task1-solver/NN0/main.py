import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

# --------------------------------------------------

path = "/app/data/DNN-Solver/bspline-data/pts/spl_data00.txt"
points = np.loadtxt(path)

#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")


def knots_to_intervals(knots: torch.Tensor) -> torch.Tensor:
    # knots is assumed 1‑D; subtract shifted versions
    # keeps dtype/device and is differentiable
    return knots[1:] - knots[:-1]

def intervals_to_knots(intervals: torch.Tensor) -> torch.Tensor:
    # prepend a 0 and do a cumulative sum
    zero = torch.zeros(1, device=intervals.device, dtype=intervals.dtype)
    return torch.cat((zero, torch.cumsum(intervals, dim=0)))


class NN(nn.Module):
    def __init__(self, num_knots=10):
        super().__init__()
        self.num_knots = num_knots
        num_intervals = num_knots - 1
        self.linear1 = nn.Linear(num_intervals, 64)
        self.linear2 = nn.Linear(64, num_intervals)

    def forward(self, x):
        x = knots_to_intervals(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=0)
        x = intervals_to_knots(x)
        return x


model = NN(num_knots=10).to(device)
print(model)

# --------------------------------------------------

def uniform_params(points):
    return np.linspace(0.0, 1.0, len(points))

def chord_length_params(points):
    pts = np.asarray(points)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = d.sum()
    if L == 0:
        return uniform_params(points)
    cum = np.concatenate(([0.0], np.cumsum(d)))
    return cum / L

def centripetal_params(points):
    pts = np.asarray(points)
    d = np.sqrt(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    S = d.sum()
    if S == 0:
        return uniform_params(points)
    cum = np.concatenate(([0.0], np.cumsum(d)))
    return cum / S

def make_grid(points, method="uniform"):
    if method == "uniform":
        return uniform_params(points)
    elif method == "chord_length":
        return chord_length_params(points)
    elif method == "centripetal":
        return centripetal_params(points)
    else:
        raise ValueError(f"Unknown parameterization method: {method}")
    

def bspline_basis(t, knots, degree, soft=False, k=100.0):
    """
    Vectorized Cox-de Boor B-spline basis (autograd-friendly).

    Inputs
    ------
    t      : (N,) tensor (float), points where to evaluate, in same dtype/device as knots
    knots  : (K,) tensor (non-decreasing)
    degree : int >= 0
    soft   : if False -> hard indicator for degree 0 (exact, non-differentiable).
             if True  -> soft indicator using sigmoid(k * (x - left))*sigmoid(k * (right - x))
    k      : steepness for sigmoid when soft=True (larger k -> closer to hard step)

    Returns
    -------
    B : (N, m) where m = K - degree - 1  (same as your `(N, m+1)` notation depending on indexing)
    """
    t = t.to(knots.dtype)
    N = t.shape[0]
    K = knots.shape[0]
    m = K - 1                  # number of degree-0 intervals (K-1)
    if m <= 0:
        return torch.zeros((N, 0), dtype=t.dtype, device=t.device)

    # degree-0 basis: vectorized
    # lefts: knots[0..K-2], rights: knots[1..K-1]
    lefts = knots[:-1]         # shape (m,)
    rights = knots[1:]         # shape (m,)
    t_exp = t[:, None]         # (N,1)

    if not soft:
        # hard mask: intervals [left, right) except last interval includes right endpoint
        mask = (t_exp >= lefts) & (t_exp < rights)        # (N,m)
        # include right endpoint for last interval
        mask_last = (t == rights[-1]).unsqueeze(1)        # (N,1)
        mask = mask.clone()
        mask[:, -1] = mask[:, -1] | mask_last.squeeze(1)
        Bprev = mask.to(dtype=knots.dtype)
    else:
        # soft mask: use product of two sigmoids -> smooth approximation of indicator
        left_term  = torch.sigmoid(k * (t_exp - lefts))   # (N,m)
        right_term = torch.sigmoid(k * (rights - t_exp))  # (N,m)
        Bprev = (left_term * right_term)

    # recursion for p = 1..degree
    # At recursion p we should produce shape (N, K-p-1)
    for p in range(1, degree + 1):
        M = K - p - 1
        if M <= 0:
            return torch.zeros((N, 0), dtype=t.dtype, device=t.device)

        # slices for denominators:
        # denom1_j = knots[j+p]   - knots[j]      for j=0..M-1  => knots[p : p+M] - knots[:M]
        # denom2_j = knots[j+p+1] - knots[j+1]    for j=0..M-1  => knots[p+1 : p+1+M] - knots[1:M+1]
        denom1 = knots[p : p + M] - knots[:M]            # (M,)
        denom2 = knots[p + 1 : p + 1 + M] - knots[1 : M + 1]  # (M,)

        # numerators broadcast to shape (N, M)
        num1 = t_exp - knots[:M]                         # (N, M)
        num2 = knots[p + 1 : p + 1 + M] - t_exp          # (N, M)

        # safe division: compute coef = num / denom where denom != 0, else 0
        # denom1/2 are 1-D; expand to (N,M) via broadcasting
        # use unsqueezing to broadcast cleanly
        denom1_expand = denom1.unsqueeze(0)              # (1, M)
        denom2_expand = denom2.unsqueeze(0)              # (1, M)

        # avoid dividing by zero; torch.where keeps autograd path for denom != 0
        coef1 = torch.where(denom1_expand != 0,
                            num1 / denom1_expand,
                            torch.zeros_like(num1))
        coef2 = torch.where(denom2_expand != 0,
                            num2 / denom2_expand,
                            torch.zeros_like(num2))

        # Bprev has shape (N, previous_M = M+1). Use appropriate slices
        left_term  = coef1 * Bprev[:, :M]        # (N, M)
        right_term = coef2 * Bprev[:, 1 : M + 1] # (N, M)

        Bp = left_term + right_term              # (N, M)
        Bprev = Bp                               # allocate new tensor each degree

    return Bprev   # shape (N, K-degree-1)

def solve_control_points(Bmat, X, reg=1e-6):
    # Bmat: (N, M), X: (N, d)
    Bt = Bmat.transpose(0,1)             # (M, N)
    G = Bt @ Bmat                        # (M, M)
    M = G.shape[0]
    Greg = G + torch.eye(M, device=Bmat.device, dtype=Bmat.dtype) * reg
    rhs = Bt @ X                         # (M, d)
    C = torch.linalg.solve(Greg, rhs)    # (M, d)
    return C


def train(model, points, degree, param="uniform",
          max_iter=1000, tol=1e-6, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    # cast to float32 on creation
    points = torch.from_numpy(points).float().to(device)
    t_grid = torch.from_numpy(make_grid(points, method=param)).float().to(device)
    
    for iter in range(max_iter):
        initial_knots = torch.linspace(0., 1., model.num_knots).to(device)  # initial guess for knots
        final_knots = model(initial_knots)
        basis_matrix = bspline_basis(t_grid, final_knots, degree)
        controls = solve_control_points(basis_matrix, points)

        loss = torch.sum((points - basis_matrix @ controls) ** 2)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if iter % 100 == 0:
            print(f"Iteration {iter:>4n}, Loss: {loss.item():>8.6f}")
        
        if iter > 0 and losses[-1] < tol:
            print(f"Converged at iteration {iter}")
            break
    
    if iter == max_iter - 1:
        print("Reached maximum iterations without convergence.")

    return losses, final_knots.detach().cpu().numpy()


losses, final_knots = train(model, points, degree=3, param="uniform")
print("Final knots:", final_knots)

# --------------------------------------------------

# Plot losses
plt.figure(figsize=(8, 4))
plt.plot(losses, label="Training Loss")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
