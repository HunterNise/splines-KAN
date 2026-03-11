# NOTE: torch implementations keep dtype/device and are differentiable by autograd


# ==================================================


# --- Knot/Interval Conversions ---


# converts knots vector to intervals vector
#   subtract adjacent knot values to get intervals between knots

def knots_to_intervals(knots):
    # knots: List | np.ndarray(shape=(m+1,), dtype=float)
    intervals = []
    for i in range(len(knots) - 1):
        intervals.append(knots[i + 1] - knots[i])
    return intervals

def knots_to_intervals_torch(knots: torch.Tensor) -> torch.Tensor:
    # knots: torch.Tensor(shape=(m+1,), dtype=torch.float32)
    return knots[1:] - knots[:-1]   # all but the first minus all but the last

# --------------------------------------------------

# converts intervals vector back to knots vector
#   prepend a 0 and do a cumulative sum

def intervals_to_knots(intervals):
    # intervals: List | np.ndarray(shape=(m,), dtype=float)
    knots = [0]
    for interval in intervals:
        knots.append(knots[-1] + interval)
    return knots

def intervals_to_knots_torch(intervals: torch.Tensor) -> torch.Tensor:
    # intervals: torch.Tensor(shape=(m,), dtype=torch.float32)
    zero = torch.zeros(1, dtype=intervals.dtype, device=intervals.device)
    return torch.cat((zero, torch.cumsum(intervals, dim=0)))


# ==================================================


# --- Parametrization ---


# generate parametrization vector according to uniform method
#   simply divide the interval [0, 1] into equal parts

def uniform_params(points):
    # points: List of Lists | np.ndarray(shape=(N, 2), dtype=float)
    return np.linspace(0.0, 1.0, len(points))

def uniform_params_torch(points: torch.Tensor) -> torch.Tensor:
    # points: torch.Tensor(shape=(N, 2), dtype=torch.float32)
    return torch.linspace(0., 1., len(points)).to(device)

# --------------------------------------------------

# generate parametrization vector according to chord length method
#   compute distances between consecutive points, then normalize cumulative distances to [0, 1]

def chord_length_params(points):
    # points: List of Lists | np.ndarray(shape=(N, 2), dtype=float)
    pts = np.asarray(points)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = d.sum()
    if L == 0:
        return uniform_params(points)
    cum = np.concatenate(([0.0], np.cumsum(d)))
    return cum / L

def chord_length_params_torch(points: torch.Tensor) -> torch.Tensor:
    # points: torch.Tensor(shape=(N, 2), dtype=torch.float32)
    distances = torch.sqrt(torch.sum((points[1:] - points[:-1]) ** 2, dim=1))
    cumulative_distances = torch.cumsum(distances, dim=0)
    total_distance = cumulative_distances[-1]
    return torch.cat((torch.tensor([0.]).to(device), cumulative_distances / total_distance))

# --------------------------------------------------

# generate parametrization vector according to centripetal method
#   compute distances between consecutive points, take square root, then normalize cumulative values to [0, 1]

def centripetal_params(points):
    # points: List of Lists | np.ndarray(shape=(N, 2), dtype=float)
    pts = np.asarray(points)
    d = np.sqrt(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    S = d.sum()
    if S == 0:
        return uniform_params(points)
    cum = np.concatenate(([0.0], np.cumsum(d)))
    return cum / S

def centripetal_params_torch(points: torch.Tensor) -> torch.Tensor:
    # points: torch.Tensor(shape=(N, 2), dtype=torch.float32)
    distances = torch.sqrt(torch.sum((points[1:] - points[:-1]) ** 2, dim=1))
    cumulative_distances = torch.cumsum(torch.sqrt(distances), dim=0)
    total_distance = cumulative_distances[-1]
    return torch.cat((torch.tensor([0.]).to(device), cumulative_distances / total_distance))

# --------------------------------------------------

# helper function to select parametrization method

def make_grid(points, method="uniform"):
    # points: List of 2D points, method: String
    if   method == "uniform":
        return uniform_params(points)
    elif method == "chord_length":
        return chord_length_params(points)
    elif method == "centripetal":
        return centripetal_params(points)
    else:
        raise ValueError(f"Unknown parameterization method: {method}")


# ==================================================


# --- B-spline evaluation ---

# evaluate B-spline curve along parametrization vector T given knots and degree
#   compute basis function values by Cox-de Boor recursion formula, ...
#   ... assemble the matrix, then multiply by control points vector to get curve points
#   this version is useful for training since it can be implemented with matrix operations and autograd



def bspline_eval_torch(basis_matrix: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    return basis_matrix @ controls

# --------------------------------------------------

# evaluate B-spline curve at parameter values t given knots and degree
#   uses de Boor algorithm to ...
#   this version is useful to sample the curve for plotting

