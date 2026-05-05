""" Explore the Splinegen dataset. """


import numpy as np

path = "data/SplinegenDataset/2d_train.npz"
# 2d_train; 2d_eval; (3d_train; 3d_eval)

with np.load(path) as data:
    print(data.files)
    # ['ctrl_pts', 'ctrl_pts_len', 'knots', 'points', 'params', 'points_len', 'degree']

    ctrl_pts    = data['ctrl_pts']
    points      = data['points']
    params      = data['params']
    knots       = data['knots']
    degree      = data['degree']

print("ctrl_pts shape:" , ctrl_pts.shape)   # (541570, 24, 2); (6310, 24, 2)
print("points shape:"   , points.shape)     # (541570, 50, 2); (6310, 50, 2)
print("params shape:"   , params.shape)     # (541570, 50)   ; (6310, 50)
print("knots shape:"    , knots.shape)      # (541570, 28)   ; (6310, 28)
print("degree:"         , degree)           # 3


num_knots = np.count_nonzero(knots, axis=1) + (degree+1) - 2*degree
values, counts = np.unique(num_knots, return_counts=True)
for v, c in zip(values, counts):
    print(f"num_knots: {v}, count: {c}")
    # num_knots:  2, count: 34660; 412
    # num_knots:  3, count: 34673; 408
    # num_knots:  4, count: 34632; 412
    # num_knots:  5, count: 34676; 411
    # num_knots:  6, count: 34680; 410
    # num_knots:  7, count: 34688; 410
    # num_knots:  8, count: 34664; 410
    # num_knots:  9, count: 34688; 409
    # num_knots: 10, count: 34680; 410
    # num_knots: 11, count: 34631; 412
    # num_knots: 12, count: 34536; 410
    # num_knots: 13, count: 34418; 410
    # num_knots: 14, count: 34578; 400
    # num_knots: 15, count: 34329; 388
    # num_knots: 16, count: 30380; 328
    # num_knots: 17, count: 15621; 168
    # num_knots: 18, count:  8385;  75
    # num_knots: 19, count:  1738;  14
    # num_knots: 20, count:   601;  10
    # num_knots: 21, count:   233;   2
    # num_knots: 22, count:    79;   1
