import os
import numpy as np

path = "/app/data/DNN-Solver/bspline-data/knot/"

for filename in sorted(os.listdir(path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(path, filename)
        knots = np.loadtxt(file_path)

        reps = 0
        for i in range(1, len(knots)):
            if knots[i] == knots[i-1]:
                reps += 1
            else:
                break
        print(filename, len(knots), reps, len(knots) - 2*reps)
