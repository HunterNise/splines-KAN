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
        # 00_knot.txt 11 3 5
        # 01_knot.txt 11 3 5
        # 02_knot.txt 11 3 5
        # 03_knot.txt 11 3 5
        # 04_knot.txt 11 3 5
        # 05_knot.txt 11 3 5
        # 06_knot.txt 11 3 5
        # 07_knot.txt 11 3 5
        # 08_knot.txt 11 3 5
        # 09_knot.txt 11 3 5
        # 10_knot.txt 12 3 6
        # 11_knot.txt 12 3 6
        # 12_knot.txt 12 3 6
        # 13_knot.txt 12 3 6
        # 14_knot.txt 12 3 6
        # 15_knot.txt 12 3 6
        # 16_knot.txt 12 3 6
        # 17_knot.txt 12 3 6
        # 18_knot.txt 12 3 6
        # 19_knot.txt 12 3 6
        # 20_knot.txt 13 3 7
        # 21_knot.txt 13 3 7
        # 22_knot.txt 13 3 7
        # 23_knot.txt 13 3 7
        # 24_knot.txt 13 3 7
        # 25_knot.txt 13 3 7
        # 26_knot.txt 13 3 7
        # 27_knot.txt 13 3 7
        # 28_knot.txt 13 3 7
        # 29_knot.txt 13 3 7
        # 30_knot.txt 14 3 8
        # 31_knot.txt 14 3 8
        # 32_knot.txt 14 3 8
        # 33_knot.txt 14 3 8
        # 34_knot.txt 14 3 8
        # 35_knot.txt 14 3 8
        # 36_knot.txt 14 3 8
        # 37_knot.txt 14 3 8
        # 38_knot.txt 14 3 8
        # 39_knot.txt 14 3 8
        # 40_knot.txt 15 3 9
        # 41_knot.txt 15 3 9
        # 42_knot.txt 15 3 9
        # 43_knot.txt 15 3 9
        # 44_knot.txt 15 3 9
        # 45_knot.txt 15 3 9
        # 46_knot.txt 15 3 9
        # 47_knot.txt 15 3 9
        # 48_knot.txt 15 3 9
        # 49_knot.txt 15 3 9
        # 50_knot.txt 16 3 10
        # 51_knot.txt 16 3 10
        # 52_knot.txt 16 3 10
        # 53_knot.txt 16 3 10
        # 54_knot.txt 16 3 10
        # 55_knot.txt 16 3 10
        # 56_knot.txt 16 3 10
        # 57_knot.txt 16 3 10
        # 58_knot.txt 16 3 10
        # 59_knot.txt 16 3 10
        # 60_knot.txt 17 3 11
        # 61_knot.txt 17 3 11
        # 62_knot.txt 17 3 11
        # 63_knot.txt 17 3 11
        # 64_knot.txt 17 3 11
        # 65_knot.txt 17 3 11
        # 66_knot.txt 17 3 11
        # 67_knot.txt 17 3 11
        # 68_knot.txt 17 3 11
        # 69_knot.txt 17 3 11
        # 70_knot.txt 18 3 12
        # 71_knot.txt 18 3 12
        # 72_knot.txt 18 3 12
        # 73_knot.txt 18 3 12
        # 74_knot.txt 18 3 12
        # 75_knot.txt 18 3 12
        # 76_knot.txt 18 3 12
        # 77_knot.txt 18 3 12
        # 78_knot.txt 18 3 12
        # 79_knot.txt 18 3 12
        # 80_knot.txt 19 3 13
        # 81_knot.txt 19 3 13
        # 82_knot.txt 19 3 13
        # 83_knot.txt 19 3 13
        # 84_knot.txt 19 3 13
        # 85_knot.txt 19 3 13
        # 86_knot.txt 19 3 13
        # 87_knot.txt 19 3 13
        # 88_knot.txt 19 3 13
        # 89_knot.txt 19 3 13
        # 90_knot.txt 20 3 14
        # 91_knot.txt 20 3 14
        # 92_knot.txt 20 3 14
        # 93_knot.txt 20 3 14
        # 94_knot.txt 20 3 14
        # 95_knot.txt 20 3 14
        # 96_knot.txt 20 3 14
        # 97_knot.txt 20 3 14
        # 98_knot.txt 20 3 14
        # 99_knot.txt 20 3 14
