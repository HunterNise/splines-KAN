"""
Post-processing script for visualizing final errors of different parameterizations.
Generates bar plots for each parameterization showing the final error per sample.
Usage:
    python post_process.py
Assumes that results files are located in "outputs-uniform", "outputs-chord_length", and "outputs-centripetal" directories,
 each containing "*-results.txt" files with a line formatted as "Final error: <value>".
Outputs:
    - "errors_per_file.png" in each output directory, showing a bar plot of final errors for each sample.
"""


import os
import glob
import numpy as np
import matplotlib.pyplot as plt


for param in ["uniform", "chord_length", "centripetal"]:
    output_dir = os.path.join(os.path.dirname(__file__), "outputs" + f"-{param}")

    errors = []
    names = []
    for results_file in glob.glob(os.path.join(output_dir, "*-results.txt")):
        with open(results_file) as f:
            first_line = f.readline().strip()
        # "Final error: 0.0556..."
        errors.append(float(first_line.split(":")[-1].strip()))
        names.append(os.path.basename(results_file).replace("-results.txt", ""))

    if not errors:
        print(f"No results files found in {output_dir}, skipping.")
        continue

    # Sort by name for easy cross-param comparison
    names_errors = sorted(zip(names, errors), key=lambda x: x[0])
    names_sorted, errors_sorted = zip(*names_errors)

    fig, ax = plt.subplots(figsize=(max(10, len(errors) * 0.15), 5))
    ax.bar(range(len(errors_sorted)), errors_sorted, edgecolor="black")
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=90, fontsize=6)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Final Error")
    ax.set_title(f"Final Errors per Sample — {param}")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "errors_per_file.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}  (n={len(errors)}, mean={np.mean(errors):.4f})")
