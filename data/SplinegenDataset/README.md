Dataset for

> Qiang Zou, Lizhen Zhu, Jiayu Wu, and Zhijie Yang. \
> SplineGen: Approximating unorganized points through generative AI. \
> Computer-Aided Design, vol.178, 103809, 2025. \
> https://doi.org/10.1016/j.cad.2024.103809

available at \
https://github.com/sgb1084864985/SplinegenDataset/


Code for generating the dataset available at \
https://github.com/Qiang-Zou/SplineGen/

---

After extracting, the file structure should be:
```
SplinegenDataset/
├── 2d_eval.npz
├── 2d_train.npz
├── 3d_eval.npz
└── 3d_train.npz
```

---

From the article:

> **Dataset.** A dataset of $500,000$ B-spline curves with their control points, knot vectors, and sampled points has been compiled. Self-intersecting curves were eliminated using a specialized detection program. For data processing, we normalize control points to $[0, 1]^3$ and utilize masked arrays for neural network training consistency, refer to Supplementary Material for more details. A test dataset of above 5,000 curves is generated in the same way.
