Dataset for

> Pascal Laube, Matthias O. Franz, Georg Umlauf. \
> Deep Learning Parametrization for B-Spline Curve Approximation. \
> https://arxiv.org/abs/1807.08304

available at \
http://www.ios.htwg-konstanz.de/parnetdatasets

---

Every two lines correspond to a point cloud:
$$
 \vdots \\
 x_1, x_2, ..., x_n \\
 y_1, y_2, ..., y_n \\
 \vdots
$$

From the article:

> §5.3 \
> [...] As for the input size we define $l = 100$. [...] we chose to synthesize the data using B-spline curves. We generate random control points $c_i$ using a normal distribution with mean $\mu$ and variance $\sigma$ to define B-spline curves of degree $k = 3$ with $(k + 1)$-fold end-knots and no interior knots. For the $y$-coordinates, we use $\sigma = 2$ and $\mu = 10$. For the $x$-coordinates, we use $\sigma = 1$ and $\mu = 10$ for the first control point and increase $\mu$ by $\Delta \mu = 1$ for all consecutive control points. Curves with self-intersections are discarded, [...]. Using this approach, we generate a dataset consisting of $150.000$ curves. Then, we sample $l$ points $p = (p_0, . . . , p_{l−1})$ along each curve. Since these curves tend to have increasing $x$-coordinates from left to right we add index-flipped versions of the point sequences to the dataset resulting in $300.000$ point sequences of which $20%$ are used as test data in the training process.
>
> [...] For the evaluation we generated four evaluation sets:
> - *Evaluation set 1* contains $500$ curves computed as described in Section 5.3. We sample $500$ equidistributed (in terms of arc length) points on each curve.
> - *Evaluation set 2* contains the curves from evaluation set 1 but sampled at random parameters.
> - *Evaluation set 3* contains $500$ curves computed as described in Section 5.3 but with random interior knots without multiplicities. We generate $3$ to $8$ random interior knots which results in a set of very diverse curves, some of high complexity. We sample $500$ equidistributed points on each curve.
> - *Evaluation set 4* contains the curves from evaluation set 3 but sampled at random parameters.
