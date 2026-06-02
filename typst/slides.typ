#import "@preview/touying:0.6.1": *
#import themes.simple: *
#show: simple-theme.with(
  aspect-ratio: "16-9",
  header: []
)

#import "/template/template.typ": *
#show: rules

// hack for disabling counters
#let c = none
#let definition = definition.with(counter: c)
#let theorem = theorem.with(counter: c)
#let proposition = proposition.with(counter: c)
#let lemma = lemma.with(counter: c)
#let corollary = corollary.with(counter: c)



= #text(size: 40pt)[B-Spline Approximation via Kolmogorov--Arnold Networks]

= Motivation

#slide(
  config: config-page(margin: (top: -0.5cm)),
)[
  #image(
    "images/deepmind-intro.png", height: 13cm
  )
  
  #place(bottom, dy: 1.5em)[
    #set text(fill: gray, size: 0.5em)
    Davies, A., Veličković, P., Buesing, L. et al. Advancing mathematics by guiding human intuition with AI. \
    Nature 600, 70–74 (2021). https://doi.org/10.1038/s41586-021-04086-x
  ]
]

#slide(
  config: config-page(margin: (top: -0.5cm)),
)[
  #image(
    "images/deepmind-framework.png"
  )
  #v(0.5em)

  #set text(size: 0.9em)

  #let left-col = [
    $quad z quad half "mathematical object"$ \
    $X(z) quad "features/input"$ \
    $Y(z) quad "target/output"$

    $f : X(z) mapsto Y(z) quad "relationship"$
  ]
  #let right-col = [
    $hat(f) quad quad quad quad "model"$ \
    $hat(f)(X(z)) quad "prediction"$
    
    #place(left, dy: 1em)[
      #box(width: 20cm)[
        #set text(size: 0.75em)
        training #h(1.1em) = iteratively construct more accurate $hat(f)$ \
        supervised = $Y(z)$ available \
        attribution = which features are important for prediction
      ]
    ]
  ]

  #only("1")[
    #grid(columns: (50%, 50%))[#left-col][#hide(right-col)]
  ]
  #only("2")[
    $quad z quad = "convex polyhedra"$ \
    $X(z) = ("#vertices", "#edges", "volume", "surface area") in ZZ times ZZ times RR times RR$ \
    $Y(z) = ("#faces") in ZZ$

    Euler's formula: $V - E + F = 2 quad arrow.squiggly quad X(z) dot (-1,1,0,0) + 2 = Y(z)$
  ]
  #only("3")[
    #grid(columns: (50%, 50%))[#left-col][#right-col]
  ]
]

#slide[
  #place(top, dy: -1cm)[
    #image(
      "images/deepmind-knots1.png", height: 7cm
    )
  ]
  
  #place(bottom, dy: 1cm)[
    #image(
      "images/deepmind-knots2.png", height: 8cm
    )
  ]
]

#slide[
  #alternatives()[
    #place(top, dx: -1.5cm, dy: -1.5cm)[
      #image(
        "images/kan-intro.png", height: auto
      )
    ]

    #place(right, dy: 3cm)[
      #box(width: 5cm)[
        different architecture

        smaller interpretable models

        uses splines
      ]
    ]
    #place(bottom, dy: -1.75cm)[
      interpretable = understand model / predictions / decisions
    ]

    #place(bottom, dy: 1.0em)[
      #set text(fill: gray, size: 0.5em)
      Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, M. Tegmark \
      KAN: Kolmogorov--Arnold Networks. https://arxiv.org/abs/2404.19756
    ]
  ][
    #place(top, dx: 2cm, dy: 1cm)[
      #image(
        "images/kan2-intro.png", height: auto
      )
    ]

    #place(left, dy: 2cm)[
      #box(width: 5cm)[
        AI \ \+ \ Science
        #v(2em)
        NO \ reverse \ engineering
      ]
    ]

    #place(bottom + right, dx: -2cm, dy: 1.0em)[
      #set text(fill: gray, size: 0.5em)
      Z. Liu, P. Ma, Y. Wang, W. Matusik, M. Tegmark \
      KAN 2.0: Kolmogorov--Arnold Networks Meet Science. https://arxiv.org/abs/2408.10205
    ]
  ]
]

#slide(
  config: config-page(margin: (top: -0.5cm)),
)[
  #image(
    "images/kan-knots.png", height: 13.5cm
  )
]


= Problem

== B-Spline approximation <problem>

#slide[
  data points : ${bold(P)_i}_(i=1)^N subset.eq RR^d$

  $ bold(S)(t) = sum_(j=0)^n B_j^p (t) dot bold(C)_j $
  
  - spline: $bold(S)(t) in RR^d$
  - control points: ${bold(C)_j}_(j=0)^n subset.eq RR^d, quad n := m-p-1$
  - parametrization: ${t_i}_(i=1)^N subset.eq RR$ such that $bold(S)(t_i) approx bold(P)_i$
  - knots: ${u_h}_(h=0)^m subset.eq RR$ non-decreasing
  
  #pagebreak()
  
  - basis functions (degree $p$): $B_j^p (t)$ $->$ Cox--de Boor recursion formula:
  $
    &B_j^0 (t) := cases(
      thin 1 thin \, quad u_j <= t <= u_(j+1) ,
      thin 0 thin \, quad "otherwise"
    )
  \
    &B_j^p (t) := 
    vfrac(t - u_j, u_(j+p) - u_j) B_j^(p-1) (t) + 
    vfrac(u_(j+p+1) - t, u_(j+p+1) - u_(j+1)) B_(j+1)^(p-1) (t) 
    thick , quad j = 0, ..., underbrace(m-p-1, n)
  $
]

==

#slide[
  given knots $bold(U) = (u_0, ..., u_m)$ and parametrization $bold(T) = (t_1, ..., t_N)$, \
  compute optimal control points $bold(C)$ by solving
  $
    min sum_(i=1)^N norm( bold(P)_i - bold(S)(t_i) )_2^2 
    = min norm( bold(P_#hide[i]) - B bold(C) )_F^2
  $
  
  #pause
  equivalent to the normal equation $wide (B^T B) bold(C) = B^T bold(P) wide$ \
  where $B$ depends on $bold(U)$ and $bold(T)$
  
  #place(bottom)[
    #set text(size: 0.8em)
    $
      B := mat(
        B_0^p (t_1), ..., B_n^p (t_1);
        dots.v, dots.down, dots.v;
        B_0^p (t_N), ..., B_n^p (t_N);
      ) in RR^(N times (n+1))
    , quad
      bold(C) = vec(bold(C)_0, dots.v, bold(C)_n) in RR^((n+1) times d)
    , quad
      bold(P) = vec(bold(P)_1, dots.v, bold(P)_N) in RR^(N times d)
    $
  ]
]


= (Vanilla) Neural Network \ #text(fill: navy, size: 0.55em)[a.k.a. fully-connected feedforward network or multi-layer perceptron]

== Neurons

#slide[
  #place(dx: -0.25em, dy: 0.5em)[
    #set text(size: 0.75em)
    #grid(columns: (50%, 50%), column-gutter: 5mm)[
      #image(
        "images/nn-neuron-bio.png"
      )
      #v(-1em)
      Biological neuron:
      - dendrites: receive signals from other neurons
      - soma: integrate incoming signals
      - if signal exceeds threshold $->$ spike
      - axon: transmit output signal to other neurons
    ][
      #image(
        "diagrams/perceptron.svg"
      )
      #v(-1em)
      Artificial neuron (perceptron):
      - inputs: $x_1, ..., x_n$ (features)
      - weights: $w_1, ..., w_n$ (learned parameters)
      - bias: $b$ (learned parameter)
      - activation function: $sigma$ (non-linear function)
      - output: $y = sigma(w_1 x_1 + ... + w_n x_n + b)$
    ]
  ]
]

== Single-Layer Perceptron

#slide[
  Perceptron / Linear Threshold Unit (LTU): $f(bold(x)) = h(bold(w) dot bold(x) + b)$ \
  where #text(baseline: 2pt)[$half 
  h(z) = cases(
    thin 1 thin \, quad z thick > thin 0 ,
    thin 0 thin \, quad "otherwise"
  ) half$] is the Heaviside step function

  #pause
  it's equivalent to a linear classifier: \
  $wide quad thick f(bold(x)) = 1 thick$ if $thin bold(w) dot bold(x) + b > 0 thick$ else $f(bold(x)) = 0$ \
  decision boundary: $half bold(w) dot bold(x) + b = 0 thick$ (a hyperplane in $RR^n$)

  #pause
  #place(bottom, dx: 0.5cm, dy: 1.5cm)[
    #grid(columns: (auto, 8cm), column-gutter: 0.75em, align: center + top)[
      #alternatives(
        image("diagrams/slp1.svg", height: 6.5cm),
        image("diagrams/slp2.svg", height: 6.5cm),
        image("diagrams/slp3.svg", height: 6.5cm),
      )
    ][
      #v(2.5em)
      #alternatives[
        #text(weight: "bold")[AND]
      ][
        #text(weight: "bold")[OR]
      ][
        #text(weight: "bold")[XOR]

        #text(size: 0.85em)[NOT linearly separable!]
      ]
    ]
  ]
]

== XOR problem

#slide[
  #v(-0.5em)
  
  #grid(columns: (auto, auto, 6.5cm), column-gutter: 1.5em, align: horizon)[
    #set text(size: 0.75em)
    A single-layer perceptron cannot learn the XOR function \ because the XOR function is not linearly separable.
  ][
    $-->$
  ][
    #set align(right)
    Minsky & Papert, _Perceptrons_ (1969)
  ]
  #v(0.25em)

  #pause
  $x_1 plus.o x_2 = x_1 dot overline(x_2) + overline(x_1) dot x_2 = overline(\(underbrace(x_1 dot x_2, h_1)\)) dot \(underbrace(x_1 + x_2, h_2)\)$
  #place(right + horizon, dx: 0.65em, dy: -1.5em)[
    #set text(size: 0.8em)
    #grid(columns: 2, column-gutter: 0.75em)[
      $dot &= "AND" \
       +   &= "OR"$  
    ][
      $overline(x) &= "NOT" \
       plus.o      &= "XOR"$  
    ]
  ]

  #pause
  #place(bottom, dy: 1.5em)[
    #image("diagrams/xor.svg", height: 7.75cm)
  ]
]

== Shallow Multi-Layer Perceptron

#slide[
  #grid(columns: (50%, 50%), column-gutter: 5mm)[
    #alternatives(
      image("diagrams/mlp-shallow1.svg"),
      image("diagrams/mlp-shallow2.svg"),
      image("diagrams/mlp-shallow3.svg"),
    )
  ][
    #v(-0.75em)
    $ f(bold(x)) = sum_(i=1)^N_"hidden" a_i thin sigma(bold(w)_i dot bold(x) + b_i) $

    $ bold(x)   &= (x_1, ..., x_n) \
      bold(w)_i &= (w_(i 1), ..., w_(i n)) $
    #v(0.5em)

    #uncover("4-")[
      "_Universal approximation theorem_" \ 
      #text(size: 16pt)[
        a single hidden layer with #underline[sufficiently many neurons] can 
        approximate any continuous function on compact subsets of $RR^n$ to any desired accuracy, 
        given appropriate activation functions (e.g., sigmoid, ReLU)
      ]
    ]
  ]
]

== Deep Multi-Layer Perceptron

#slide[
  #grid(columns: (55%, auto), column-gutter: 2mm)[
    #image("diagrams/mlp-deep.svg")
  ][
    $ bold(f) (bold(x)) = (bold(W)_L compose sigma_(L-1) compose ... compose sigma_1 compose bold(W)_1)(bold(x)) $
    #v(0.75em)

    $ half bold(x) half = (x_1, ..., x_n) $

    $ bold(W)_l = mat(
      w_(1 1)^((l)), ..., w_(1 N_(l-1))^((l));
      dots.v, dots.down, dots.v;
      w_(N_l 1)^((l)), ..., w_(N_l N_(l-1))^((l));
    ) $

    #align(right)[$N_l$ : \# neurons in layer $l$]
  ]
]

== (Supervised) Learning / Training

#slide[
  *how to find weights?* $quad$ #text(size: 0.8em)[(don't know the true $f$, want to "learn" it from data)]

  #pause
  $bold(f)_bold(theta)$ : statistical model, parametric function [network] \
  $bold(theta)$ : parameters [weights and biases] \
  ${(bold(x)_i, bold(y)_i)}_(i=1)^N$ : (supervised) dataset, (input, label) pairs \
  $hat(bold(y)) = bold(f)_bold(theta) (bold(x))$ : prediction \
  $cal(L) (bold(y), hat(bold(y)))$ : loss function, #text(size: 0.9em)[measures error between true label and prediction]

  #pause
  *Idea*: adjust parameters $bold(theta)$ to minimize error on training data
  #place(bottom + center, dy: 0.5em)[
    $ min_(bold(theta) in Theta) sum_(i=1)^N cal(L) (bold(y)_i, bold(f)_bold(theta) (bold(x)_i)) $
  ]
]

#slide[
  training $<-->$ optimization over parameter space
  
  #pause
  #place(top + center, dx: 5cm, dy: 2.5cm)[
    #curve(
      curve.move((0.75cm, -0.1cm)),
      curve.cubic(
        (3cm, 0cm),
        (0cm, -1.5cm), (3cm, -1.75cm)
      )
    )
  ]
  #place(top + right)[
    very complicated \
    non-linear/convex \
    high-dimensional
  ]

  #pause
  if $f_bold(theta)$ is differentiable w.r.t. $bold(theta)$, we can use gradient descent

  #v(-0.5em)
  $ bold(theta)^(k+1) := bold(theta)^k - eta thin nabla_(#h(-5pt) bold(theta)) cal(E)(bold(theta^k)) thick , wide 
  cal(E)(bold(theta)) = 1/N sum_(i=1)^N cal(L) (bold(y)_i, bold(f)_bold(theta) (bold(x)_i)) $
  #v(-0.65em)

  $eta$ : learning rate, step size for parameter updates
  #v(-0.5em)
  variants: SGD / mini-batch, momentum, adaptive lr etc.
  //#v(-0.5em)

  #pause
  *how to compute gradients?* \
  backpropagation: chain rule, but efficient (automatic differentiation)
]


= Kolmogorov--Arnold Network (KAN)

== Kolmogorov--Arnold representation theorem

#slide(repeat: 8, self => {
  let (uncover, only, alternatives) = utils.methods(self)
  [
    *Idea*: multivariate is complicated (except +), but univariate is easy! \
    $wide half$ separate variables as much as possible ($->$ easier derivatives)

    #pause
    $f(x,y) = phi.alt_1 (x) + phi.alt_2 (y) quad$ #text(size: 1.2em, fill: red)[#emoji.crossmark]
    
    #pause
    #[
      #show math.equation.where(block: true): it => align(left, it)
      $ #h(8cm) x y = - x^2/2 - y^2/2 + #alternatives(start: 3)[$(x+y)$][$z$]^2/2
      #only("4-")[
        #h(4cm) $z = x + y$
      ] $
    ]
    #v(0.5em)

    #uncover("5-")[
    #set text(size: 0.8em)
    // #let addend(i) = [
    //   #uncover("6-")[$Phi_#i ($] #uncover("5-")[$phi.alt_(#i,1) ($] $x_1$ #uncover("5-")[$)$] + #uncover("5-")[$phi.alt_(#i,2)($] $x_2$ #uncover("5-")[$)$] #uncover("6-")[$)$]
    // ]
    #let addend(i) = [
      #uncover("7-")[$Phi_#i ($]
        #alternatives(start: 5,)[
          $#hide[$phi.alt_(#i,1) ($] x_1 #hide[$)$] + #hide[$phi.alt_(#i,2)($] x_2 #hide[$)$]$
        ][
          $phi.alt_(#i,1)(x_1) + phi.alt_(#i,2)(x_2)$
        ][
          $phi.alt_(#i,1)(x_1) + phi.alt_(#i,2)(x_2)$
        ][
          $underbrace(phi.alt_(#i,1)(x_1) + phi.alt_(#i,2)(x_2), t_#i)$
        ]
        #uncover("7-")[$)$]
    ]
    $ f(x_1, x_2) &= addend(1) + addend(2) + ... + addend(k) $
    ]

    #place(bottom, dy: 1em)[
      #set text(fill: gray, size: 0.5em)
      The Kolmogorov-Arnold Theorem. Luis Serrano Academy. \ https://www.youtube.com/watch?v=nS2hnm0JRBk
    ]
  ]
})

#slide[
  #theorem[
    Any continuous function $f : [0,1]^n -> RR$ can be represented as
    #v(-0.45em)
    $ f(x_1, ..., x_n) = sum_(q=1)^(2n+1) Phi_q ( sum_(p=1)^n phi.alt_(q, p) (x_p) ) $
    #v(-0.45em)
    where $Phi_q : RR -> RR$ and $phi.alt_(q, p) : [0,1] -> RR$ are continuous functions.
  ]

  #pause
  #place(bottom, dx: -1em, dy: 2em)[
    #image("images/yt-bounded.png", height: 6cm)
  ]
]

== Kolmogorov--Arnold Network
//== Shallow KAN

#slide[
  #image("images/yt-comparison.png")

  #place(bottom, dy: 1em)[
      #set text(fill: gray, size: 0.5em)
      Kolmogorov-Arnold Networks (KANs) - What are they and how do they work?. Luis Serrano Academy. \ https://www.youtube.com/watch?v=myFtp5zMv8U
    ]
]

//== Problems

#slide[
  #place(top, dy: 1.25em)[
  #grid(columns: (50%, 1fr), column-gutter: 5mm, align: (left, center))[
    #v(0.75em)

    2 layers are enough for \ universal approximation!

    #uncover("2-")[
      but $phi.alt$ may be very "ugly" \
      (e.g., non-smooth, fractal-like) \
      #h(5em) $arrow.b$ \
      not practical for ML \
      (Girosi & Poggio, 1989)
      #v(1em)
    ]

    #uncover("3-")[*Idea*: go deeper!]
  ][
    #alternatives()[
      #image("images/yt-shallow.png", height: 12cm)
    ][
      #align(center)[
        #image("images/CantorEscalier-2.svg", height: 11cm)
        #v(-0.5em)
        #text(fill: gray, size: 0.5em)[https://commons.wikimedia.org/wiki/File:CantorEscalier-2.svg]
      ]
    ][
      #image("images/yt-deep.png", height: 13cm)
    ]
  ]]
]

//== Deep KAN

#slide[
  #grid(columns: (60%, 45%))[
    $bold(f) (bold(x)) = (bold(Phi)_L compose ... compose bold(Phi)_1) (bold(x))$

    $ bold(Phi)_l = mat(
      phi.alt^((l))_(1, 1) (dot), ..., phi.alt^((l))_(1, N_(l-1))(dot);
      dots.v, dots.down, dots.v;
      phi.alt^((l))_(N_(l), 1) (dot), ..., phi.alt^((l))_(N_(l), N_(l-1))(dot);
    ) $

    arbitrary width KAN layer
    #v(-0.5em)
    $ x^((l+1))_j = sum_(i=1)^N_l phi.alt^((l+1))_(j,i) \( x^((l))_i \) , quad j = 1, ..., N_(l+1) $
  ][
    #image("images/kan-notation.png", width: auto)
  ]

  #place(top + right, dy: 1em)[
    theorem $->$ $[n, 2n+1, 1]$ KAN
  ]
]

#slide[
  *how to parametrize $phi.alt$'s ?*

  use 1D B-splines:
  - flexible, can approximate \ any continuous function
  - efficient to compute and differentiate
  - need few parameters (\#ctrls = $G+k$)
  - can be "fine-grained"
  
  #place(bottom + right, dx: 1em)[
    #image("images/kan-spline.png", width: 12cm)
  ]
]

== Interpretability

#slide[
  #place(bottom, dx: -1em)[
    #image("images/kan-symbolic.png", height: 11cm)
  ]

  #place(top + right, dy: 0.5em)[
    sparse: penalize \ non-zero prms

    visualize: \
    transparency

    prune: smaller net
    #v(1em)

    symbolic setting: \
    i/o shift/scaling \
    $->$ fitting aff. prms \
    $y approx c f (a x + b) + d $
  ]
]


= Experiments

== Task #h(1fr) #text(weight: "regular", fill: green)[#link(<problem>)[$arrow.t$ problem]]

#slide[
  *Goal*: find optimal knot vector of a B-spline fitting data points

  #pause
  For simplicity
  - fixed degree $p=3$ (cubic B-splines)
  - fixed number of points, ordered
  - fixed number of knots
  - fixed parametrization (uniform)

  #pause
  #place(bottom, dy: 1em)[
    *What we want?* \
    - find relationships with features, e.g. curvature
    - accuracy is not the priority (but still important)
    - learn something about the problem
  ]

  #place(bottom + right, dx: -5cm, dy: -1.5cm)[
  #curve(
    curve.move((0.5cm, -0.1cm)),
    curve.cubic(
      (3cm, 0cm),
      (0cm, -1.5cm), (4cm, -1.75cm)
    )
  )]
  #place(bottom + right, dy: -1em)[
    #box(width: 5cm)[let's pretend we don't know this]
  ]
]

== DNN Solver

#slide[
  #v(-1em)
  #alternatives()[
    #image("diagrams/pipeline1.svg", width: 25cm)
  ][
    #image("diagrams/pipeline2.svg", width: 25cm)
    #v(-0.5em)
  ]
  
  unsupervised training: only data points, no "ground truth" knots \
  physics inspired loss : $norm( bold(P_#hide[i]) - B bold(C) )_F^2$

  #place(bottom, dy: 1em)[
    #set text(fill: gray, size: 0.5em)
    Zepeng Wen, Jiaqi Luo, Hongmei Kang.
    The deep neural network solver for B-spline approximation. \
    Computer-Aided Design, vol.169, 103668, 2024.
    https://doi.org/10.1016/j.cad.2023.103668
  ]
]

== Trained model

#slide[
  #image("diagrams/pipeline3.svg", width: 25cm)

  input: sampled B-spline curve + ground truth knots

  #place(bottom, dy: 1em)[
    #set text(fill: gray, size: 0.5em)
    Pascal Laube, Matthias O. Franz, Georg Umlauf.
    Deep Learning Parametrization for B-Spline Curve Approximation. \
    https://arxiv.org/abs/1807.08304
  ]
]

== Conclusions

#slide[
  #set list(spacing: 1em)
  - lukewarm results, better ML skills & more time could help
  - ML-driven development (no need for domain knowledge) \
    e.g. sliding window $quad$ #sym.crossmark curvature local feature \
    #hide[e.g. sliding window] $quad$ #sym.checkmark #box(baseline: 1.25em)[reduce number of input variables \ for more interpretable formulas]
  - AI + human collaboration also at meta level \ 
    (e.g. Copilot "auto-research")
]

== Future directions

#slide[
  - try other tasks (e.g. knot insertion/removal, variable number of knots, knots-parametrization alignment, surface fitting)
  - use actually good architectures, i.e. SplineGen

  #place(bottom)[
    #alternatives(
      image("images/splinegen-1.png", height: 8cm),
      image("images/splinegen-2.png", height: 8cm)
    )
  ]
  #only(2)[
    #place(horizon + right, dx: 1em, dy: -2.25em)[
      *how to make it \ interpretable???*
    ]
  ]

  #place(bottom, dy: 1.25em)[
    #set text(fill: gray, size: 0.5em)
    Qiang Zou, Lizhen Zhu, Jiayu Wu, and Zhijie Yang.
    SplineGen: Approximating unorganized points through generative AI. \
    Computer-Aided Design, vol.178, 103809, 2025.
    https://doi.org/10.1016/j.cad.2024.103809
  ]
]


== SSPA

#slide[
  #set list(marker: [#sym.checkmark])
  - docker + uv (docker-less setup on lab pc)
  - bash, ssh, tmux
  - git, pytest
  - github actions & releases
  - typst (compiler) $quad$ #sym.crossmark make (so fast, just recompile everything)

  #set list(marker: [#sym.crossmark])
  - doxygen/sphinx (comments/docstrings are enough)
  - parallel training (only 1 GPU)

  #pause
  Honorable mention: Copilot (in particular Claude Sonnet) \
  [RIP billing for request]
]
