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



= Title

= Motivation

#slide(
  config: config-page(margin: (top: -0.5cm)),
)[
  #image(
    "figures/deepmind-intro.png", height: 13cm
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
    "figures/deepmind-framework.png"
  )
  
  #v(0.5em)

  #set text(size: 0.9em)
  #alternatives()[
    #grid(columns: (50%, 50%))[
      $quad z quad half "mathematical object"$ \
      $X(z) quad "features/input"$ \
      $Y(z) quad "target/output"$

      $f : X(z) mapsto Y(z) quad "relationship"$
    ][
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
  ][
    $quad z quad = "convex polyhedra"$ \
    $X(z) = ("#vertices", "#edges", "volume", "surface area") in ZZ times ZZ times RR times RR$ \
    $Y(z) = ("#faces") in ZZ$

    Euler's formula: $V - E + F = 2 quad arrow.squiggly quad X(z) dot (-1,1,0,0) + 2 = Y(z)$
  ]
]

#slide[
  #place(top, dy: -1cm)[
    #image(
      "figures/deepmind-knots1.png", height: 7cm
    )
  ]
  
  #place(bottom, dy: 1cm)[
    #image(
      "figures/deepmind-knots2.png", height: 8cm
    )
  ]
]

#slide[
  #alternatives()[
    #place(top, dx: -1.5cm, dy: -1.5cm)[
      #image(
        "figures/kan-intro.png", height: auto
      )
    ]

    #place(right, dy: 5cm)[
      #box(width: 5cm)[
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
      KAN: Kolmogorov-Arnold Networks. https://arxiv.org/abs/2404.19756
    ]
  ][
    #place(top, dx: 2cm, dy: 1cm)[
      #image(
        "figures/kan2-intro.png", height: auto
      )
    ]

    #place(left, dy: 2cm)[
      #box(width: 5cm)[
        AI \
        \+ \
        Science

        #v(2em)

        NO \
        reverse \
        engineering
      ]
    ]

    #place(bottom + right, dx: -2cm, dy: 1.0em)[
      #set text(fill: gray, size: 0.5em)
      Z. Liu, P. Ma, Y. Wang, W. Matusik, M. Tegmark \
      KAN 2.0: Kolmogorov-Arnold Networks Meet Science. https://arxiv.org/abs/2408.10205
    ]
  ]
]

#slide(
  config: config-page(margin: (top: -0.5cm)),
)[
  #image(
    "figures/kan-knots.png", height: 13cm
  )
]


= Problem

== B-Spline approximation

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
    = min norm( bold(P) - B bold(C) )_F^2
  $
  equivalent to the normal equation $wide (B^T B) bold(C) = B^T bold(P) wide$ \
  where $B$ depends on $bold(U)$ and $bold(T)$
  #place(bottom)[
    #set text(size: 0.75em)
    $
      B := mat(
        B_0^p (t_1), ..., B_n^p (t_1);
        dots.v, dots.down, dots.v;
        B_0^p (t_N), ..., B_n^p (t_N);
      ) in RR^(N times (n+1))
    , wide
      bold(C) = vec(bold(C)_0, dots.v, bold(C)_n) in RR^((n+1) times d)
    , wide
      bold(P) = vec(bold(P)_1, dots.v, bold(P)_N) in RR^(N times d)
    $
  ]
]

