#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#set page(width: auto, height: auto, margin: (top: 6.5mm, rest: 5mm), fill: white)
#set text(font: "Libertinus Serif", size: 14pt)


#let circle(pos, body, name: none, 
            radius: 4.5mm, fill: white, stroke: 0.8pt + luma(55%)) = node(
  pos,
  body,
  name  : name,
  radius: radius,
  fill  : fill,
  stroke: stroke,
)

#let textnode(pos, body) = node(
  pos,
  body,
  fill  : none,
  stroke: none,
  inset : 0pt,
)

// draw threshold symbol
#let threshold(size: 14mm) = box(width: size, height: size)[
  #set align(bottom + left)
  #curve(
    stroke: (paint: black, thickness: 1.1pt),
    curve.move((14.29%, -25%)),
    curve.line((35.71%, 0%), relative: true),
    curve.line((0%  , -50%), relative: true),
    curve.line((35.71%, 0%), relative: true),
  )
]


#diagram(
  spacing: (18mm, 16mm),
  edge-stroke: 0.85pt + luma(55%),
  node-stroke: 0.80pt + luma(55%),
  edge-corner-radius: none,

  // Inputs
  circle((0, 0), [], name: <x1>, fill: luma(72%)),
  circle((0, 1), [], name: <x2>, fill: luma(72%)),
  circle((0, 3), [], name: <xn>, fill: luma(72%)),

  textnode((rel: (-0.4, 0), to: <x1>), [$x_1$]),
  textnode((rel: (-0.4, 0), to: <x2>), [$x_2$]),
  textnode((rel: (-0.4, 0), to: <xn>), [$x_n$]),
  textnode((0,2), [$dots.v$]),

  // Bias
  circle((1.95, -0.1), [], name: <x0>, radius: 0mm, fill: luma(72%)),
  textnode((rel: (-0.1, -0.25), to: <x0>), [$x_0 = 1$]),

  // Summation
  circle((2.4, 1.5), text(size: 2em)[$Sigma$], name: <acc>, radius: 7mm, fill: white),

  // Activation
  circle((4.5, 1.5), threshold(size: 12mm), name: <thr>, radius: 7mm, fill: white),

  // Output
  circle((6.5, 1.5), [], name: <out>, radius: 2.5mm, fill: white),

  // Arrows
  edge(<x1>, <acc>, "--", [$w_1$]),
  edge(<x2>, <acc>, "--", [$w_2$]),
  edge(<xn>, <acc>, "--", [$w_n$]),
  edge(<x0>, <acc>, "--", [$w_0 = b$]),
  edge(<acc>, <thr>, "->", 
    label-side: right, label-sep: 1.5em,
    [$"net" = sum_(i=0)^n w_i x_i$]
  ),
  edge(<thr>, <out>, "->", 
    label-side: right, label-sep: 1.5em,
    [$"out" = sigma("net")$]
  ),
)
