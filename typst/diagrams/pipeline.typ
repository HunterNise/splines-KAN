#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: (top: 4.5mm, rest: 5mm), fill: white)
#set text(font: "Libertinus Serif", size: 14pt)
#let R = 8mm


#let circle(pos, body, name: none, 
            radius: R, fill: white, stroke: 0.8pt + luma(55%)) = node(
  pos,
  body,
  name  : name,
  radius: radius,
  fill  : fill,
  stroke: stroke,
  shape: rect,
)

#let textnode(pos, body) = node(
  pos,
  body,
  fill  : none,
  stroke: none,
  inset : 0pt,
)

#let arrow(..args) = edge(
  ..args,
  marks: "->",
)


#let knots-icon(num, offsets: none) = cetz.canvas(length: R * 0.4, {
  import cetz.draw: *
  let r = 0.38
  for i in range(num) {
    let dx = if offsets == none { 0 } else { offsets.at(i) }
    line(
      (i + dx,                          r),
      (i - r * calc.sqrt(3) / 2 + dx, -r / 2),
      (i + r * calc.sqrt(3) / 2 + dx, -r / 2),
      close: true, fill: olive, stroke: none,
    )
  }
})

#let intervals-icon(num, offsets: none) = cetz.canvas(length: R * 0.45, {
  import cetz.draw: *
  for i in range(num) {
    let dx = if offsets == none { 0 } else { offsets.at(i) }
    circle((i + dx, 0), radius: 0.2, fill: white, stroke: 0.2pt + black, name: "c" + str(i))
  }
  for i in range(num - 1) {
    line("c" + str(i) + ".east", "c" + str(i + 1) + ".west", stroke: 1pt + black)
  }
})

#let network-icon(num_input, num_hidden, num_output) = cetz.canvas(length: R * 0.45, {
  import cetz.draw: *

  let col-spacing = 1.5
  let r = 0.25

  let get-ys(num) = {
    let dist = 2 / calc.ln(num + 1)
    let L = (num - 1) * dist
    range(num).map(i => i * dist - L / 2)
  }

  let input-ys  = get-ys(num_input)
  let hidden-ys = get-ys(num_hidden)
  let output-ys = get-ys(num_output)

  // Edges first so nodes render on top
  for iy in input-ys {
    for hy in hidden-ys {
      line((0, iy), (col-spacing, hy), stroke: 0.3pt + luma(65%))
    }
  }
  for hy in hidden-ys {
    for oy in output-ys {
      line((col-spacing, hy), (col-spacing * 2, oy), stroke: 0.3pt + luma(65%))
    }
  }

  // Nodes
  for y in input-ys {
    circle((0, y), radius: r, fill: luma(72%), stroke: 0.3pt + black)
  }
  for y in hidden-ys {
    circle((col-spacing, y), radius: r, fill: white, stroke: 0.3pt + black)
  }
  for y in output-ys {
    circle((col-spacing * 2, y), radius: r, fill: white, stroke: 0.3pt + black)
  }
})


#let _scatter-pts = (
  (-0.75, -0.10), 
  (-0.50, -0.60), 
  (-0.10, -0.45),
  ( 0.15,  0.20), 
  ( 0.40,  0.55), 
  ( 0.65,  0.75), 
  ( 0.85,  0.10),
)

#let points-icon() = cetz.canvas(length: R * 0.65, {
  import cetz.draw: *
  for p in _scatter-pts {
    circle(p, radius: 0.15, fill: luma(55%), stroke: none)
  }
})

#let plot-icon() = cetz.canvas(length: R * 0.75, {
  import cetz.draw: *
  // Slanted S: dips down-left, inflects at centre, rises to upper-right
  catmull(
    (-0.90, -0.25), (-0.40, -0.70), (0.0, 0.0), (0.40, 0.70), (0.90, 0.25),
    stroke: 1.2pt + blue.darken(10%),
    fill: none,
  )
  for p in _scatter-pts {
    circle(p, radius: 0.1, fill: luma(55%), stroke: none)
  }
})

#let curve-icon() = cetz.canvas(length: R * 0.75, {
  import cetz.draw: *
  catmull(
    (-0.90, -0.25), (-0.40, -0.70), (0.0, 0.0), (0.40, 0.70), (0.90, 0.25),
    stroke: 1.2pt + blue.darken(10%),
    fill: none,
  )
})



#diagram(
  spacing: (18mm, 16mm),
  edge-stroke: 0.85pt + luma(55%),
  node-stroke: 0.80pt + luma(55%),
  edge-corner-radius: none,

  circle((0, 0), knots-icon(4), name: <knots_in>),
  textnode((rel: (0, 0.5), to: <knots_in>), [uniform \ knots]),
  circle((1, 0), intervals-icon(4), name: <intervals_in>),
  textnode((rel: (0, 0.4), to: <intervals_in>), [intervals]),
  circle((2, 0), network-icon(2, 3, 1), name: <network>),
  textnode((rel: (0, 0.4), to: <network>), [network]),
  circle((3, 0), intervals-icon(4, offsets: (0, -0.2, 0.1, 0)), name: <intervals_out>),
  textnode((rel: (0, 0.4), to: <intervals_out>), [intervals]),
  circle((4, 0), knots-icon(4, offsets: (0, -0.2, 0.1, 0)), name: <knots_out>),
  textnode((rel: (0, 0.5), to: <knots_out>), [optimal \ knots]),
  circle((5, 0), curve-icon(), name: <curve>),
  textnode((rel: (0, 0.4), to: <curve>), [curve]),
  circle((6, 0), plot-icon(), name: <error>),
  textnode((rel: (0, 0.4), to: <error>), [error]),

  arrow(<knots_in>, <intervals_in>),
  arrow(<intervals_in>, <network>),
  arrow(<network>, <intervals_out>),
  arrow(<intervals_out>, <knots_out>),
  arrow(<knots_out>, <curve>),
  arrow(<curve>, <error>),
  
  // loop
  circle(<network>, [], radius: R * 1.3, fill: none, stroke: none, name: <hidden>),
  arrow(<hidden>, <hidden>, bend: -140deg, loop-angle: -270deg, label: [train on single sample], label-pos: 0.72),
  
  // backprop
  arrow((rel: (0, -0.5), to: <error>), (rel: (0, -0.5), to: <knots_in>), "dashed", label: [backprop]),

  // loss arrow
  circle((5.5, 1), points-icon(), name: <points>),
  textnode((6, 1), [points]),
  arrow(<points>, (rel: (0, -1), to: <points>), "dashed", label: [PI loss], label-pos: 0.17),
)
#pagebreak()


#diagram(
  spacing: (18mm, 16mm),
  edge-stroke: 0.85pt + luma(55%),
  node-stroke: 0.80pt + luma(55%),
  edge-corner-radius: none,

  circle((0, 0), points-icon(), name: <points>),
  textnode((rel: (0, 0.5), to: <points>), [points]),
  circle((2, 0), network-icon(2, 3, 1), name: <network>),
  textnode((rel: (0, 0.4), to: <network>), [network]),
  circle((3, 0), intervals-icon(4, offsets: (0, -0.2, 0.1, 0)), name: <intervals_out>),
  textnode((rel: (0, 0.4), to: <intervals_out>), [intervals]),
  circle((4, 0), knots-icon(4, offsets: (0, -0.2, 0.1, 0)), name: <knots_out>),
  textnode((rel: (0, 0.5), to: <knots_out>), [optimal \ knots]),
  circle((5, 0), curve-icon(), name: <curve>),
  textnode((rel: (0, 0.4), to: <curve>), [curve]),
  circle((6, 0), plot-icon(), name: <error>),
  textnode((rel: (0, 0.4), to: <error>), [error]),

  arrow(<points>, <network>),
  arrow(<network>, <intervals_out>),
  arrow(<intervals_out>, <knots_out>),
  arrow(<knots_out>, <curve>),
  arrow(<curve>, <error>),
  
  // loop
  circle(<network>, [], radius: R * 1.3, fill: none, stroke: none, name: <hidden>),
  arrow(<hidden>, <hidden>, bend: -140deg, loop-angle: -270deg, label: [train on single sample], label-pos: 0.74),
  
  // backprop
  arrow((rel: (0, -0.5), to: <error>), (rel: (0, -0.5), to: <points>), "dashed", label: [backprop]),

  // loss arrow
  arrow(
    (rel: (0, 0.7), to: <points>), 
    (rel: (0, 1.8), to: <points>), 
    (rel: (-0.5, 1.8), to: <error>), 
    (rel: (-0.5, 0), to: <error>), 
    "dashed", label: [PI loss], label-pos: 0.75
  ),
)
#pagebreak()


#diagram(
  spacing: (18mm, 16mm),
  edge-stroke: 0.85pt + luma(55%),
  node-stroke: 0.80pt + luma(55%),
  edge-corner-radius: none,

  circle((0.2, -0.2), points-icon(), name: <points3>),
  circle((0.1, -0.1), points-icon(), name: <points2>),
  circle((0, 0), points-icon(), name: <points>),
  textnode((rel: (0, 0.5), to: <points>), [points]),
  circle((2, 0), network-icon(2, 3, 1), name: <network>),
  textnode((rel: (0, 0.4), to: <network>), [network]),
  circle((3, 0), intervals-icon(4, offsets: (0, -0.2, 0.1, 0)), name: <intervals_out>),
  textnode((rel: (0, 0.4), to: <intervals_out>), [intervals]),
  circle((4, 0), knots-icon(4, offsets: (0, -0.2, 0.1, 0)), name: <knots_out>),
  textnode((rel: (0, 0.5), to: <knots_out>), [optimal \ knots]),
  circle((5, 0), curve-icon(), name: <curve>),
  textnode((rel: (0, 0.4), to: <curve>), [curve]),
  circle((6, 0), plot-icon(), name: <error>),
  textnode((rel: (0, 0.4), to: <error>), [error]),

  arrow(<points>, <network>),
  arrow(<network>, <intervals_out>),
  arrow(<intervals_out>, <knots_out>),
  arrow(<knots_out>, <curve>),
  arrow(<curve>, <error>),
  
  // backprop
  arrow((rel: (0, -0.5), to: <error>), (rel: (0, -0.5), to: <points>), "dashed", label: [backprop]),

  // loss arrow
  arrow(
    (rel: (0, 0.7), to: <points>), 
    (rel: (0, 1), to: <points>), 
    (rel: (-0.5, 1), to: <error>), 
    (rel: (-0.5, 0), to: <error>), 
    "dashed", label: [PI + supervised loss], label-pos: 0.5, label-side: right
  ),
)
