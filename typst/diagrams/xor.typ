#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.5.1"

#set page(width: auto, height: auto, margin: 5mm, fill: white)
#set text(font: "Libertinus Serif", size: 14pt)

// --------------------------------------------------

// MLP diagram

#let circle(pos, body, name: none, 
            radius: 4mm, fill: white, stroke: 0.8pt + luma(55%)
) = node(
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


#let draw-xor-mlp = diagram(
  edge-stroke: 0.85pt + luma(55%),

  circle((0, 0), [$x_1$], name: <x1>, fill: luma(72%)),
  circle((0, 1), [$x_2$], name: <x2>, fill: luma(72%)),
  circle((1, 0), [$h_1$], name: <h1>),
  circle((1, 1), [$h_2$], name: <h2>),
  circle((1.75, 0.5), [$y$], name: <y>),
  
  edge(<x1>, <h1>,  text(size: 0.75em)[$1$], "--|>", mark-scale: 75%, label-side:  left, label-sep: 0.1em,),
  edge(<x2>, <h2>,  text(size: 0.75em)[$1$], "--|>", mark-scale: 75%, label-side: right, label-sep: 0.1em,),
  edge(<x1>, <h2>,  text(size: 0.75em)[$1$], "--|>", mark-scale: 75%, label-side:  left, label-sep: -0.15em, label-pos: 20%,),
  edge(<x2>, <h1>,  text(size: 0.75em)[$1$], "--|>", mark-scale: 75%, label-side: right, label-sep: -0.15em, label-pos: 20%,),
  edge(<h1>,  <y>, text(size: 0.75em)[$-1$], "--|>", mark-scale: 75%, label-side:  left, label-sep: -0.15em, label-pos: 35%,),
  edge(<h2>,  <y>,  text(size: 0.75em)[$1$], "--|>", mark-scale: 75%, label-side: right, label-sep: -0.05em, label-pos: 45%,),
  edge(<y>, (2.2, 0.5), "-|>", mark-scale: 75%),

  textnode((rel: (0pt, -6mm), to: <h1>), text(size: 0.55em, fill: red)[$bold(-1.5)$]),
  textnode((rel: (0pt, -6mm), to: <h2>), text(size: 0.55em, fill: red)[$bold(-0.5)$]),
  textnode((rel: (0pt, -6mm), to: <y>), text(size: 0.55em, fill: red)[$bold(-0.5)$]),

  textnode((rel: (3.5mm, 6mm), to: <h1>), text(size: 0.55em, weight: "bold")[AND]),
  textnode((rel: (3.5mm, 6mm), to: <h2>), text(size: 0.55em, weight: "bold")[OR]),
)


// --------------------------------------------------

// Plots

// Clip the decision boundary w0 + w1*x + w2*y = 0 to a bounding box.
// Returns an array of 0, 1, or 2 endpoint coordinates.
#let clip-boundary(weights, 
                    xmin: -0.35, xmax: 1.4, 
                    ymin: -0.35, ymax: 1.4) = {
  let (w0, w1, w2) = (weights.at(0), weights.at(1), weights.at(2))
  let pts = ()

  // Vertical edges (x = const) — use ≤/≥ to include corners
  if w2 != 0 {
    for x in (xmin, xmax) {
      let y = (-w0 - w1 * x) / w2
      if y >= ymin and y <= ymax { pts += ((x, y),) }
    }
  }
  // Horizontal edges (y = const) — use strict < > to avoid double-counting corners
  if w1 != 0 {
    for y in (ymin, ymax) {
      let x = (-w0 - w2 * y) / w1
      if x > xmin and x < xmax { pts += ((x, y),) }
    }
  }

  pts
}


#let draw-xor-plots = cetz.canvas({
  import cetz.draw: *

  scale(2)

  // anchor canvas size regardless of boundary position
  //rect((-0.35, -0.35), (1.4, 1.4), stroke: none, fill: none)


  // first plot
  
  group({
    // axis
    group({
      set-style(mark: (end: "stealth", fill: black))
      line((0, 0), (1.5, 0))
      line((0, 0), (0, 1.5))
      content((1.52, -0.25), text(size: 12pt)[$x_1$], align: center)
      content((-0.25, 1.52), text(size: 12pt)[$x_2$], align: center)
    })

    // circles
    for comb in ((0, 0), (0, 1), (1, 0), (1, 1),) {
      circle(comb, radius: 0.05, fill: luma(72%))
    }

    // labels
    content((rel: (-0.15, -0.15), to: (0, 0)), text(size: 12pt)[$-$], align: center)
    content((rel: (-0.15, 0.15), to: (0, 1)), text(size: 12pt)[$+$], align: center)
    content((rel: (0.15, -0.15), to: (1, 0)), text(size: 12pt)[$+$], align: center)
    content((rel: (0.15, 0.15), to: (1, 1)), text(size: 12pt)[$-$], align: center)

    // decision boundary segments
    for weights in ((-1.5, 1, 1), (-0.5, 1, 1)) {
      let seg = clip-boundary(weights)
      if seg.len() >= 2 {
        line(seg.at(0), seg.at(1), stroke: 0.75pt + luma(25%))
      }
    }
  })


  // arrow
  line((1.55, 0.65), (2.05, 0.65), stroke: 4pt + luma(45%), mark: (end: "stealth", fill: none, scale: 2))


  // second plot

  let dx = 2.35

  group({
    translate(x: dx)

    // axis
    group({
      set-style(mark: (end: "stealth", fill: black))
      line((0, 0), (1.5, 0))
      line((0, 0), (0, 1.5))
      content((1.52, -0.25), text(size: 12pt)[$h_1$], align: center)
      content((-0.25, 1.52), text(size: 12pt)[$h_2$], align: center)
    })

    // circles
    for comb in ((0, 0), (0, 1), (1, 1),) {
      circle(comb, radius: 0.05, fill: luma(72%))
    }

    // labels
    content((rel: (-0.15, -0.15), to: (0, 0)), text(size: 12pt)[$-$], align: center)
    content((rel: (-0.15, 0.15), to: (0, 1)), text(size: 12pt)[$+$], align: center)
    content((rel: (0.15, 0.15), to: (1, 1)), text(size: 12pt)[$-$], align: center)

    // decision boundary segments
    for weights in ((-0.5, -1, 1),) {
      let seg = clip-boundary(weights)
      if seg.len() >= 2 {
        line(seg.at(0), seg.at(1), stroke: 0.75pt + luma(25%))
      }
    }
  })


  // ---- Blue arrow between plots ----

  let p1 = (1, 0)        // point in left plot (data coords)
  let p2 = (dx + 0, 1)   // corresponding point in right plot
  
  // ellipses
  group({
    rotate(origin: p1, z: -45deg)
    translate(x: 0.07)
    cetz.draw.circle(p1, radius: (0.3, 0.2), stroke: (paint: blue, dash: "dashed"))
  })
  group({
    rotate(origin: p2, z: -45deg)
    translate(x: -0.07)
    cetz.draw.circle(p2, radius: (0.3, 0.2), stroke: (paint: blue, dash: "dashed"))
  })

  // arrow
  bezier(
    (p1.at(0) + 0.05, p1.at(1) + 0.20),  // start
    (p2.at(0) - 0.30, p2.at(1) + 0.05),  // end
    (p1.at(0) + 0.35, p2.at(1) + 0.15),  // control point
    stroke: (paint: blue, thickness: 0.75pt),
    mark: (end: "stealth", fill: blue, scale: 0.75),
  )

})


// --------------------------------------------------

// Main

#grid(columns: 2, column-gutter: 10mm)[
  #draw-xor-mlp
][
  #draw-xor-plots
]
