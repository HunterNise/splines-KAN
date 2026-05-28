#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.5.1"

#set page(width: auto, height: auto, margin: 5mm, fill: white)
#set text(font: "Libertinus Serif", size: 14pt)

// --------------------------------------------------

// Logic table

// n-level flattening of nested arrays
#let flatten(arr, n: 1) = {
  // recursively apply 1-level flatten to arr n times
  range(n).fold(
    arr, 
    (a, _) => a.fold((), (acc, x) => acc + x)   // 1-level flatten
  )
}

// Cartesian product of arrays
#let cart-prod(..arrays) = arrays.pos().fold(
  ((),),
  (acc, arr) => flatten(
    // iterate over accumulated combinations
    acc.map(
      combo => arr.map(x => combo + (x,))   // append each element of arr to each combo
    )
  )
)

// Returns a stroke for table cells, where the inside edges are different from the frame edges.
#let frame(strk-frame, strk-inside) = (x, y) => (
  left: if x > 0 { strk-inside } else { strk-frame },
  right: strk-frame,    // overridden by left of the next cell
  top: if y <= 1 { strk-frame } else { strk-inside },
  bottom: strk-frame,   // overridden by top of the next row
)


#let logic-table(lambda, n) = {
  let inputs  = cart-prod(..range(n).map(_ => (0, 1)))
  let outputs = inputs.map(lambda)

  table(
    columns: n + 1,
    align: center,
    fill: (x, y) => if (y == 0) { luma(90%) } else { none },
    stroke: frame(0.8pt + black, 0.3pt + gray),
    table.vline(x: n, stroke: 0.8pt + black),

    table.header(
      ..range(n).map(i => $x_#(i+1)$),
      $f(bold(x))$
    ),
    
    ..flatten(
      inputs.zip(outputs).map( ((input, output)) =>
        input.map(v => [#v]) + ([#output],)
      )
    ),
  )
}

// --------------------------------------------------

// Single-layer perceptron diagram

#let circle(pos, body, name: none, 
            radius: 2.5mm, fill: white, stroke: 0.8pt + luma(55%)
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

// return coordinates for num nodes in a column col, with vertical spacing based on the number of nodes
#let get-coords(num, col) = {
  let dist = 1 / calc.ln(num+1)
  let L = (num - 1) * dist
  let y = range(num).map(i => i * dist - (L / 2))
  y.map(y => (col, y))
}

#let draw-neurons(coords, ..args) = {
  for (i, value) in coords.enumerate() {
    circle(value, ..args, 
      text(size: 9pt, baseline: -0.5pt)[$x_#(i+1)$],
    )
  }
}

// draw threshold symbol
#let threshold(
  size: 14mm,
  stroke: 1.1pt + black,
) = box(width: size, height: size)[
  #set align(bottom + left)
  #curve(
    stroke: stroke,
    curve.move((14.29%, -25%)),
    curve.line((35.71%, 0%), relative: true),
    curve.line((0%  , -50%), relative: true),
    curve.line((35.71%, 0%), relative: true),
  )
]


#let slp-diagram(
  n,
  weights: none,
) = {
diagram(debug: 0,

  spacing: (18mm, 16mm),
  edge-stroke: 0.85pt + luma(55%),
  node-stroke: 0.80pt + luma(55%),
  edge-corner-radius: none,

  let (inputs, bias) = get-coords(n+1, 0).chunks(n, exact: false),
  bias = bias.flatten(),
  let output = get-coords(1, 1).flatten(),

  draw-neurons(inputs, fill: luma(72%)),
  circle(bias, [$1$],  fill: none, stroke: none),
  circle(output, threshold(size: 4mm, stroke: 0.75pt), fill: white),

  for (j, input) in inputs.enumerate() {
    edge(input, output, "--|>", mark-scale: 75%,
          label-sep: -1pt, label-pos: 0.35, //label-angle: auto,
          text(size: 10pt)[$w_#(j+1) = weights.at(#(j+1))$],)
  },
  edge(bias, output, "--|>", mark-scale: 75%,
        label-side: right, label-sep: -1pt, label-pos: 0.35,
        text(size: 10pt)[$w_0 = weights.first()$],),
  edge(output, (1.5, output.at(1)), "-|>", mark-scale: 75%)

)}

// --------------------------------------------------

// 2D separating hyperplane diagram

// dot product of two vectors
#let dot-prod(a, b) = a.zip(b).map(x => x.product()).sum()

// sum of two vectors
#let sum(a,b) = a.zip(b).map(x => x.sum())

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


#let plot-2d(
  lambda,
  weights: none,
) = {
cetz.canvas({
  import cetz.draw: *

  scale(2)

  // anchor canvas size regardless of boundary position
  rect((-0.35, -0.35), (1.4, 1.4), stroke: none, fill: none)

  // function values
  //let lambda = (inputs) => if (dot-prod(weights.slice(1), inputs) + weights.first() > 0) { 1 } else { 0 }
  let inputs = cart-prod(..range(2).map(_ => (0, 1)))

  // axis
  group({
    set-style(mark: (end: "stealth", fill: black))
    line((0, 0), (1.5, 0))
    line((0, 0), (0, 1.5))
  })

  for comb in inputs {
    // points
    circle(comb, radius: 0.05, fill: luma(72%))
    // labels
    let offset = (0.15, 0.15)
    let symb = if (lambda(comb) > 0) { $+$ } else { $-$ }
    content(sum(comb, offset), text(size: 12pt)[#symb], align: center)
  }

  // decision boundary
  if weights.all(x => type(x) == float) {
    let seg = clip-boundary(weights)
    if seg.len() >= 2 {
      line(seg.at(0), seg.at(1), stroke: 1pt + black)
    }
  }

})}

// --------------------------------------------------

// 3D separating hyperplane diagram

// 3D cross product
#let cross-prod(a, b) = (
  a.at(1)*b.at(2) - a.at(2)*b.at(1),
  a.at(2)*b.at(0) - a.at(0)*b.at(2),
  a.at(0)*b.at(1) - a.at(1)*b.at(0),
)

// Normalize a vector
#let normalize(v) = { let n = calc.sqrt(dot-prod(v, v)); v.map(x => x / n) }

// Evaluate plane equation w0 + w1*x1 + w2*x2 + w3*x3 at point p
#let eval-plane(w, p) = w.at(0) + w.at(1)*p.at(0) + w.at(2)*p.at(1) + w.at(3)*p.at(2)

// Find vertices of the intersection polygon of the plane with the unit cube [0,1]^3
#let clip-plane-to-cube(weights) = {
  let edges = (
    ((0,0,0),(1,0,0)), ((0,1,0),(1,1,0)), ((0,0,1),(1,0,1)), ((0,1,1),(1,1,1)),
    ((0,0,0),(0,1,0)), ((1,0,0),(1,1,0)), ((0,0,1),(0,1,1)), ((1,0,1),(1,1,1)),
    ((0,0,0),(0,0,1)), ((1,0,0),(1,0,1)), ((0,1,0),(0,1,1)), ((1,1,0),(1,1,1)),
  )
  let pts = ()
  for (p0, p1) in edges {
    let f0 = eval-plane(weights, p0)
    let f1 = eval-plane(weights, p1)
    let d = f0 - f1
    if calc.abs(d) > 1e-10 {
      let t = f0 / d
      if t >= 0 and t <= 1 {
        let pt = range(3).map(i => p0.at(i) + t * (p1.at(i) - p0.at(i)))
        // deduplicate (plane may pass through a cube vertex, hitting two edges)
        if not pts.any(q => range(3).all(j => calc.abs(pt.at(j) - q.at(j)) < 1e-9)) {
          pts += (pt,)
        }
      }
    }
  }
  pts
}

// Sort polygon vertices into convex (angular) order around their centroid in the plane
#let sort-polygon(pts, normal) = {
  if pts.len() < 3 { return pts }
  let centroid = range(3).map(i => pts.map(p => p.at(i)).sum() / pts.len())
  let n = normalize(normal)
  let ref = if calc.abs(n.at(0)) < 0.9 { (1.0, 0.0, 0.0) } else { (0.0, 1.0, 0.0) }
  let u = normalize(cross-prod(n, ref))
  let v = cross-prod(n, u)
  pts.sorted(key: p => {
    let d = range(3).map(i => p.at(i) - centroid.at(i))
    calc.atan2(dot-prod(d, v), dot-prod(d, u)).rad()
  })
}


#let plot-3d(lambda, weights: none) = cetz.canvas({
  import cetz.draw: *

  scale(1.1)

  // Cabinet oblique projection: x1 → right, x3 → up, x2 → upper-right at 45° (half-scale)
  let sc = 2.0
  let ax =  calc.sqrt(2) / 4  // x2 horizontal component (cos 45° × ½)
  let ay = -calc.sqrt(2) / 4  // x2 vertical component   (sin 45° × ½, upward = back view)
  let p3 = (x1, x2, x3) => (sc*(x1 + x2*ax), sc*(x3 - x2*ay))
  let pv = pt => p3(pt.at(0), pt.at(1), pt.at(2))

  let inputs = cart-prod(..range(3).map(_ => (0, 1)))

  // Reserve consistent canvas size regardless of plane position
  rect((-0.5, -0.35), (3.35, 3.35), stroke: none, fill: none)

  // Cube skeleton — light dotted edges drawn behind everything
  let cube-edges = (
    ((0,0,0),(1,0,0)), ((0,1,0),(1,1,0)), ((0,0,1),(1,0,1)), ((0,1,1),(1,1,1)),
    ((0,0,0),(0,1,0)), ((1,0,0),(1,1,0)), ((0,0,1),(0,1,1)), ((1,0,1),(1,1,1)),
    ((0,0,0),(0,0,1)), ((1,0,0),(1,0,1)), ((0,1,0),(0,1,1)), ((1,1,0),(1,1,1)),
  )
  for (p0, p1) in cube-edges {
    line(pv(p0), pv(p1), stroke: (paint: luma(70%), dash: "dotted", thickness: 0.75pt))
  }

  // Semi-transparent decision boundary plane
  if weights != none {
    let plane-pts = sort-polygon(clip-plane-to-cube(weights), weights.slice(1))
    if plane-pts.len() >= 3 {
      line(
        ..plane-pts.map(pv),
        close: true,
        stroke: 0.8pt + blue.darken(20%),
        fill: blue.lighten(55%).transparentize(50%),
      )
    }
  }

  // Axes with arrowheads
  group({
    set-style(mark: (end: "stealth", fill: black))
    line(p3(0,0,0), p3(1.4, 0, 0))
    line(p3(0,0,0), p3(0, 1.75, 0))
    line(p3(0,0,0), p3(0, 0, 1.4))
  })
  content(p3(1.55,  0,    0   ), $x_1$)
  content(p3(0,     0.95, 0.3 ), $x_2$)
  content(p3(-0.12, 0,    1.55), $x_3$)

  // Data points with +/- labels offset outward from the cube center
  let center = pv((0.5, 0.5, 0.5))
  for pt in inputs {
    let pp = pv(pt)
    let dx = pp.at(0) - center.at(0)
    let dy = pp.at(1) - center.at(1)
    let dlen = calc.sqrt(dx*dx + dy*dy)

    // Override label direction for points that would land on a line
    let (ldx, ldy) = if pt == (0, 1, 0) or pt == (1, 0, 1) {
      (calc.sqrt(2)/2, -calc.sqrt(2)/2)  // bottom-right
    } else {
      (dx/dlen, dy/dlen)
    }

    circle(pp, radius: 0.065, fill: luma(72%), stroke: 0.8pt + luma(40%))
    content(
      (pp.at(0) + ldx*0.22, pp.at(1) + ldy*0.22),
      text(size: 9pt)[#if lambda(pt) > 0 { $+$ } else { $-$ }],
    )
  }
})

// --------------------------------------------------

// Main

#let draw-diagrams(
  lambda, n, weights,
) = {
  grid(
    columns: 3,
    align: horizon,
    column-gutter: (10mm, 5mm,),

    logic-table(lambda, n),
    slp-diagram(n, weights: weights),
    if (n == 2) { plot-2d(lambda, weights: weights) }
    else if (n == 3) { plot-3d(lambda, weights: weights) },
  )
}


// AND
#{
  let n = 2
  let lambda = (inputs) => if (inputs.any(i => i == 0)) { 0 } else { 1 }
  let weights = (-n + 0.5,) + range(n).map(_ => 1.0)
  draw-diagrams(lambda, n, weights)
}
#pagebreak()

// OR
#{
  let n = 2
  let lambda = (inputs) => if (inputs.any(i => i == 1)) { 1 } else { 0 }
  let weights = (-0.5,) + range(n).map(_ => 1.0)
  draw-diagrams(lambda, n, weights)
}
#pagebreak()

// XOR
#{
  let n = 2
  let lambda = (inputs) => if (inputs.reduce((a, b) => a + b) == 1) { 1 } else { 0 }
  let weights = ("?", "?", "?")
  draw-diagrams(lambda, n, weights)
}
#pagebreak()

// Majority vote (n=3): output 1 iff at least 2 of 3 inputs are 1
#{
  let n = 3
  let lambda = (inputs) => if inputs.sum() >= 2 { 1 } else { 0 }
  let weights = (-1.5,) + range(n).map(_ => 1.0)
  draw-diagrams(lambda, n, weights)
}
