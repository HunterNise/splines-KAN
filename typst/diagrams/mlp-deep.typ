#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#set page(width: auto, height: auto, margin: 5mm, fill: white)
#set text(font: "Libertinus Serif", size: 20pt)


#let circle(pos, body, name: none, 
            radius: 4.5mm, fill: white, stroke: 0.8pt + luma(55%)
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
  let dist = 2 / calc.ln(num+1)
  let L = (num - 1) * dist
  let y = range(num).map(i => i * dist - (L / 2))
  y.map(y => (col, y))
}

#let draw-neurons(coords, ..args) = {
  coords.map(pos => circle(pos, [], ..args))
}


#let draw-diagram(num_neurons) = diagram(debug: 0,

  spacing: (18mm, 16mm),
  edge-stroke: 0.85pt + luma(55%),
  node-stroke: 0.80pt + luma(55%),
  edge-corner-radius: none,

  let num_layers = num_neurons.len(),

  let old_coords = get-coords(num_neurons.at(0), 0),
  let y_bottom = get-coords(calc.max(..num_neurons), 0).last().at(1) + 0.5,
  
  for l in range(num_layers) {
    let coords = get-coords(num_neurons.at(l), l * 2.25)
    
    // neurons
    draw-neurons(coords, radius: 4.5mm, fill: if l == 0 { luma(72%) } else { white })
    
    // edges
    if (l != 0) {
      for i in coords {
        for j in old_coords {
          ( edge(j, i, "-") ,)
        }
      }
      old_coords = coords
    }

    // labels
    if (l == 0) {
      ( textnode((l * 2.25, y_bottom), [$bold(x)$]) ,)
    } else if (l == num_layers - 1) {
      ( textnode((l * 2.25, y_bottom), [$bold(y)$]) ,)
    }
    if (l != 0) {
      ( textnode((l * 2.25 - (2.25 / 2), y_bottom), [$W_#l$]) ,)
    }
    if (l != 0 and l != num_layers - 1) {
      ( textnode((l * 2.25, y_bottom), [$sigma_#l$]) ,)
    }
  },

)



#draw-diagram(
  (3, 5, 4, 2)
)
