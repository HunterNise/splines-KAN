#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#set page(width: auto, height: auto, margin: 5mm, fill: white)
#set text(font: "Libertinus Serif", size: 14pt)


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


#let draw-diagram(
  edge-stroke: 0.85pt + luma(55%),
  spot: none,
) = {
diagram(debug: 0,

  spacing: (18mm, 16mm),
  edge-stroke: edge-stroke,
  node-stroke: 0.80pt + luma(55%),
  edge-corner-radius: none,

  let num_input  = 3,
  let num_hidden = 4,
  let num_output = 1,

  
  // Neurons

  let input_coords  = get-coords(num_input,  0),
  let hidden_coords = get-coords(num_hidden, 2.25),
  let output_coords = get-coords(num_output, 4.5),

  draw-neurons(input_coords , radius: 4.5mm, fill: luma(72%)),
  draw-neurons(hidden_coords, radius: 4.5mm, fill: white),
  draw-neurons(output_coords, radius: 4.5mm, fill: white),


  // Edges

  for i in hidden_coords {
    for j in input_coords {
      edge(j, i, "-")
    }
  },
  for i in output_coords {
    for j in hidden_coords {
      edge(j, i, "-")
    }
  },


  // Spotlight

  if (spot != none) {
    
    // Input nodes labels
    for (i, pos) in input_coords.enumerate() {
      textnode((rel: (-0.4, 0), to: pos), [$x_#(i+1)$])
    }
    
    // Highlight the neuron
    let end = hidden_coords.at(spot - 1)
    circle(end, [$sigma$], stroke: 1.5pt)
    
    // Highlight the incoming edges
    for (j, value) in input_coords.enumerate() {
      edge(value, end, "-", stroke: (paint: blue, thickness: 1pt), [$w_(spot #(j+1))$])
    }

    // Highlight the outgoing edges
    if output_coords.len() == 1 {
      edge(end, output_coords.first(), "-", stroke: (paint: olive, thickness: 1.25pt), [$a_#spot$])
    }
  }
  
  else {
    // Layer labels
    
    let y_bottom = calc.max(input_coords.last().at(1), hidden_coords.last().at(1), output_coords.last().at(1)) + 0.65
    //set text(size: 10pt)
    textnode((0.00, y_bottom), [Input  \ Layer $in RR^#num_input$])
    textnode((2.25, y_bottom), [Hidden \ Layer $in RR^#num_hidden$])
    textnode((4.50, y_bottom), [Output \ Layer $in RR^#num_output$])
  }
)}



// Main

#draw-diagram()
#pagebreak()

#draw-diagram(
  edge-stroke: 0.85pt + luma(85%),
  spot: 1,
)
#pagebreak()

#draw-diagram(
  edge-stroke: 0.85pt + luma(85%),
  spot: 2,
)
