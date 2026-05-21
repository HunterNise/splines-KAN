#let math-env(
    name    : none, 
    counter : none, 
    numbering: "1.1", 
    prefix  : none, 
    suffix  : none, 
    style   : auto, 
  ) = {
  // check if name was provided
  if name == none {
    panic("You have created a `math-env` without a `name`.")
  }

  // wrap native counter
  if counter != none and type(counter) != dictionary {
    counter = (
      step   : (..args) => { counter.step(..args) },
      get    : (..args) => { counter.get(..args) },
      at     : (..args) => { counter.at(..args) },
      display: (..args) => { counter.display(..args) },
    )
  }

  // default style
  if style == auto {
    style = (
        name, number, title, prefix, suffix, body
      ) => [
      #block(width: 100%)[
        #prefix
        *#name #number.*
        #if title != none [(#title)]
        #body
        #suffix
      ]]
  }

  // return the environment for the user
  return (
    name  : name, 
    counter: counter,
    number: auto, 
    numbering: numbering, 
    title : none, 
    prefix: prefix, 
    suffix: suffix, 
    body  : none, 
    label : none, 
    style : style, 
    ..args
  ) => {
    // interface with positional arguments
    let pos = args.pos()
    if pos.len() == 2 {
      (title, body) = (pos.at(0), pos.at(1))
    } else if pos.len() == 1 {
      body = pos.at(0)
    } else {
      panic("Usage: #name[title][body] or #name[body]")
    }
    
    [#figure(
      kind: "math-env", 
      supplement: name, 
      outlined: false
    )[
      // if we have a counter, step it (unless user supplied an explicit number)
      #if counter != none {
        if number == auto [
          // normal automatic numbering
          #(counter.step)()
          #{number = context (counter.display)(numbering)}
          // store a function that can compute the final numbering from a location
          #metadata((loc) => { std.numbering(numbering, ..((counter.at)(loc))) })
          #std.label("math-envs:numberfunc")
        ] else [
          // user provided a manual number; store it so refs still work
          #metadata((loc) => number)
          #std.label("math-envs:numberfunc")
        ] // we had to use square brackets to attach labels
      } else {
        // don't add extra space if there is no number
        number = [#h(0pt, weak: true)]
      }

      // apply the style
      #style(name, number, title, prefix, suffix, body)
      
    ] #if label != none {std.label(label)}]   // attach label parameter (also here we needed square brackets)
  }
}


#let math-envs-init = doc => {
  show figure.where(kind: "math-env"): set align(start)
  show figure.where(kind: "math-env"): set block(breakable: true)
  show figure.where(kind: "math-env"): fig => fig.body

  show ref: it => {
    if (it.element != none 
        and it.element.func() == figure 
        and it.element.kind == "math-env") {
      // prefer an explicit supplement from the citation, otherwise fall back to the figure's supplement
      let supplement = (
        if it.citation.supplement != none 
          { it.citation.supplement } 
        else 
          { it.element.supplement }
      )

      // try to find the internal number-function label placed by counted blocks
      let data = query(selector(label("math-envs:numberfunc")).after(it.target)).first()
      // if we found a label, it's a counted block: call the stored function to compute the final number
      if data != none {
        let numberfunc = data.value
        link(it.target, 
          [#supplement #numberfunc(data.location())]
        )
      } else {
        // no label => uncounted block: just show the supplement
        link(it.target, [#supplement])
      }
    } else {
      // not one of our special figures: leave the ref alone
      it
    }
  }

  doc
}