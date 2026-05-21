// General

#let lead = 0.65em
#set par(leading: lead)   // does not work like this


// Text

#let bozza(body) = text(fill: gray.darken(20%), body)

#let url = it => {
  set text(fill: blue)
  show: underline
  it
}

// Footnotes

#import "@preview/latedef:0.1.0" as latedef

#let (fmark, ftext) = latedef.latedef-setup(simple: true, footnote: true)


// Math

#let dominates(T) = $attach(succ, br: #T)$
#let dominated(T) = $attach(prec, br: #T)$
#let card(A) = $\##A$
#let neq = math.eq.not
#let iff = math.arrow.l.r.double
#let vfrac = math.frac.with(style: "vertical")
#let big = math.lr.with(size: 150%)
#let half = h(0.5em)
#let absurd = box(width: 0.6em, height: 1em, baseline: 0.2em)[
  #polygon(
    fill: black,
    (97.0%, 6.3%),
    (63.6%, 6.3%),
    (15.2%, 54.7%),
    (52.7%, 50.0%),
    (20.0%, 80.0%),
    (5.5%, 69.1%),
    (4.2%, 95.3%),
    (62.4%, 85.3%),
    (27.9%, 83.1%),
    (89.7%, 38.8%),
    (41.2%, 45.0%)
  )
]


#import "/template/math-env.typ": *
#import "/template/style.typ"

#let c = counter("math-env")

#let definition = math-env(
  name: "Definition", 
  counter: c, 
  style: style.indent.with(acapo: true)
)
#let theorem = math-env(
  name: "Theorem", 
  counter: c, 
  style: style.indent.with(acapo: true)
)
#let proposition = math-env(
  name: "Proposition", 
  counter: c, 
  style: style.indent.with(acapo: true)
)
#let lemma = math-env(
  name: "Lemma", 
  counter: c, 
  style: style.indent.with(acapo: true)
)
#let corollary = math-env(
  name: "Corollary", 
  counter: c, 
  style: style.indent.with(acapo: true)
)
#let remark = math-env(
  name: "Remark", 
  counter: none, 
  style: style.indent
)
#let example = math-env(
  name: "Example", 
  counter: none, 
  style: style.indent
)
#let problem = math-env(
  name: "Problem", 
  counter: none, 
  style: style.indent
)
#let conjecture = math-env(
  name: "Conjecture", 
  counter: none, 
  style: style.indent
)
#let notation = math-env(
  name: "Notation", 
  counter: none, 
  style: style.indent
)
#let proof = math-env(
  name: "Proof", 
  counter: none, 
  suffix: place(bottom + right, $square$), 
  style: style.indent.with(
    name_fmt: it => [#emph[#it]]
  )
)


#let rules = doc => {

  set underline(offset: 0.15em)
  
  
  show: math-envs-init

  // https://forum.typst.app/t/is-it-possible-to-avoid-breaking-a-line-of-math-over-a-linebreak/540/2
  show math.equation.where(block: false): box
  show math.equation.where(block: false): set math.frac(style: "horizontal")

  
  show ref: it => {
    let el = it.element

    let is-math-env = (
      el != none 
      and el.func() == figure 
      and el.kind == "math-env"
    )

    let is-equation = (
      el != none 
      and el.func() == math.equation 
    )
    
    if is-math-env or is-equation {
      set text(fill: blue)
      it
    } else {it}
  }

  show cite: it => {
    set text(fill: lime)
    it
  }

  //show link: set text(fill: blue)
  //show link: underline

  
  doc
}



// TO DO

// add a way to set global parameters to selectively change math-env instatiations by documents
// in particular remove counter param from the closure (it is used only for a hack)