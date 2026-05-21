#let plain(
    name, number, title, prefix, suffix, body, 
    name_fmt  : it => [*#it*], 
    name_sep  : [.], 
    title_fmt : it => [(#it)], 
    title_sep : [], 
    body_fmt  : it => [#it]
  ) = [
  #block(width: 100%)[
    #prefix
    #name_fmt[#name #number#name_sep]
    #if title != none [#title_fmt(title) #title_sep]
    #body_fmt(body)
    #suffix
  ]
]

#let acapo = plain.with(
  title_sep: [#linebreak()]
)


#let indent(
    name, number, title, prefix, suffix, body, 
    name_fmt  : it => [*#it*], 
    name_sep  : [.], 
    title_fmt : it => [(#it)], 
    indent    : 2em, 
    acapo     : false, 
    body_fmt  : it => [#it]
  ) = [
  #block()[
    #prefix
    #if title != none {acapo = true}
    #if acapo == false [
      #grid(
        columns: (auto, 1fr),
        column-gutter: 0.5em,
        name_fmt[#name #number#name_sep],
        body_fmt(body)
      )
    ] else [
      #name_fmt[#name #number#name_sep]
      #if title != none [#title_fmt(title)]
      #box()[   // block would break paragraph
      #grid(
        columns: (indent, 1fr),
        "",
        body_fmt(body)
      )]
    ]
    #suffix
  ]
]