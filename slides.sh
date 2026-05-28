#!/bin/bash
# slides.sh — compiles the typst presentation to PDF.


# set typst root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export TYPST_ROOT="$SCRIPT_DIR/typst"
cd "$TYPST_ROOT"

# compile dependencies
typst compile diagrams/perceptron.typ   diagrams/perceptron.svg
typst compile diagrams/slp.typ          diagrams/slp{p}.svg
typst compile diagrams/xor.typ          diagrams/xor.svg
typst compile diagrams/mlp-shallow.typ  diagrams/mlp-shallow{p}.svg
typst compile diagrams/mlp-deep.typ     diagrams/mlp-deep.svg

# compile the main file
typst compile "slides.typ" "slides.pdf"
