#!/bin/bash
# slides.sh — compiles the typst presentation to PDF.


# set typst root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export TYPST_ROOT="$SCRIPT_DIR/typst"
cd "$TYPST_ROOT"

# compile dependencies
# ...

# compile the main file
typst compile "slides.typ" "slides.pdf"
