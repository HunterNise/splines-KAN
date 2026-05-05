#!/bin/bash
# Launch train.py as a Python module so relative imports (from .model import NN) work.
# Walks upward from this script's directory to find the repo root (marked by an anchor file),
# then converts the path to a dotted module name and runs with python -m.
# Copy this script alongside train.py + model.py to any subfolder and it will still work.


# Get the directory of the script, regardless of where it's called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
#  - first non-flag argument is the script to run (train/eval/eval-attr)
#  - -O flag is passed to python for optimized mode (no asserts, faster but less informative errors)
#  - all other arguments are passed through to the Python script
SCRIPT=""
PYTHON_FLAGS=""
PASSTHROUGH=()
for arg in "$@"; do
    case "$arg" in
        train|eval|eval-attr) SCRIPT="$arg" ;;
        -O)         PYTHON_FLAGS="-O" ;;
        *)          PASSTHROUGH+=("$arg") ;;
    esac
done
# Check that a script was specified, otherwise print usage and exit
if [ -z "$SCRIPT" ]; then
    echo "Usage: $(basename "${BASH_SOURCE[0]}") train|eval|eval-attr [-O] [...]" >&2
    exit 1
fi

# Find repo root by walking upward until we find the anchor file
ROOT="$SCRIPT_DIR"
ANCHOR="setup.sh"
while [ ! -f "$ROOT/$ANCHOR" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
# If we reached the root directory without finding the anchor, print an error and exit
if [ ! -f "$ROOT/$ANCHOR" ]; then
    echo "Error: could not find repo root (no $ANCHOR found)" >&2
    exit 1
fi

# Build dotted module path:  source/task/NN  ->  source.task.NN.train
REL="${SCRIPT_DIR#$ROOT/}"
MODULE="${REL//\//.}.${SCRIPT}"

# Change to the repo root and run the script as a module with python -m, passing through any additional arguments
cd "$ROOT"
exec python $PYTHON_FLAGS -m "$MODULE" "${PASSTHROUGH[@]}"
