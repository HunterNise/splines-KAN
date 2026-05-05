#!/bin/bash
# Launch train.py as a Python module so relative imports (from .model import NN) work.
# Walks upward from this script's directory to find the repo root (marked by pytest.ini),
# then converts the path to a dotted module name and runs with python -m.
# Copy this script alongside train.py + model.py to any subfolder and it will still work.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# parse arguments
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
if [ -z "$SCRIPT" ]; then
    echo "Usage: $(basename "${BASH_SOURCE[0]}") train|eval|eval-attr [-O] [...]" >&2
    exit 1
fi

# find repo root
ROOT="$SCRIPT_DIR"
while [ ! -f "$ROOT/pytest.ini" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
if [ ! -f "$ROOT/pytest.ini" ]; then
    echo "Error: could not find repo root (no pytest.ini found)" >&2
    exit 1
fi

# build dotted module path:  source/task1/NN1  ->  source.task1.NN1.train
REL="${SCRIPT_DIR#$ROOT/}"
MODULE="${REL//\//.}.${SCRIPT}"

cd "$ROOT"
exec python $PYTHON_FLAGS -m "$MODULE" "${PASSTHROUGH[@]}"
