#!/bin/bash
# run.sh — starts an interactive shell for the project.
# Reads .mode (set by setup.sh) to decide between Docker and uv.
# Run setup.sh first if .mode does not exist yet.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE=$(cat "$SCRIPT_DIR/.mode" 2>/dev/null || echo "docker")

if [ "$MODE" = "docker" ]; then
    # Mount the project at /app, matching the container's WORKDIR and root.txt value
    docker run -it --rm -v "$SCRIPT_DIR":/app -w /app splines:latest
else
    # Set PYTHONPATH so that `from source.functions import ...` works from any subdirectory
    export PYTHONPATH="$SCRIPT_DIR"
    uv run --directory "$SCRIPT_DIR" bash
fi