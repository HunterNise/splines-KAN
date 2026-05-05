#!/bin/bash
# setup.sh — run once after cloning to configure the environment.
# Detects whether Docker is available and sets up accordingly:
#   - Docker:  builds the image, writes root.txt = /app  (the container workdir)
#   - uv:      creates a venv, installs deps, writes root.txt = this directory
#
# The generated files (root.txt, .mode) are gitignored and machine-specific.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v docker &>/dev/null; then
    echo "Docker found. Building image..."
    docker build -t splines -f "$SCRIPT_DIR/docker/Dockerfile" "$SCRIPT_DIR"

    echo "docker" > "$SCRIPT_DIR/.mode"
    echo "Done. Run ./run.sh to start a container shell."
else
    echo "Docker not found. Setting up with uv..."
    cd "$SCRIPT_DIR"
    uv venv
    uv pip install -r docker/requirements.txt

    echo "uv" > "$SCRIPT_DIR/.mode"
    echo "Done. Run ./run.sh to start an interactive shell with the venv active."
fi
