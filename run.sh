#!/bin/bash
# run.sh — starts an interactive shell for the project.
# Reads .mode (set by setup.sh) to decide between Docker and uv.
#
# You should run setup.sh first if .mode does not exist yet.


# Get the directory of the script, regardless of where it's called from
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Read the mode from the file .mode, which should have been created by setup.sh.
# If the file doesn't exist, print an error and exit.
if [ ! -f "$SCRIPT_DIR/.mode" ]; then
    echo "Error: .mode file not found. Please run setup.sh first to configure the environment."
    exit 1
fi
MODE=$(cat "$SCRIPT_DIR/.mode" 2>/dev/null)

# Check the mode and start the appropriate environment
if [ "$MODE" = "docker" ]; then
    # Run a Docker container with the splines image (tag latest)
    # Mount the project directory to /app in the container and set the working directory to /app
    # Returns an interactive bash shell, and removes the container when you exit
    docker run -it --rm -v "$SCRIPT_DIR":/app -w /app splines:latest
elif [ "$MODE" = "uv" ]; then
    # Set environment variable to the project directory so that imports work correctly
    export PYTHONPATH="$SCRIPT_DIR"
    # Run an interactive bash shell with the uv environment active, using the project directory as the working directory
    uv run --directory "$SCRIPT_DIR" bash
else
    echo "Error: Invalid mode in .mode file. Expected 'docker' or 'uv', got '$MODE'. Please check your setup.sh configuration."
    exit 1
fi
