#!/bin/bash
# setup.sh — run once after cloning the repository to configure the environment.
# Detects whether Docker is available and sets up accordingly:
#   - Docker:  builds the image with a Dockerfile, then writes "docker" to .mode
#   - uv:      creates a venv and installs dependencies, then writes "uv" to .mode
# If neither is available, it prints an error message and exits.
#
# The .mode file is used by run.sh to determine how to start the environment when you run it.


# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory of the script, regardless of where it's called from
#   $0 is the path to this script, dirname gets the directory part, cd enters the directory and pwd gives its absolute path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check if Docker is available
if command -v docker &>/dev/null; then
    echo "Docker found. Building image..."
    # Build the docker image with the tag splines:latest, using the Dockerfile located at $SCRIPT_DIR/docker/Dockerfile and the build context as $SCRIPT_DIR
    docker build -t splines:latest -f "$SCRIPT_DIR/docker/Dockerfile" "$SCRIPT_DIR"

    # Write "docker" to .mode to indicate that Docker should be used when running run.sh
    echo "docker" > "$SCRIPT_DIR/.mode"
    echo "Done. Run ./run.sh to start a container shell."
# If Docker is not available, check if uv is available
elif command -v uv &>/dev/null; then
    echo "Docker not found. Setting up with uv..."
    cd "$SCRIPT_DIR"
    # Create a virtual environment using uv and install the dependencies from the requirements.txt file located in the docker directory
    uv venv
    uv pip install -r docker/requirements.txt
    
    # Write "uv" to .mode to indicate that uv should be used when running run.sh
    echo "uv" > "$SCRIPT_DIR/.mode"
    echo "Done. Run ./run.sh to start an interactive shell with the venv active."
else
    echo "Neither Docker nor uv found. Please install one of them to set up the environment."
    exit 1
fi
