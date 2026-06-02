#!/bin/bash
# setup.sh — run once after cloning the repository to configure the environment.
# Accepts an optional argument to force a specific installation method:
#   ./setup.sh [docker|uv]
# If no argument is given, auto-detects based on available tools:
#   - Docker:  builds the image with a Dockerfile, then writes "docker" to .mode
#   - uv:      creates a venv and installs dependencies, then writes "uv" to .mode
# If neither is available (or the forced method is not found), prints an error and exits.
#
# The .mode file is used by run.sh to determine how to start the environment when you run it.


# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory of the script, regardless of where it's called from
#   $0 is the path to this script, dirname gets the directory part, cd enters the directory and pwd gives its absolute path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Optional first argument: "docker" or "uv". Defaults to "auto" (detect from available tools).
MODE="${1:-auto}"
if [[ "$MODE" != "auto" && "$MODE" != "docker" && "$MODE" != "uv" ]]; then
    echo "Usage: $0 [docker|uv]"
    exit 1
fi

if [[ "$MODE" == "docker" ]] || { [[ "$MODE" == "auto" ]] && command -v docker &>/dev/null; }; then
    echo "Setting up with Docker..."
    # Build the docker image with the tag splines:latest, using the Dockerfile located at $SCRIPT_DIR/docker/Dockerfile and the build context as $SCRIPT_DIR
    docker build -t splines:latest -f "$SCRIPT_DIR/docker/Dockerfile" "$SCRIPT_DIR"

    # Write "docker" to .mode to indicate that Docker should be used when running run.sh
    echo "docker" > "$SCRIPT_DIR/.mode"
    echo "Done. Run ./run.sh to start a container shell."
elif [[ "$MODE" == "uv" ]] || { [[ "$MODE" == "auto" ]] && command -v uv &>/dev/null; }; then
    echo "Setting up with uv..."
    cd "$SCRIPT_DIR"
    # Create a virtual environment using uv and install the dependencies from the requirements.txt file located in the docker directory
    uv venv
    uv pip install -r docker/requirements.txt

    # Write "uv" to .mode to indicate that uv should be used when running run.sh
    echo "uv" > "$SCRIPT_DIR/.mode"
    echo "Done. Run ./run.sh to start an interactive shell with the venv active."
else
    echo "Neither Docker nor uv found. Please install one of them, or specify a mode: $0 [docker|uv]"
    exit 1
fi
