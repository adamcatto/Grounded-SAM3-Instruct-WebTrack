#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load the machine-local project root when the caller did not provide one.
# .env is intentionally gitignored; copy .env.example to create it.
if [ -z "${SAM3_PROJECTS_DIR:-}" ] && [ -f "$SCRIPT_DIR/.env" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
fi

# --------------------------------------------------
# Reuse currently active conda env if present
# --------------------------------------------------

if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "Using existing conda env:"
    echo "  $CONDA_PREFIX"

else
    echo "No active conda env detected"

    # ONLY do discovery if absolutely necessary

    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"

    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"

    else
        echo "ERROR: Could not locate user conda install"
        exit 1
    fi

    conda activate sam3
fi

echo "Python: $(which python)"
echo "Conda prefix: $CONDA_PREFIX"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# Keep the documented precedence in backend/project_manager.py: the newer
# tracking-specific variable wins over SAM3_PROJECTS_DIR when both are set.
if [ -n "${SAM3_TRACKING_PROJECTS_DIR:-}" ]; then
    echo "SAM3 projects dir: $SAM3_TRACKING_PROJECTS_DIR"
elif [ -n "${SAM3_PROJECTS_DIR:-}" ]; then
    export SAM3_PROJECTS_DIR
    echo "SAM3 projects dir: $SAM3_PROJECTS_DIR"
else
    echo "SAM3 projects dir: not configured (using backend default)"
    echo "Tip: cp .env.example .env and edit SAM3_PROJECTS_DIR"
fi

fuser -k 8000/tcp >/dev/null 2>&1 || true
sleep 0.3

cd "$SCRIPT_DIR/backend"

# The frontend reaches the API through the Vite proxy, so the backend only
# needs to listen on localhost. Override with BACKEND_HOST if you must.
exec python -m uvicorn server:app \
    --host "${BACKEND_HOST:-127.0.0.1}" \
    --port 8000 \
    --reload
