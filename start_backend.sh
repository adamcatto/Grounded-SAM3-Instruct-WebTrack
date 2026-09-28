#!/bin/bash
set -euo pipefail

# --------------------------------------------------
# Reuse currently active conda env if present
# --------------------------------------------------

if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "Using existing conda env:"
    echo "  $CONDA_PREFIX"

else
    echo "No active conda env detected"

    # ONLY do discovery if absolutely necessary

    if [ -f "/sc/arion/work/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/sc/arion/work/$USER/miniconda3/etc/profile.d/conda.sh"

    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"

    else
        echo "ERROR: Could not locate user conda install"
        exit 1
    fi

    conda activate sam3
fi

echo "Python: $(which python)"
echo "Conda prefix: $CONDA_PREFIX"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export SAM3_TRACKING_PROJECTS_DIR="${SAM3_TRACKING_PROJECTS_DIR:-/opt/projects/segmentation_tracking_projects/}"
echo "SAM3 projects dir: $SAM3_TRACKING_PROJECTS_DIR"

fuser -k 8000/tcp >/dev/null 2>&1 || true
sleep 0.3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"

# The frontend reaches the API through the Vite proxy, so the backend only
# needs to listen on localhost. Override with BACKEND_HOST if you must.
exec python -m uvicorn server:app \
    --host "${BACKEND_HOST:-127.0.0.1}" \
    --port 8000 \
    --reload
