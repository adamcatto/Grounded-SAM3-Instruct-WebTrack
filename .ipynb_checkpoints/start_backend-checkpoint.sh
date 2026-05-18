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

fuser -k 8000/tcp >/dev/null 2>&1 || true
sleep 0.3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"

exec python -m uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload