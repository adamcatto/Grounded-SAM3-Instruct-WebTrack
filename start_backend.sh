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

# --------------------------------------------------
# Agent LLM (vLLM on :8001)
# --------------------------------------------------
# The chat agent auto-detects vLLM on :8001 and otherwise falls back to
# Ollama, so start vLLM alongside the backend. It runs detached and survives
# backend restarts (model load takes minutes). Set AGENT_LLM_AUTOSTART=0 to
# skip, e.g. when AGENT_LLM_BASE_URL points at a remote server.
# a100-shared fits next to SAM3 on one 80GB GPU; use a100/h100x4 for a
# dedicated LLM GPU.
export AGENT_LLM_PROFILE="${AGENT_LLM_PROFILE:-a100-shared}"
VLLM_PORT="${VLLM_PORT:-8001}"
if [ "${AGENT_LLM_AUTOSTART:-1}" != "0" ]; then
    if fuser "$VLLM_PORT/tcp" >/dev/null 2>&1 || pgrep -u "$(id -u)" -f "vllm serve" >/dev/null 2>&1; then
        echo "Agent LLM: vLLM already running (port $VLLM_PORT)"
    else
        mkdir -p "$SCRIPT_DIR/logs"
        echo "Agent LLM: starting vLLM (profile $AGENT_LLM_PROFILE) on :$VLLM_PORT"
        echo "  log: $SCRIPT_DIR/logs/vllm_agent.log"
        # Clean LD_LIBRARY_PATH: the sam3 env libs exported above must not
        # shadow the vllm env (serve_agent_llm.sh adds its own lib dir).
        LD_LIBRARY_PATH="" VLLM_PORT="$VLLM_PORT" setsid nohup bash "$SCRIPT_DIR/scripts/serve_agent_llm.sh" vllm \
            --profile "$AGENT_LLM_PROFILE" \
            > "$SCRIPT_DIR/logs/vllm_agent.log" 2>&1 < /dev/null &
    fi
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
