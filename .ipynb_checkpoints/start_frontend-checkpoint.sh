#!/bin/bash
set -euo pipefail

# --------------------------------------------------
# Resolve node executable robustly
# --------------------------------------------------

if command -v node >/dev/null 2>&1; then
    NODE_BIN_DIR="$(dirname "$(command -v node)")"

else
    NODE_BIN_DIR="$(ls -d ~/.nvm/versions/node/v22*/bin 2>/dev/null | sort -V | tail -1 || true)"

    if [ -z "${NODE_BIN_DIR:-}" ]; then
        echo "ERROR: Could not find Node.js"
        echo "Load a module with:"
        echo "  ml nodejs/22.11.0"
        echo "or install Node via nvm."
        exit 1
    fi

    export PATH="$NODE_BIN_DIR:$PATH"
fi

echo "Using node: $(which node)"
echo "Node version: $(node --version)"

# --------------------------------------------------
# Kill stale Vite server
# --------------------------------------------------

fuser -k 5173/tcp >/dev/null 2>&1 || true
sleep 0.3

# --------------------------------------------------
# Resolve repo location robustly
# --------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

cd "$REPO_ROOT/frontend"

echo "Frontend dir: $(pwd)"

# --------------------------------------------------
# Install deps
# --------------------------------------------------

npm install

# --------------------------------------------------
# Run vite directly (avoid broken .bin wrappers)
# --------------------------------------------------

node node_modules/vite/bin/vite.js --host 0.0.0.0
