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

# Support the established lab/VPN workflow by listening on every interface.
# The app has no authentication, so use FRONTEND_HOST=127.0.0.1 when the host
# is not on a trusted network and access it through an SSH tunnel instead.
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
if [ "$FRONTEND_HOST" != "127.0.0.1" ] && [ "$FRONTEND_HOST" != "localhost" ]; then
    echo "WARNING: Frontend has no authentication and will listen on $FRONTEND_HOST:5173"
fi
node node_modules/vite/bin/vite.js --host "$FRONTEND_HOST"
