#!/bin/bash
# Find the nvm node 22 bin directory and put it first in PATH
NODE_BIN="$(ls -d /home/adam/.nvm/versions/node/v22*/bin 2>/dev/null | sort -V | tail -1)"
if [ -z "$NODE_BIN" ]; then
    echo "ERROR: Node v22 not found in ~/.nvm/versions/node/"
    exit 1
fi
export PATH="$NODE_BIN:$PATH"
echo "Using $(node --version)"

# Kill any stale process on port 5173
fuser -k 5173/tcp >/dev/null 2>&1; sleep 0.3

cd /opt/software/SAM3WebTrack/frontend
npm install
# Run vite directly with the explicit node binary
node node_modules/.bin/vite --host 0.0.0.0
