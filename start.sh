#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Starting SAM3 Web Tracker..."
bash "$SCRIPT_DIR/start_backend.sh" &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
bash "$SCRIPT_DIR/start_frontend.sh"
kill $BACKEND_PID 2>/dev/null
