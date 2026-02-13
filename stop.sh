#!/bin/bash
# Kill any processes running on the backend (8000) and frontend (5173) ports.

kill_port() {
    local port=$1
    local pids
    pids=$(fuser "$port/tcp" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Killing process(es) on port $port: $pids"
        fuser -k "$port/tcp" >/dev/null 2>&1
        sleep 0.5
        # Force kill if still alive
        remaining=$(fuser "$port/tcp" 2>/dev/null)
        if [ -n "$remaining" ]; then
            echo "Force killing remaining on port $port: $remaining"
            fuser -k -9 "$port/tcp" >/dev/null 2>&1
        fi
    else
        echo "Port $port is already free."
    fi
}

kill_port 8000
kill_port 5173
echo "Done."
