#!/bin/bash
echo "Starting SAM3 Web Tracker..."
bash /opt/software/Grounded-SAM3-Instruct-WebTrack/start_backend.sh &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
bash /opt/software/Grounded-SAM3-Instruct-WebTrack/start_frontend.sh
kill $BACKEND_PID 2>/dev/null
