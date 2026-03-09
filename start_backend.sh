#!/bin/bash
source /home/adam/miniconda3/etc/profile.d/conda.sh
conda activate sam3
# Use libstdc++ from sam2_app env which has CXXABI_1.3.15 (needed by Python 3.12 libicui18n)
export LD_LIBRARY_PATH="/home/adam/miniconda3/envs/sam2_app/lib:${LD_LIBRARY_PATH}"

# Kill any stale process on port 8000
fuser -k 8000/tcp >/dev/null 2>&1; sleep 0.3

cd /opt/software/Grounded-SAM3-Instruct-WebTrack/backend
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
    