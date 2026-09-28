#!/bin/bash
# Batch-job body: propagate every eligible video in a project.
#
# Scheduler-agnostic — scripts/hpc_submit.py wraps this in an #SBATCH / #BSUB
# script after activating the environment. Can also be run directly on any
# GPU machine:
#
#   bash scripts/hpc/tracking_job.sh /path/to/project [worker args...]
#
# Many copies may run at once against the same project (job array); the
# worker coordinates through a claims file beside config.json.
#
# SAM3_JOB_GPUS (set by hpc_submit.py, default 1):
#   1  → start one backend on a free port, run the worker against it
#   N  → let the worker start one backend per GPU (--local-gpu-workers N)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 PROJECT_DIR [parallel_tracking_worker.py args...]" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$1" && pwd)"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${SAM3_JOB_GPUS:-1}"
PYTHON="${SAM3_JOB_PYTHON:-python}"

# The backend resolves projects by id under its projects root.
export SAM3_TRACKING_PROJECTS_DIR="$(dirname "${PROJECT_DIR}")"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
# Cluster HTTP proxies commonly intercept localhost and return 503.
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

echo "PROJECT_DIR=${PROJECT_DIR}  GPUS=${GPUS}  host=$(hostname)  date=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if (( GPUS > 1 )); then
    exec "${PYTHON}" scripts/parallel_tracking_worker.py \
        --local-gpu-workers "${GPUS}" \
        --auto-start-local-backends \
        --use-all-anchors \
        --clear-stuck \
        "${PROJECT_DIR}" \
        "$@"
fi

# ── Single GPU: own backend on an OS-assigned port ─────────────────────────
# Several jobs can land on one node; a fixed port would collide.
BACKEND_PORT="$("${PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
echo "Starting backend on 127.0.0.1:${BACKEND_PORT}"

(cd backend && exec "${PYTHON}" -m uvicorn server:app \
    --host 127.0.0.1 --port "${BACKEND_PORT}" --log-level info) &
UVICORN_PID=$!

cleanup() { kill "${UVICORN_PID}" 2>/dev/null || true; wait "${UVICORN_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# Model load on a cold shared filesystem can be slow; allow up to 20 minutes.
HEALTH_URL="http://127.0.0.1:${BACKEND_PORT}/api/health"
WAITED=0
MAX_WAIT=1200
until [[ "$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' --max-time 5 "${HEALTH_URL}" || echo 000)" == "200" ]]; do
    if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
        echo "ERROR: backend exited before becoming healthy" >&2
        exit 1
    fi
    if (( WAITED >= MAX_WAIT )); then
        echo "ERROR: backend not healthy after ${MAX_WAIT}s" >&2
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
echo "Backend healthy after ${WAITED}s"

"${PYTHON}" scripts/parallel_tracking_worker.py \
    --local-gpu-workers 0 \
    --backend "http://127.0.0.1:${BACKEND_PORT}" \
    --use-all-anchors \
    --clear-stuck \
    "${PROJECT_DIR}" \
    "$@"
