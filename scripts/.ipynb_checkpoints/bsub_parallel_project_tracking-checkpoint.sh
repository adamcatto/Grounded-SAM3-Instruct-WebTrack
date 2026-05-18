#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage:"
    echo "  $0 PROJECT_DIR [extra worker args]"
    exit 1
fi

PROJECT_DIR="$1"
shift || true

EXTRA_ARGS="$*"

export PROJECT_DIR
export EXTRA_ARGS

bsub < ./bsub_parallel_project_tracking.bsub