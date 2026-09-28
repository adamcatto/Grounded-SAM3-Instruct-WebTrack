#!/bin/bash
# Batch-job body: behavior quantification (clustering) for one project.
#
# Sequence features -> normalization (+ optional ComBat) -> Leiden clustering
# -> UMAP -> condition comparisons -> plots + Excel. Frame features cached by
# `hpc_submit.py features` are reused; missing ones are extracted inline.
#
#   bash scripts/hpc/behavior_quant_job.sh /path/to/project [clustering args...]
#
# Extra args go to `python -m downstream_analysis.clustering`, e.g.
# --single-animal, --resolution 0.5, --window-size 150 --stride 50.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 PROJECT_DIR [clustering args...]" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$1" && pwd)"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${SAM3_JOB_PYTHON:-python}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

echo "PROJECT_DIR=${PROJECT_DIR}  host=$(hostname)  date=$(date)"

# Tracking already validated the masks.
"${PYTHON}" -m downstream_analysis.clustering \
    --project-dir "${PROJECT_DIR}" \
    --skip-mask-verification \
    "$@"

echo "Done: $(date)"
echo "Outputs under: ${PROJECT_DIR}/analysis_of_tracking_data/clustering/"
