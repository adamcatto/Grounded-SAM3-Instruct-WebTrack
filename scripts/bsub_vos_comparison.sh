#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One-command vos_comparison launcher for LSF/Minerva:
#   1. build  the single_shot_vos_<project> (inline; fast, CPU, just symlinks)
#   2. submit a GPU tracking job ARRAY (workers split videos via a claims file)
#   3. submit one CPU evaluate+figures "master" job that waits for the whole array
#
# Usage:
#   scripts/bsub_vos_comparison.sh SOURCE_PROJECT [N_JOBS] [BATCH_SIZE] [THRESHOLD] [MAX_CONCURRENT]
#
#   SOURCE_PROJECT  path / folder name / short id of the anchor-based project
#   N_JOBS          number of array elements (parallel workers)  (default 4)
#   BATCH_SIZE      inherit | single | <int>                     (default inherit)
#   THRESHOLD       IoU collapse threshold for evaluation        (default 0.5)
#   MAX_CONCURRENT  cap on simultaneously RUNNING array elements (default: no cap)
#
# Example (8 workers, at most 4 running at once):
#   scripts/bsub_vos_comparison.sh \
#     /sc/arion/projects/KennyComputational/Behavior/projects/3c9bddf2-Home-Cage-Interactions-0126-test-day \
#     8 inherit 0.5 4
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '2,30p' "$0"
    exit 1
fi

SOURCE_PROJECT="$1"
N_JOBS="${2:-4}"
BATCH_SIZE="${3:-inherit}"
THRESHOLD="${4:-0.5}"
MAX_CONCURRENT="${5:-}"

REPO_ROOT="/sc/arion/projects/KennyComputational/Behavior/Grounded-SAM3-Instruct-WebTrack"
SCRIPT_DIR="${REPO_ROOT}/scripts"

cd "${REPO_ROOT}"

# ── env for the inline build (needs the sam3 env: backend mask_store import) ──
source /sc/arion/work/cattoa01/miniconda3/etc/profile.d/conda.sh
conda activate /sc/arion/work/cattoa01/miniconda3/envs/sam3
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ── 1. Build (writes the emitted project dir to a temp file) ─────────────────
PATH_FILE="$(mktemp /tmp/vos_ss_dir.XXXXXX)"
trap 'rm -f "${PATH_FILE}"' EXIT

echo "── Building single-shot project from: ${SOURCE_PROJECT}"
python -m experiments.vos_comparison build \
    --source-project "${SOURCE_PROJECT}" \
    --batch-size "${BATCH_SIZE}" \
    --emit-path-file "${PATH_FILE}"

PROJECT_DIR="$(head -n1 "${PATH_FILE}" | tr -d '[:space:]')"
if [[ -z "${PROJECT_DIR}" || ! -d "${PROJECT_DIR}" ]]; then
    echo "ERROR: build did not produce a valid project dir (got: '${PROJECT_DIR}')" >&2
    exit 1
fi
echo "── Single-shot project: ${PROJECT_DIR}"

# Unique LSF job-name tag so the eval job depends only on THIS run's trackers.
TAG="$(basename "${PROJECT_DIR}" | tr -cd 'A-Za-z0-9_')_$$"
TRACK_NAME="vost_${TAG}"
EVAL_NAME="vose_${TAG}"

export PROJECT_DIR THRESHOLD

# ── 2. Submit the tracking job ARRAY (elements split videos via claims file) ──
ARRAY_SPEC="${TRACK_NAME}[1-${N_JOBS}]"
if [[ -n "${MAX_CONCURRENT}" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi
echo "── Submitting tracking array as '${ARRAY_SPEC}'"
bsub -J "${ARRAY_SPEC}" < "${SCRIPT_DIR}/bsub_vos_comparison_track.bsub"

# ── 3. Submit the eval+figures master, gated on the whole array finishing ────
# 'ended' (not 'done') fires once ALL array elements have ended, regardless of
# exit status — so one failed element can't leave the master pending forever
# (evaluation scores whatever masks were produced).
echo "── Submitting eval+figures master as '${EVAL_NAME}' (waits for '${TRACK_NAME}')"
bsub -J "${EVAL_NAME}" -w "ended(\"${TRACK_NAME}\")" \
     < "${SCRIPT_DIR}/bsub_vos_comparison_eval.bsub"

echo
echo "Submitted. Monitor with:  bjobs -w"
echo "Results will land in:     ${PROJECT_DIR}/vos_comparison_results/"
