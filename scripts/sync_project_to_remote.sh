#!/usr/bin/env bash
# Rsync a SAM3 project directory to a remote host. Symlinks to video files are followed
# on the receiver only: --copy-links materializes file contents on the remote side while
# leaving your local tree unchanged.
#
# Usage:
#   SAM3_PROJECT=/path/to/ab12-myproject  # directory that contains config.json
#   REMOTE=user@cluster.example.com:/data/sam3_projects/
#   bash scripts/sync_project_to_remote.sh "${SAM3_PROJECT}" "${REMOTE}"
#
# The trailing slash after LOCAL copies the contents into DEST/<basename>/...
#
# Optional: RSYNC_EXTRA="--dry-run" RSYNC_DELETE=1 (use --delete-delay on remote mirror)

set -euo pipefail

SRC="${1:-}"
DEST="${2:-}"

if [[ -z "$SRC" ]] || [[ -z "$DEST" ]]; then
  echo "Usage: $0 <LOCAL_PROJECT_DIR> <REMOTE_PARENT_OR_FULL_DEST>" >&2
  echo "Example: $0 ~/.sam3_zero_projects/ab12-myproj user@host:/srv/sam3/" >&2
  exit 2
fi

if [[ ! -f "${SRC%/}/config.json" ]]; then
  echo "Expected config.json inside project directory: ${SRC}" >&2
  exit 2
fi

RSYNC_OPTS=(-avh --progress --copy-links)
if [[ "${RSYNC_DELETE:-}" == "1" ]]; then
  RSYNC_OPTS+=(--delete-delay)
fi

# shellcheck disable=SC2086
set -x
exec rsync "${RSYNC_OPTS[@]}" ${RSYNC_EXTRA:-} "${SRC%/}/" "${DEST%/}/${SRC##*/}/"
