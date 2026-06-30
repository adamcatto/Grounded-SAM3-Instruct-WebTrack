#!/usr/bin/env bash
# Pull a SAM3 project FROM a remote host to this machine, without copying the
# heavy / regenerable data, then fix up the source video references so the
# project is usable locally.
#
# It does three things:
#   1. rsync the project here, EXCLUDING masks/ , bboxes/ , analysis_of_tracking_data/
#      and the source video files (source.*). masks.sqlite is a file, not the
#      masks/ folder, so it IS copied — tracked masks come across.
#   2. (optional) Re-create each video's source.* symlink by pointing at the real
#      file referenced by an existing local copy of the same project
#      (--link-sources-from). Targets are resolved (readlink -f) so you get a
#      direct link to the real video, not a symlink-to-a-symlink.
#   3. Rewrite each video's "source_path" in config.json to the LOCAL source.*
#      symlink. Without this, source_path still points at the remote machine's
#      path and on-demand frame extraction silently fails (frames 404).
#
# Usage:
#   bash scripts/sync_project_from_remote.sh \
#       user@host:/remote/path/<id>-myproject \
#       /local/dest/parent \
#       [--link-sources-from /local/existing/<id>-myproject]
#
# Example:
#   bash scripts/sync_project_from_remote.sh \
#     cattoa01@minerva.hpc.mssm.edu:/sc/arion/.../3c9bddf2-Home-Cage-Interactions-0126-test-day \
#     /opt/projects/Adam/2026/tracked_behavior_projects \
#     --link-sources-from /opt/projects/segmentation_tracking_projects/3c9bddf2-Home-Cage-Interactions-0126-test-day
#
# Env:
#   RSYNC_EXTRA="--dry-run"   preview without transferring
#   EXTRA_EXCLUDES="frames/ annotated_frames/"   space-separated extra excludes

set -euo pipefail

REMOTE="${1:-}"
DEST_PARENT="${2:-}"
shift $(( $# >= 2 ? 2 : $# )) || true

OLD_COPY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --link-sources-from) OLD_COPY="${2:-}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$REMOTE" || -z "$DEST_PARENT" ]]; then
  echo "Usage: $0 <user@host:/remote/<id>-project> <local_dest_parent> [--link-sources-from <local_existing_project>]" >&2
  exit 2
fi

BASENAME="${REMOTE##*/}"          # <id>-projectname
DEST="${DEST_PARENT%/}/${BASENAME}"

# ── 1. rsync with excludes ──────────────────────────────────────────────────
EXCLUDES=(masks/ bboxes/ analysis_of_tracking_data/ "source.*")
RSYNC_OPTS=(-avh --progress)
for e in "${EXCLUDES[@]}" ${EXTRA_EXCLUDES:-}; do
  RSYNC_OPTS+=(--exclude="$e")
done

mkdir -p "$DEST_PARENT"
echo ">> rsync ${REMOTE} -> ${DEST}"
# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" ${RSYNC_EXTRA:-} "${REMOTE%/}" "${DEST_PARENT%/}/"

if [[ "${RSYNC_EXTRA:-}" == *"--dry-run"* ]]; then
  echo ">> dry-run: skipping symlink + config rewrite"
  exit 0
fi

if [[ ! -f "${DEST}/config.json" ]]; then
  echo "ERROR: ${DEST}/config.json not found after rsync" >&2
  exit 1
fi

# ── 2. (optional) re-link source videos from an existing local copy ─────────
if [[ -n "$OLD_COPY" ]]; then
  if [[ ! -d "${OLD_COPY}/videos" ]]; then
    echo "ERROR: --link-sources-from has no videos/ dir: ${OLD_COPY}" >&2
    exit 1
  fi
  echo ">> linking source videos from ${OLD_COPY}"
  while IFS= read -r src; do
    vid_dir_name="$(basename "$(dirname "$src")")"
    dest_dir="${DEST}/videos/${vid_dir_name}"
    mkdir -p "$dest_dir"
    target="$(readlink -f "$src")"          # resolve to the real video file
    ln -sfn "$target" "${dest_dir}/source.mp4"
    [[ -e "$target" ]] && ok="ok" || ok="MISSING TARGET"
    echo "   ${vid_dir_name}/source.mp4 -> ${target} [${ok}]"
  done < <(find "${OLD_COPY}/videos" -maxdepth 2 -name 'source.*' \( -type f -o -type l \))
fi

# ── 3. rewrite config.json source_path to the local source.* symlinks ───────
echo ">> rewriting source_path in config.json"
python3 - "$DEST" <<'PY'
import json, sys, pathlib
proj = pathlib.Path(sys.argv[1])
cfg = proj / "config.json"
c = json.loads(cfg.read_text())
vroot = proj / "videos"
changed = False
for vid, vm in (c.get("videos") or {}).items():
    if not isinstance(vm, dict):
        continue
    vdir = next((d for d in vroot.iterdir()
                 if d.is_dir() and (d.name == vid or d.name.startswith(vid + "_"))), None)
    if not vdir:
        print(f"   NO DIR for {vid}")
        continue
    src = next((s for s in sorted(vdir.glob("source.*"))
                if s.is_file() or s.is_symlink()), None)
    if not src:
        print(f"   NO source.* in {vdir.name} (source_path left as-is)")
        continue
    vm["source_path"] = str(src)
    changed = True
    print(f"   {vid}: source_path -> {src} | resolves={src.resolve().exists()}")
if changed:
    cfg.write_text(json.dumps(c, indent=2))
PY

echo ">> done. Project at: ${DEST}"
echo "   Point the backend at $(dirname "${DEST}") via SAM3_PROJECTS_DIR (or move ${DEST} into your projects root)."
