#!/usr/bin/env bash
# Synchronize only project-level camera registration data to a remote project.
#
# This script never reads from or writes to the remote videos/ directory. It:
#   1. extracts only config.json["registration"] to a temporary payload;
#   2. rsyncs only LOCAL_PROJECT/registration/;
#   3. locks remote config.json, backs it up, replaces only its registration key,
#      and atomically swaps the updated file into place;
#   4. verifies the semantic hash of remote config.json["videos"] is unchanged.
#
# Usage:
#   sync_registration_to_remote.sh LOCAL_PROJECT USER@HOST:/remote/projects/
#
# Optional connection settings:
#   SSH_OPTS="-S /path/to/control.sock"
#   RSYNC_EXTRA="--dry-run"  # preview registration-directory transfer only

set -euo pipefail

LOCAL_PROJECT="${1:-}"
REMOTE_PARENT="${2:-}"

if [[ -z "$LOCAL_PROJECT" || -z "$REMOTE_PARENT" || "$REMOTE_PARENT" != *:* ]]; then
    echo "Usage: $0 <LOCAL_PROJECT_DIR> <USER@HOST:REMOTE_PROJECTS_PARENT>" >&2
    exit 2
fi

LOCAL_PROJECT="${LOCAL_PROJECT%/}"
if [[ ! -f "$LOCAL_PROJECT/config.json" ]]; then
    echo "ERROR: missing $LOCAL_PROJECT/config.json" >&2
    exit 2
fi

REMOTE_HOST="${REMOTE_PARENT%%:*}"
REMOTE_PARENT_PATH="${REMOTE_PARENT#*:}"
REMOTE_PARENT_PATH="${REMOTE_PARENT_PATH%/}"
PROJECT_NAME="${LOCAL_PROJECT##*/}"
REMOTE_PROJECT="$REMOTE_PARENT_PATH/$PROJECT_NAME"

read -r -a SSH_ARGS <<< "${SSH_OPTS:-}"
# scp reserves -S for an alternate SSH executable, while ssh uses -S for a
# control socket. Translate control-socket pairs for scp.
SCP_ARGS=()
for ((i = 0; i < ${#SSH_ARGS[@]}; i++)); do
    if [[ "${SSH_ARGS[$i]}" == "-S" && $((i + 1)) -lt ${#SSH_ARGS[@]} ]]; then
        SCP_ARGS+=("-o" "ControlPath=${SSH_ARGS[$((i + 1))]}")
        i=$((i + 1))
    else
        SCP_ARGS+=("${SSH_ARGS[$i]}")
    fi
done

PAYLOAD="$(mktemp)"
cleanup() {
    rm -f "$PAYLOAD"
}
trap cleanup EXIT

python3 - "$LOCAL_PROJECT/config.json" "$PAYLOAD" <<'PY'
import json
import os
import sys

config_path, payload_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as fh:
    config = json.load(fh)
registration = config.get("registration")
if not isinstance(registration, dict) or not registration:
    raise SystemExit(f"ERROR: no project registration data in {config_path}")
with open(payload_path, "w", encoding="utf-8") as fh:
    json.dump(registration, fh, indent=2, sort_keys=True)
    fh.write("\n")
    fh.flush()
    os.fsync(fh.fileno())
PY

ssh "${SSH_ARGS[@]}" "$REMOTE_HOST" \
    "test -f '$REMOTE_PROJECT/config.json' && mkdir -p '$REMOTE_PROJECT/registration'"

if [[ -d "$LOCAL_PROJECT/registration" ]]; then
    RSYNC_RSH_CMD="ssh"
    if [[ ${#SSH_ARGS[@]} -gt 0 ]]; then
        printf -v SSH_JOINED ' %q' "${SSH_ARGS[@]}"
        RSYNC_RSH_CMD+=" $SSH_JOINED"
    fi
    # Only this directory is in scope. No project root or videos path is passed.
    RSYNC_RSH="$RSYNC_RSH_CMD" rsync -avh --progress ${RSYNC_EXTRA:-} \
        "$LOCAL_PROJECT/registration/" \
        "$REMOTE_HOST:$REMOTE_PROJECT/registration/"
else
    echo "WARNING: $LOCAL_PROJECT/registration does not exist; syncing config metadata only" >&2
fi

if [[ "${RSYNC_EXTRA:-}" == *"--dry-run"* ]]; then
    echo "Dry run complete; remote config.json was not modified."
    exit 0
fi

REMOTE_PAYLOAD="$REMOTE_PROJECT/registration/.registration-sync-incoming.json"
scp "${SCP_ARGS[@]}" "$PAYLOAD" "$REMOTE_HOST:$REMOTE_PAYLOAD"

ssh "${SSH_ARGS[@]}" "$REMOTE_HOST" \
    python3 - "$REMOTE_PROJECT/config.json" "$REMOTE_PAYLOAD" <<'PY'
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

config_path = Path(sys.argv[1])
payload_path = Path(sys.argv[2])
lock_path = config_path.with_name(config_path.name + ".registration.lock")

def semantic_hash(value):
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()

with lock_path.open("a+") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    with config_path.open(encoding="utf-8") as fh:
        current = json.load(fh)
    with payload_path.open(encoding="utf-8") as fh:
        registration = json.load(fh)

    videos_before = semantic_hash(current.get("videos"))
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = config_path.with_name(f"{config_path.name}.before-registration-{timestamp}")
    suffix = 1
    while backup.exists():
        backup = config_path.with_name(
            f"{config_path.name}.before-registration-{timestamp}-{suffix}"
        )
        suffix += 1
    shutil.copy2(config_path, backup)

    # This is the only mutation: every current remote key, especially videos,
    # remains sourced from the just-read Minerva config.
    current["registration"] = registration

    fd, tmp_name = tempfile.mkstemp(
        prefix=".config.registration.",
        suffix=".tmp",
        dir=config_path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, config_path.stat().st_mode)
        os.replace(tmp_name, config_path)
        dir_fd = os.open(config_path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    with config_path.open(encoding="utf-8") as fh:
        written = json.load(fh)
    videos_after = semantic_hash(written.get("videos"))
    if videos_after != videos_before:
        shutil.copy2(backup, config_path)
        raise SystemExit("ERROR: remote videos metadata changed; restored backup")

payload_path.unlink(missing_ok=True)
print(f"backup={backup}")
print(f"videos_sha256_before={videos_before}")
print(f"videos_sha256_after={videos_after}")
print("registration_merge=ok")
PY

echo "Registration-only sync complete: $PROJECT_NAME"
