"""
Drive SAM3 propagation over the single-shot project.

Propagation is run through the backend's HTTP/SSE ``/propagate`` endpoint — the
same path the web UI and ``scripts/run_pending_inference.py`` use — so all of
SAM3's session/state workarounds (see CLAUDE.md) are reused untouched. Masks are
written to each video's ``masks.sqlite`` by the backend.

Unlike ``run_pending_inference.py`` we deliberately bypass the anchor-completeness
eligibility gate: a single-shot video has exactly one anchor on purpose, so it
would never report "anchor labeling complete". We propagate every video that is
not already complete.

Requires a running backend (``uvicorn server:app``) reachable at ``--backend``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import REPO_ROOT, load_config

# Reuse the SSE consumer from the existing CLI tracker.
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_pending_inference import propagation_sse  # noqa: E402


def _api_base(backend: str) -> str:
    backend = backend.rstrip("/")
    return backend if backend.endswith("/api") else backend + "/api"


def run_tracking(
    project_dir: Path,
    *,
    backend: str = "http://127.0.0.1:8000",
    only_videos: list[str] | None = None,
    sse_timeout: float | None = None,
    quiet_stream: bool = False,
    force: bool = False,
) -> int:
    """Propagate every (eligible) single-shot video. Returns a process exit code."""
    project_dir = project_dir.resolve()
    config = load_config(project_dir)
    pid = config["id"]
    api_base = _api_base(backend)
    only = set(only_videos) if only_videos else None

    videos = list((config.get("videos") or {}).items())
    todo = []
    for vid, vm in videos:
        if only is not None and vid not in only:
            continue
        if vm.get("propagation_complete") and not force:
            print(f">>> SKIP {vid} ({vm.get('name')}): already complete (use --force to redo)")
            continue
        todo.append((vid, vm))

    if not todo:
        print("[track] nothing to propagate.")
        return 0

    print(f"[track] backend={api_base}")
    print(f"[track] project={pid} ({config.get('name')})")
    print(f"[track] propagating {len(todo)} video(s): {', '.join(v for v, _ in todo)}")

    exit_code = 0
    for vid, vm in todo:
        label = f"{vid} ({vm.get('name')})"
        print(f"\n>>> Propagate {label} … (single anchor at frame {vm.get('start_frame')})", flush=True)
        ok, last, err = propagation_sse(
            api_base,
            pid,
            vid,
            use_all_anchors=False,
            timeout_s=sse_timeout,
            stream_log=not quiet_stream,
        )
        if not ok:
            print(f"(error) SSE failed for {vid}: {err}", file=sys.stderr, flush=True)
            exit_code = 1
            continue

        # Re-read config from disk to confirm the backend marked it complete.
        refreshed = load_config(project_dir).get("videos", {}).get(vid, {})
        if refreshed.get("propagation_complete"):
            print(f">>> Done ({vid}). last_event={last!r}", flush=True)
        else:
            print(f">>> Incomplete for {vid}; last_sse={last!r}", file=sys.stderr, flush=True)
            exit_code = 1

    return exit_code
