"""Run queue-memory propagation for the forked ablation project."""

from __future__ import annotations

import sys
from pathlib import Path

from .common import REPO_ROOT, load_config

_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_pending_inference import propagation_sse  # noqa: E402


def _api_base(backend: str) -> str:
    backend = backend.rstrip("/")
    return backend if backend.endswith("/api") else backend + "/api"


def run_queue_tracking(
    project_dir: Path,
    *,
    backend: str = "http://127.0.0.1:8000",
    only_videos: list[str] | None = None,
    sse_timeout: float | None = None,
    quiet_stream: bool = False,
    force: bool = False,
) -> int:
    project_dir = project_dir.resolve()
    cfg = load_config(project_dir)
    pid = str(cfg["id"])
    api_base = _api_base(backend)
    only = set(only_videos) if only_videos else None
    todo = []
    for vid, vm in (cfg.get("videos") or {}).items():
        if only is not None and vid not in only:
            continue
        if vm.get("propagation_complete") and not force:
            print(f"[track] skip {vid}: already complete")
            continue
        todo.append((vid, vm))
    if not todo:
        print("[track] nothing to propagate")
        return 0

    print(f"[track] backend={api_base}")
    print(f"[track] project={pid} ({cfg.get('name')})")
    print("[track] mode=recent_queue_memory (use_all_anchors=false)")
    rc = 0
    for vid, vm in todo:
        print(f"\n>>> Propagate {vid} ({vm.get('name')}) with queue memory", flush=True)
        ok, last, err = propagation_sse(
            api_base,
            pid,
            vid,
            use_all_anchors=False,
            timeout_s=sse_timeout,
            stream_log=not quiet_stream,
        )
        if not ok:
            print(f"[track] ERROR {vid}: {err}", file=sys.stderr, flush=True)
            rc = 1
            continue
        refreshed = load_config(project_dir).get("videos", {}).get(vid, {})
        if refreshed.get("propagation_complete"):
            print(f">>> Done ({vid}). last_event={last!r}", flush=True)
        else:
            print(f"[track] incomplete {vid}; last_sse={last!r}", file=sys.stderr, flush=True)
            rc = 1
    return rc
