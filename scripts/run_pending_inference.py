#!/usr/bin/env python3
"""
Run whole-video propagation (tracking) only for videos that are ready:

  • anchor labeling is complete (all anchor frames committed in the UI)
  • propagation_complete is False
  • whole_video_inference.status is not "running" (failed/complete/none are OK to queue when
    incomplete)

Requires the SAM backend (uvicorn server:app) to be reachable.

Examples:
  export SAM3_PROJECTS_DIR=/path/to/projects-root   # optional
  uvicorn server:app --host 127.0.0.1 --port 8000   # elsewhere
  python scripts/run_pending_inference.py … --backend http://127.0.0.1:8000 [--list-only] [--quiet-stream]

  • RUN/SKIP eligibility is written to stderr via os.write (unbuffered), so it appears
    immediately even with `conda run`; SSE progress still prints on stdout.
  • Without --quiet-stream, propagation progress is mirrored on this terminal from SSE events.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Tuple

REPO_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(REPO_BACKEND) not in sys.path:
    sys.path.insert(0, str(REPO_BACKEND))

from anchor_helpers import is_anchor_labeling_complete  # noqa: E402
from project_manager import ProjectManager  # noqa: E402


def _emit_plan_lines(lines: list[str]) -> None:
    """Write plan to stderr (fd 2) directly — bypasses Python/conda stdout block buffering."""

    for s in lines:
        blob = (s + "\n").encode("utf-8", errors="replace")
        try:
            os.write(2, blob)
        except OSError:
            try:
                sys.__stderr__.write(s + "\n")
                sys.__stderr__.flush()
            except Exception:
                pass


def _log_sse_to_terminal(ev: str, data: dict[str, Any], *, stream_active: list[bool]) -> None:
    """Print one SSE event line (or carriage-return status for progress streams)."""

    def _end_carry():
        if stream_active[0]:
            print(file=sys.stdout, flush=True)
            stream_active[0] = False

    if ev == "heartbeat":
        return
    # Collapse noisy progress streams to updating lines (same terminal row)
    if ev == "extract_progress":
        stream_active[0] = True
        ext = data.get("extracted", "?")
        print(f"\r  extract … {ext} jpegs written in batch window   ", end="", flush=True)
        return
    if ev == "progress":
        stream_active[0] = True
        fr = data.get("frame", "?")
        pr = float(data.get("progress", 0))
        print(f"\r  propagate … frame={fr}  overall {100 * pr:.1f}%       ", end="", flush=True)
        return

    _end_carry()

    if ev == "init":
        print(
            f"  [init] starting at frame={data.get('actual_start')} "
            f"batches≈{data.get('total_batches')}  frames≈{data.get('frames_to_process')}",
            flush=True,
        )
        return
    if ev == "batch_start":
        st = data.get("status", "")
        bi = data.get("batch")
        tb = data.get("total_batches")
        b_start = data.get("batch_start")
        b_end = data.get("batch_end")
        n = (int(bi) + 1) if isinstance(bi, int) else "?"
        t = tb if tb is not None else "?"
        print(
            f"  [batch {n}/{t}] frames [{b_start},{b_end}) status={st!r}",
            flush=True,
        )
        return
    if ev == "done":
        print(f"  [done] last_frame={data.get('frame')}", flush=True)
        return
    if ev == "error":
        print(f"  [error] {data.get('error')}", file=sys.stderr, flush=True)
        return
    if ev in ("catch_up", "message"):
        # optional noise reduction
        return
    # Unknown event — lightweight peek
    print(f"  [{ev}] {data}", flush=True)


def propagation_sse(
    backend_api_base: str,
    pid: str,
    vid: str,
    *,
    use_all_anchors: bool,
    timeout_s: float | None,
    stream_log: bool = True,
) -> Tuple[bool, dict[str, Any] | None, str]:
    """
    Consume GET /api/.../propagate SSE until done or error.

    Returns (ok, last_payload_dict, human_error_message).
    """
    q = urllib.parse.urlencode({"use_all_anchors": "true"} if use_all_anchors else {})
    tail = ("?" + q) if q else ""
    api = backend_api_base.rstrip("/")
    url = f"{api}/projects/{urllib.parse.quote(pid)}/videos/{urllib.parse.quote(vid)}/propagate{tail}"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    to = timeout_s if timeout_s is not None else None

    seen_done = False
    seen_err = False
    last_payload: dict[str, Any] | None = None
    stream_active_carry = [False]  # list so flush_block closure can mutate

    ev_name: str | None = None
    data_parts: list[str] = []

    def flush_block():
        nonlocal seen_done, seen_err, last_payload, ev_name, data_parts
        raw = "".join(data_parts).strip()
        cur_ev = ev_name or "message"
        data_parts = []
        ev_name = None
        if raw == "":
            return
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            d = {"_raw": raw}

        if isinstance(d, dict):
            last_payload = d
        else:
            last_payload = {"_non_dict": str(d)}

        if cur_ev == "done":
            seen_done = True
        elif cur_ev == "error":
            seen_err = True

        if stream_log and isinstance(d, dict):
            _log_sse_to_terminal(cur_ev, d, stream_active=stream_active_carry)

    try:
        with urllib.request.urlopen(req, timeout=to) as resp:
            decoder = io.TextIOWrapper(resp, encoding="utf-8", newline="")
            for line in decoder:
                if line.endswith("\r\n"):
                    line = line[:-2]
                elif line.endswith("\n"):
                    line = line[:-1]
                if line == "":
                    flush_block()
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    ev_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_parts.append(line[5:])
        if data_parts or ev_name is not None:
            flush_block()
        if stream_log and stream_active_carry[0]:
            print(file=sys.stdout, flush=True)
            stream_active_carry[0] = False
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return False, last_payload, f"HTTP {e.code}: {body or e.reason}"
    except Exception as e:
        return False, last_payload, str(e)

    ok = seen_done and not seen_err
    if not ok and not seen_err and not seen_done:
        return False, last_payload, "stream ended without done/error events"
    if seen_err:
        return False, last_payload, last_payload.get("error", "") if isinstance(last_payload, dict) else "error event"

    return True, last_payload, ""


def video_eligibility(vid: str, vm: dict) -> Tuple[bool, str]:
    inferred = is_anchor_labeling_complete(vm)
    anchored = bool(vm.get("anchor_labeling_complete")) or inferred
    if not anchored:
        return False, "anchor labeling incomplete"

    if vm.get("propagation_complete"):
        return False, "already tracked (propagation_complete)"

    winf = vm.get("whole_video_inference") or {}
    st = (winf.get("status") or "none").strip() or "none"
    if st == "running":
        return False, "inference_running (another job or stale — use --clear-stuck)"
    if st == "complete" and not vm.get("propagation_complete"):
        return False, "status complete but propagation_complete false (repair config)"

    prompts = vm.get("point_prompts") or {}
    if not any(prompts.values()):
        return False, "no point prompts in config"

    if st == "failed":
        return (
            True,
            "ready for propagation (retry after previous failed inference; anchors labeled, propagation incomplete)",
        )
    return True, "ready for propagation (anchors labeled, propagation incomplete, inference not blocking)"


def clear_stuck_running(pm: ProjectManager, pid: str, vid: str) -> None:
    pm.set_video_inference_status(pid, vid, "none", host=None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--project-dir",
        help="Directory containing config.json for one project "
        "(e.g. ~/.sam3_zero_projects/ab12-name). Overrides --pid.",
    )
    ap.add_argument(
        "--pid",
        help="Project id only (looks under SAM3_PROJECTS_DIR / ~/.sam3_zero_projects).",
    )
    ap.add_argument(
        "--backend",
        default="http://127.0.0.1:8000",
        help="HTTP origin of FastAPI backend (default %(default)s). /api prefix is optional.",
    )
    ap.add_argument(
        "--use-all-anchors",
        action="store_true",
        help="Pass use_all_anchors=true on propagate (matches all-anchors web tracking).",
    )
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help=argparse.SUPPRESS,  # obsolete: failures queue by default; kept for CLI compatibility
    )
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="After RUN/SKIP summary, exit without connecting to propagation (no SSE).",
    )
    ap.add_argument(
        "--clear-stuck",
        action="store_true",
        help="Reset whole_video_inference running→none for stuck jobs before eligibility.",
    )
    ap.add_argument(
        "--videos",
        nargs="*",
        help="Restrict to specific video ids (default: all in project).",
    )
    ap.add_argument(
        "--sse-timeout",
        type=float,
        default=None,
        help="urllib urlopen timeout in seconds (default: infinite).",
    )
    ap.add_argument(
        "--quiet-stream",
        action="store_true",
        help="Suppress per-event SSE progress on stdout (eligibility still on stderr first; done/error lines may still print).",
    )

    args = ap.parse_args()

    pm = ProjectManager()

    cfg_path_explicit: Path | None = None
    if args.project_dir:
        pd = Path(args.project_dir).expanduser().resolve()
        cfg_path = pd / "config.json"
        if not cfg_path.is_file():
            print(f"No config.json under {pd}", file=sys.stderr, flush=True)
            return 2
        cfg_path_explicit = cfg_path
        project = pm._normalize_project_config(json.loads(cfg_path.read_text()))
        pid = project["id"]
    elif args.pid:
        pid = args.pid.strip()
        project = pm.get_project(pid)
        if project is None:
            print(f"Project {pid!r} not found under SAM3 projects root.", file=sys.stderr, flush=True)
            return 2
    else:
        print("Provide --project-dir or --pid", file=sys.stderr, flush=True)
        return 2

    def _reload_project_after_disk_mutations() -> bool:
        nonlocal project, pid
        if cfg_path_explicit is not None and cfg_path_explicit.is_file():
            project = pm._normalize_project_config(json.loads(cfg_path_explicit.read_text()))
            pid = project["id"]
            return True
        refreshed = pm.get_project(pid)
        if refreshed is None:
            print(
                f"Cannot reload project {pid!r} from ProjectManager roots "
                f"(SAM3_PROJECTS_DIR). Re-run with matching env or pass --project-dir.",
                file=sys.stderr,
                flush=True,
            )
            return False
        project = refreshed
        return True

    if args.clear_stuck:
        for vid in list((project.get("videos") or {})):
            vm = project["videos"][vid]
            if (vm.get("whole_video_inference") or {}).get("status") == "running":
                clear_stuck_running(pm, pid, vid)
        if not _reload_project_after_disk_mutations():
            return 2

    vid_filter = set(args.videos) if args.videos else None
    pending: list[Tuple[str, dict[str, Any], bool, str]] = []
    filtered_out: list[str] = []

    for vid, vm in (project["videos"] or {}).items():
        if vid_filter is not None and vid not in vid_filter:
            filtered_out.append(vid)
            continue
        ok, msg = video_eligibility(vid, vm)
        pending.append((vid, vm, ok, msg))

    will_run: list[Tuple[str, str]] = []
    will_skip: list[Tuple[str, str]] = []
    for vid, vm, ok, msg in sorted(pending, key=lambda x: x[0]):
        name = (vm.get("name") or "").strip()
        label = f"{vid}" + (f" — {name}" if name else "")
        if ok:
            will_run.append((label, msg))
        else:
            will_skip.append((label, msg))

    pname = project.get("name") or ""
    plan_lines: list[str] = [
        "run_pending_inference — eligibility (next: propagation over HTTP unless --list-only)",
        "",
        "project=" + pid + (f" ({pname})" if pname else ""),
    ]
    if filtered_out:
        plan_lines.extend(
            (
                "",
                "Not considered (--videos filter; excluded: " + ", ".join(sorted(filtered_out)) + ")",
            )
        )

    plan_lines.extend(("", "── Will run whole-video inference ──"))
    if will_run:
        for label, msg in will_run:
            plan_lines.append(f"  RUN    {label}")
            plan_lines.append(f"         {msg}")
    else:
        plan_lines.append("  (none)")

    plan_lines.extend(("", "── Skipping (not running inference) ──"))
    if will_skip:
        for label, msg in will_skip:
            plan_lines.append(f"  SKIP   {label}")
            plan_lines.append(f"         reason: {msg}")
    else:
        plan_lines.append("  (none)")

    n_run = len(will_run)
    n_skip = len(will_skip)
    plan_lines.extend(
        (
            "",
            f"Summary: {n_run} to propagate, {n_skip} skipped"
            f"{(' (plus ' + str(len(filtered_out)) + ' excluded by --videos)') if filtered_out else ''}.",
        )
    )

    _emit_plan_lines(plan_lines)

    if args.list_only:
        _emit_plan_lines(["", "(--list-only: no SSE / propagation.)", ""])
        return 0

    to_run = [(vid, vm) for vid, vm, ok, _ in sorted(pending, key=lambda x: x[0]) if ok]
    if not to_run:
        _emit_plan_lines(["No videos to propagate (nothing eligible).", ""])
        return 0

    backend = args.backend.rstrip("/")
    api_base = backend if backend.endswith("/api") else backend + "/api"

    run_ids = [x[0] for x in to_run]
    _emit_plan_lines(
        [
            "",
            "── Starting propagation (SSE progress on stdout follows) ──",
            f"  videos ({len(to_run)}): {', '.join(run_ids)}",
            "",
        ]
    )

    hostname = socket.gethostname()

    exit_code = 0
    for vid, vm in to_run:
        vname = (vm.get("name") or "").strip()
        label = f"{vid}" + (f" ({vname})" if vname else "")
        print(f"\n>>> Propagate {label} …", flush=True)
        ok, last, err = propagation_sse(
            api_base,
            pid,
            vid,
            use_all_anchors=args.use_all_anchors,
            timeout_s=args.sse_timeout,
            stream_log=not args.quiet_stream,
        )
        if not ok:
            print(f"(error) SSE failed: {err}", file=sys.stderr, flush=True)
            try:
                pm.set_video_inference_status(pid, vid, "failed", host=hostname)
            except Exception:
                pass
            exit_code = 1
            continue
        refreshed = pm.get_video(pid, vid)
        if refreshed and refreshed.get("propagation_complete"):
            print(f">>> Done ({vid}) last_event={last!r}", flush=True)
        else:
            print(f">>> Incomplete for {vid}; last_sse={last!r}", file=sys.stderr, flush=True)
            try:
                pm.set_video_inference_status(pid, vid, "failed", host=hostname)
            except Exception:
                pass
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
