#!/usr/bin/env python3
"""
Run whole-video propagation (tracking) from the CLI for eligible videos in one project.

Use this when anchor frames are labeled in the web UI (or on disk) but propagation has not
finished — e.g. after closing the browser, on a login node, or to retry failed jobs without
clicking "Track Objects" again. For many videos in parallel on HPC, prefer
``hpc_submit.py tracking`` (which runs ``parallel_tracking_worker.py`` over the same
HTTP/SSE propagate path).

Eligibility (per video)
  • Anchor labeling complete (``anchor_labeling_complete`` in config, or all required anchor
    frames have point prompts / mask ``.npz`` files on disk)
  • ``propagation_complete`` is false
  • At least one object has point prompts
  • ``whole_video_inference.status`` is not actively ``running`` (unless stale — see
    ``--stale-timeout``). Status ``failed`` / ``none`` queue for retry; ``complete`` with
    incomplete propagation is skipped (config repair needed)
  • Not listed in the project's ``tracked_videos_in_progress.txt`` (claimed by a parallel worker)

Requires a running SAM backend (``uvicorn server:app``) reachable at ``--backend``.

Usage:
  run_pending_inference.py (--project-dir PATH | --pid PROJECT_ID) [options]

Options (common):
  --list-only          Print RUN/SKIP eligibility summary and exit (no HTTP propagation)
  --clear-stuck        Reset ``whole_video_inference`` status ``running`` -> ``none`` before checks
  --videos VID [VID …] Restrict to specific video ids
  --use-all-anchors    Match web UI "use all anchors" propagate mode (default)
  --no-use-all-anchors Use standard recent-queue memory instead
  --quiet-stream       Suppress per-frame SSE progress on stdout (eligibility still on stderr)
  --backend URL        FastAPI origin (default http://127.0.0.1:8000; ``/api`` added if omitted)
  --stale-timeout SEC  Treat ``running`` as killed if no progress file update (default 300)

Output:
  • RUN/SKIP plan -> stderr (unbuffered via ``os.write``, so it appears immediately under
    ``conda run``)
  • Propagation SSE progress -> stdout (unless ``--quiet-stream``)

Examples:
  # Start backend elsewhere, then dry-run eligibility only:
  uvicorn server:app --host 127.0.0.1 --port 8000
  python scripts/run_pending_inference.py \\
    --project-dir /opt/projects/segmentation_tracking_projects/ab12-demo \\
    --backend http://127.0.0.1:8000 \\
    --list-only

  # Propagate all eligible videos sequentially (live progress on stdout):
  python scripts/run_pending_inference.py --pid ab12-demo --backend http://127.0.0.1:8000

  # Retry after a crashed job left status stuck at "running":
  python scripts/run_pending_inference.py \\
    --project-dir /path/to/project --clear-stuck --backend http://127.0.0.1:8000

  # One video, all anchor frames, minimal terminal noise:
  python scripts/run_pending_inference.py --pid ab12-demo \\
    --videos vid001 --use-all-anchors --quiet-stream
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Tuple

REPO_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(REPO_BACKEND) not in sys.path:
    sys.path.insert(0, str(REPO_BACKEND))

from anchor_helpers import STREAM_BATCH_SIZE, is_anchor_labeling_complete  # noqa: E402
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


# Inactivity timeout for the propagation SSE stream. A healthy propagation emits
# progress events far more often than this (per-batch cadence is ~1-3 min); a
# stream silent for this long means the backend wedged mid-propagation (e.g. a
# SAM3/CUDA stall or a dropped connection). Without a finite cap the client
# blocks in a socket read forever — the LSF job stays RUN with flat CPU and never
# re-claims, wasting a GPU. 20 min is a generous margin over legitimate gaps yet
# finite, so a stall surfaces as an error the worker loop recovers from (it
# releases the claim and moves on; the video's status becomes retryable).
DEFAULT_SSE_INACTIVITY_TIMEOUT_S = 1200.0


class _ProgressStall(Exception):
    """Raised when the SSE stream keeps heart-beating but makes no forward progress."""

    def __init__(self, elapsed_s: float) -> None:
        self.elapsed_s = elapsed_s
        super().__init__(f"no forward progress for {elapsed_s:.0f}s")


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

    ``timeout_s`` is both the per-read socket (byte-level inactivity) timeout AND
    the forward-progress stall timeout: if the stream goes silent, or if only
    heartbeats arrive with no real propagation event, for this many seconds the
    read is aborted and a stall error returned rather than blocking forever.
    ``None`` uses DEFAULT_SSE_INACTIVITY_TIMEOUT_S.

    Two distinct stall detectors are needed because the backend emits a periodic
    ``heartbeat`` event (server.py) whenever propagation makes no progress. Those
    heartbeat bytes reset the socket read timeout, so a byte-level timeout alone
    can never catch a backend that is wedged mid-propagation (it keeps
    heart-beating forever). The progress-stall clock below is reset only by
    non-heartbeat events, so it fires even while heartbeats keep flowing.

    Returns (ok, last_payload_dict, human_error_message).
    """
    q = urllib.parse.urlencode({"use_all_anchors": "true" if use_all_anchors else "false"})
    tail = ("?" + q) if q else ""
    api = backend_api_base.rstrip("/")
    url = f"{api}/projects/{urllib.parse.quote(pid)}/videos/{urllib.parse.quote(vid)}/propagate{tail}"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    # A finite per-read timeout is what prevents the infinite-hang failure mode:
    # urlopen(timeout=to) sets the socket timeout, which applies to every read on
    # the streamed response, so a silent stream raises socket.timeout instead of
    # blocking the worker forever.
    to = timeout_s if timeout_s is not None else DEFAULT_SSE_INACTIVITY_TIMEOUT_S

    seen_done = False
    seen_err = False
    last_payload: dict[str, Any] | None = None
    stream_active_carry = [False]  # list so flush_block closure can mutate
    last_progress = [time.monotonic()]  # list so flush_block closure can mutate

    ev_name: str | None = None
    data_parts: list[str] = []

    def flush_block():
        nonlocal seen_done, seen_err, last_payload, ev_name, data_parts
        raw = "".join(data_parts).strip()
        cur_ev = ev_name or "message"
        data_parts = []
        ev_name = None
        # A heartbeat means "still connected but no progress": do not reset the
        # forward-progress clock, and do not overwrite last_payload/log it — we
        # keep the last real event for diagnostics on stall.
        if cur_ev == "heartbeat":
            return
        last_progress[0] = time.monotonic()
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

    # Bypass HTTP proxy for local backend connections (HPC http_proxy breaks localhost)
    no_proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(no_proxy_handler)
    try:
        with opener.open(req, timeout=to) as resp:
            decoder = io.TextIOWrapper(resp, encoding="utf-8", newline="")
            for line in decoder:
                if line.endswith("\r\n"):
                    line = line[:-2]
                elif line.endswith("\n"):
                    line = line[:-1]
                if line == "":
                    flush_block()
                    # Forward-progress watchdog: heartbeats keep the socket read
                    # alive but do not advance last_progress, so a backend wedged
                    # mid-propagation is caught here instead of hanging forever.
                    if time.monotonic() - last_progress[0] > to:
                        raise _ProgressStall(time.monotonic() - last_progress[0])
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
    except _ProgressStall as e:
        # Progress stall: heartbeats still arriving but no real propagation event
        # for `to` seconds (backend wedged mid-propagation). Surface as a
        # recoverable error so the worker releases the claim and moves on; the
        # video stays retryable via video_eligibility.
        return (
            False,
            last_payload,
            f"SSE stream stalled: no forward progress for {e.elapsed_s:.0f}s "
            f"(heartbeats still arriving; backend wedged mid-propagation)",
        )
    except (TimeoutError, socket.timeout) as e:
        # Inactivity timeout: the stream went silent (backend wedged mid-
        # propagation). Surface as a recoverable error so the worker releases the
        # claim and moves on; the video stays retryable via video_eligibility.
        return (
            False,
            last_payload,
            f"SSE stream stalled: no data for {to:.0f}s (inactivity timeout; "
            f"backend likely wedged mid-propagation) [{e!r}]",
        )
    except Exception as e:
        return False, last_payload, str(e)

    ok = seen_done and not seen_err
    if not ok and not seen_err and not seen_done:
        return False, last_payload, "stream ended without done/error events"
    if seen_err:
        return False, last_payload, last_payload.get("error", "") if isinstance(last_payload, dict) else "error event"

    return True, last_payload, ""


def _find_video_dir_in_project(project_dir: Path, vid: str) -> Path | None:
    """Locate a video subdirectory by vid prefix under <project>/videos/."""
    vids_root = project_dir / "videos"
    if not vids_root.is_dir():
        return None
    for d in vids_root.iterdir():
        if d.is_dir() and (d.name == vid or d.name.startswith(vid + "_")):
            return d
    return None


def _latest_mtime(directory: Path, glob_pattern: str) -> float | None:
    """Return the most recent mtime among files matching *glob_pattern* in *directory*, or None."""
    newest: float | None = None
    try:
        for f in directory.glob(glob_pattern):
            mt = f.stat().st_mtime
            if newest is None or mt > newest:
                newest = mt
    except OSError:
        pass
    return newest


def _is_inference_stale(video_dir: Path, stale_timeout_s: float) -> Tuple[bool, str]:
    """Check whether a supposedly-running inference has gone silent.

    Checks: propagation_progress.txt mtime, masks/*.npz mtime.
    Returns (is_stale, human_detail).
    """
    now = time.time()
    progress_path = video_dir / "propagation_progress.txt"

    last_activity: float | None = None
    activity_source = ""

    # Check propagation progress file
    if progress_path.exists():
        mt = progress_path.stat().st_mtime
        if last_activity is None or mt > last_activity:
            last_activity = mt
            activity_source = "progress file"

    # Check newest mask file
    masks_mt = _latest_mtime(video_dir / "masks", "*.npz")
    if masks_mt is not None and (last_activity is None or masks_mt > last_activity):
        last_activity = masks_mt
        activity_source = "mask file"

    if last_activity is None:
        return True, "no progress file or masks found"

    age_s = now - last_activity
    if age_s > stale_timeout_s:
        age_min = age_s / 60
        return True, f"last {activity_source} update {age_min:.0f}m ago (threshold {stale_timeout_s / 60:.0f}m)"

    return False, f"active — last {activity_source} update {age_s:.0f}s ago"


def _anchor_labeling_complete_with_masks(
    vm: dict, video_dir: Path, batch_size: int = STREAM_BATCH_SIZE,
) -> bool:
    """Like is_anchor_labeling_complete but also counts mask .npz files on disk."""
    from anchor_helpers import compute_anchor_frames  # already at module level via STREAM_BATCH_SIZE

    start = int(vm.get("start_frame") or 0)
    num_frames = int(vm["num_frames"])
    required = compute_anchor_frames(start, num_frames, batch_size)

    annotated = set(vm.get("annotated_anchors") or [])
    for obj_prompts in (vm.get("point_prompts") or {}).values():
        if isinstance(obj_prompts, dict):
            for fk in obj_prompts:
                try:
                    annotated.add(int(fk))
                except (ValueError, TypeError):
                    pass

    masks_dir = video_dir / "masks"
    if masks_dir.is_dir():
        for f in masks_dir.iterdir():
            if f.suffix == ".npz":
                try:
                    annotated.add(int(f.stem))
                except (ValueError, TypeError):
                    pass

    return bool(required) and all(a in annotated for a in required)


def video_eligibility(
    vid: str,
    vm: dict,
    *,
    project_dir: Path | None = None,
    stale_timeout_s: float = 300,
) -> Tuple[bool, str]:
    inferred = is_anchor_labeling_complete(vm)
    anchored = bool(vm.get("anchor_labeling_complete")) or inferred
    if not anchored and project_dir is not None:
        # Fallback: check mask files on disk for missing anchor frames
        vdir = _find_video_dir_in_project(project_dir, vid)
        if vdir is not None:
            anchored = _anchor_labeling_complete_with_masks(vm, vdir)
    if not anchored:
        return False, "anchor labeling incomplete"

    if vm.get("propagation_complete"):
        return False, "already tracked (propagation_complete)"

    winf = vm.get("whole_video_inference") or {}
    st = (winf.get("status") or "none").strip() or "none"
    if st == "running":
        # Check if inference is truly running or was killed
        if project_dir is not None:
            vdir = _find_video_dir_in_project(project_dir, vid)
            if vdir is not None:
                stale, detail = _is_inference_stale(vdir, stale_timeout_s)
                if stale:
                    return (
                        True,
                        f"ready for propagation (prior inference killed/stale — {detail})",
                    )
                else:
                    return False, f"inference_running ({detail})"
        # No project_dir or couldn't find video dir — fall back to old behavior
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
        "(e.g. /opt/projects/segmentation_tracking_projects/ab12-name). Overrides --pid.",
    )
    ap.add_argument(
        "--pid",
        help="Project id only (looks under SAM3_PROJECTS_DIR or /opt/projects/segmentation_tracking_projects).",
    )
    ap.add_argument(
        "--backend",
        default="http://127.0.0.1:8000",
        help="HTTP origin of FastAPI backend (default %(default)s). /api prefix is optional.",
    )
    ua = ap.add_mutually_exclusive_group()
    ua.add_argument(
        "--use-all-anchors",
        dest="use_all_anchors",
        action="store_true",
        help="Pass use_all_anchors=true on propagate (matches all-anchors web tracking).",
    )
    ua.add_argument(
        "--no-use-all-anchors",
        dest="use_all_anchors",
        action="store_false",
        help="Pass use_all_anchors=false on propagate (standard recent-queue memory).",
    )
    ap.set_defaults(use_all_anchors=True)
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
    ap.add_argument(
        "--stale-timeout",
        type=float,
        default=300,
        help="Seconds without progress before a 'running' inference is considered killed/stale (default %(default)s).",
    )

    args = ap.parse_args()

    pm = ProjectManager()

    cfg_path_explicit: Path | None = None
    project_dir: Path | None = None
    if args.project_dir:
        pd = Path(args.project_dir).expanduser().resolve()
        cfg_path = pd / "config.json"
        if not cfg_path.is_file():
            print(f"No config.json under {pd}", file=sys.stderr, flush=True)
            return 2
        cfg_path_explicit = cfg_path
        project_dir = pd
        # Point ProjectManager at the parent so saves (--clear-stuck, status updates) work
        try:
            pm.set_projects_root(str(pd.parent))
        except Exception:
            pass  # non-fatal; direct config reads still work
        project = pm._normalize_project_config(json.loads(cfg_path.read_text()))
        pid = project["id"]
    elif args.pid:
        pid = args.pid.strip()
        project = pm.get_project(pid)
        if project is None:
            print(f"Project {pid!r} not found under SAM3 projects root.", file=sys.stderr, flush=True)
            return 2
        try:
            project_dir = pm._project_dir(pid)
        except ValueError:
            pass
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

    # Read claims file to skip videos already being tracked by another worker
    claimed_vids: set[str] = set()
    if project_dir is not None:
        for cname in ("tracked_videos_in_progress.txt",):
            cpath = project_dir / cname
            try:
                if cpath.exists():
                    from parallel_tracking_worker import _parse_claim_ids
                    claimed_vids = _parse_claim_ids(cpath.read_text(encoding="utf-8"))
            except Exception:
                pass

    stale_cleared: set[str] = set()
    for vid, vm in (project["videos"] or {}).items():
        if vid_filter is not None and vid not in vid_filter:
            filtered_out.append(vid)
            continue
        if vid in claimed_vids:
            pending.append((vid, vm, False, "already claimed (in tracked_videos_in_progress.txt)"))
            continue
        ok, msg = video_eligibility(vid, vm, project_dir=project_dir, stale_timeout_s=args.stale_timeout)
        if ok and "killed/stale" in msg:
            stale_cleared.add(vid)
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

    # Reset inference status for stale/killed jobs before starting propagation
    if stale_cleared:
        for vid in stale_cleared:
            try:
                pm.set_video_inference_status(pid, vid, "none", host=None)
            except Exception as e:
                _emit_plan_lines([f"  (warning) could not reset stale status for {vid}: {e}"])

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
