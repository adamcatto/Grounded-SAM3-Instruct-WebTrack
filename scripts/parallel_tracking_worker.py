#!/usr/bin/env python3
"""
Embarrassingly-parallel worker: claim videos from a project, run whole-video propagation
over HTTP (same SSE path as run_pending_inference.py), coordinated via a shared claims file
beside config.json.

Multi-GPU on one node (parallel_tracking.local_gpu_workers in env.yaml):

  • Spawns one OS process per GPU. Each process uses its own backend URL
    (http://<host>:<backend_port_base + gpu_index>) so you run one uvicorn + one visible GPU
    per port (set auto_start_local_backends: true to launch those servers automatically).

  • max_concurrent_propagations_per_gpu > 1 runs multiple propagations in parallel against the
    same backend (experimental). On CUDA OOM errors, concurrency can shrink for the rest of
    the run if oom_reduce_concurrency is true. The FastAPI server is not multi-job-safe for all
    workloads — start with 1.

Submit many LSF jobs with the same project directory for cross-node parallelism; combine with
local_gpu_workers for within-node parallelism.

Claims file: one video id per line while in flight; released after success or failure.

Defaults: configs/env.yaml (see env_template.yaml) or SAM3_ENV_YAML. CLI overrides YAML.

Requirements: POSIX fcntl+flock on the claims path.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import multiprocessing
import os
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin
from typing import Any, Callable, Optional, Set


def _parse_claim_ids(raw: str) -> Set[str]:
    out: Set[str] = set()
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        vid = s.split()[0].strip()
        if vid:
            out.add(vid)
    return out


def _format_claim_file(claims: Set[str]) -> str:
    lines = [
        "# Video ids currently being tracked by parallel_tracking_worker jobs.\n",
        "# Removing a stale line frees that video for another attempt.\n",
        *[f"{v}\n" for v in sorted(claims)],
    ]
    return "".join(lines)


def _locked_claims_update(
    path: Path, fn: Callable[[Set[str]], Set[str]],
) -> Set[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            prev = fh.read()
            ids = _parse_claim_ids(prev)
            new_ids = fn(ids)
            data = _format_claim_file(new_ids)
            fh.seek(0)
            fh.truncate()
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
            return new_ids
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _load_project_from_disk(cfg_path: Path) -> tuple[dict[str, Any], str]:
    project = json.loads(cfg_path.read_text(encoding="utf-8"))
    pid = str(project.get("id") or "").strip()
    if not pid:
        raise SystemExit(f"config.json missing id: {cfg_path}")
    return project, pid


def _prefix(pid: str, gpu: int | None = None) -> str:
    if gpu is None:
        return f"[parallel_track pid={pid}] "
    return f"[parallel_track pid={pid} gpu={gpu}] "


def _looks_like_cuda_oom(msg: str) -> bool:
    m = (msg or "").lower()
    needles = ("out of memory", "cuda out of memory", "torch.cuda.outofmemoryerror", "cudnn", "cudaerror")
    return any(x in m for x in needles)


def _wait_for_health(
    origin: str,
    timeout_s: float,
    proc: subprocess.Popen | None = None,
    err_log: Path | None = None,
) -> None:
    """Wait until GET {origin}/api/health returns HTTP 200."""

    health = urljoin(origin.rstrip("/") + "/", "api/health")
    deadline = time.monotonic() + timeout_s
    first_err = ""
    last_err = ""
    attempts = 0
    last_progress = time.monotonic()
    # Bypass HTTP proxy for local health checks (HPC nodes often have
    # http_proxy set, which routes 127.0.0.1 through a proxy returning 503).
    no_proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(no_proxy_handler)
    while time.monotonic() < deadline:
        # Bail early if the child process has already exited
        if proc is not None and proc.poll() is not None:
            _print_err_log_tail(err_log)
            raise RuntimeError(
                f"backend process exited (rc={proc.returncode}) before becoming healthy ({health}): {last_err}"
            )
        try:
            with opener.open(health, timeout=5) as resp:
                code = getattr(resp, "status", None)
                if code is None and hasattr(resp, "getcode"):
                    code = resp.getcode()
                if code == 200:
                    return
                last_err = f"HTTP {code}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            last_err = str(e)
        if not first_err:
            first_err = last_err
        attempts += 1
        # Print progress every 30 seconds so users/logs can see what's happening
        now = time.monotonic()
        if now - last_progress >= 30:
            elapsed = int(now - (deadline - timeout_s))
            remaining = int(deadline - now)
            print(
                f"  health check {health}: {last_err} "
                f"({elapsed}s elapsed, {remaining}s remaining, {attempts} attempts)",
                flush=True,
            )
            last_progress = now
        time.sleep(0.5)
    _print_err_log_tail(err_log)
    raise RuntimeError(
        f"backend health check timeout ({health}): {last_err}"
        f" (first_err={first_err!r}, attempts={attempts},"
        f" proc_alive={proc.poll() is None if proc else 'N/A'})"
    )


def _print_err_log_tail(err_log: Path | None, n: int = 30) -> None:
    """Print the last *n* lines of a uvicorn error log for diagnostics."""
    if err_log is None:
        return
    try:
        if not err_log.exists():
            return
        lines = err_log.read_text(errors="replace").splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        if tail:
            print(f"--- last {len(tail)} lines of {err_log.name} ---", flush=True)
            for ln in tail:
                print(f"  {ln}", flush=True)
            print("--- end ---", flush=True)
    except OSError:
        pass


def _start_uvicorn_per_gpu(
    n: int,
    base_port: int,
    bind_host: str,
    backend_dir: Path,
) -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    py = sys.executable
    log_dir = backend_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup() -> None:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    atexit.register(_cleanup)

    for i in range(n):
        port = base_port + i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        env.setdefault("SAM3_PROJECTS_DIR", env.get("SAM3_PROJECTS_DIR", ""))
        # Mirror start_backend.sh: ensure CONDA_PREFIX/lib is on LD_LIBRARY_PATH
        # so the dynamic linker can find torch/CUDA .so files without slow NFS searches.
        conda_prefix = env.get("CONDA_PREFIX", "")
        if conda_prefix:
            ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{ld}" if ld else f"{conda_prefix}/lib"
        out_path = log_dir / f"uvicorn_gpu{i}_{port}.out"
        err_path = log_dir / f"uvicorn_gpu{i}_{port}.err"
        fout = open(out_path, "ab", buffering=0)
        ferr = open(err_path, "ab", buffering=0)
        cmd = [
            py,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            bind_host,
            "--port",
            str(port),
            "--log-level",
            "info",
        ]
        p = subprocess.Popen(
            cmd,
            cwd=str(backend_dir),
            env=env,
            stdout=fout,
            stderr=ferr,
        )
        procs.append(p)
    return procs


def _coalesce_env_backend(pt: dict[str, Any]) -> str | None:
    v = pt.get("backend_url")
    if v is not None and str(v).strip():
        return str(v).strip()
    return None


def _apply_parallel_tracking_yaml(
    args: argparse.Namespace, pt: dict[str, Any],
) -> argparse.Namespace:
    if args.backend is None:
        args.backend = _coalesce_env_backend(pt)
        args.backend = args.backend or (
            os.environ.get("SAM3_BACKEND_URL") or "http://127.0.0.1:8000"
        )
    os.environ.setdefault("SAM3_BACKEND_URL", str(args.backend).strip())

    if args.claims_file is None:
        cf = pt.get("claims_file")
        args.claims_file = (
            str(cf).strip() if cf is not None and str(cf).strip() else "tracked_videos_in_progress.txt"
        )

    if getattr(args, "local_gpu_workers", None) is None:
        lg = pt.get("local_gpu_workers")
        args.local_gpu_workers = int(lg) if lg is not None else 0

    if getattr(args, "backend_port_base", None) is None:
        bp = pt.get("backend_port_base")
        args.backend_port_base = int(bp) if bp is not None else 8810

    if getattr(args, "local_backend_bind_host", None) is None:
        bh = pt.get("local_backend_bind_host")
        args.local_backend_bind_host = str(bh).strip() if bh is not None else "127.0.0.1"

    if getattr(args, "auto_start_local_backends", None) is None:
        ab = pt.get("auto_start_local_backends")
        args.auto_start_local_backends = bool(ab) if ab is not None else False

    if getattr(args, "backend_startup_timeout_seconds", None) is None:
        bt = pt.get("backend_startup_timeout_seconds")
        args.backend_startup_timeout_seconds = float(bt) if bt is not None else 600.0

    if getattr(args, "max_concurrent_propagations_per_gpu", None) is None:
        mc = pt.get("max_concurrent_propagations_per_gpu")
        args.max_concurrent_propagations_per_gpu = int(mc) if mc is not None else 1

    if getattr(args, "oom_reduce_concurrency", None) is None:
        orc = pt.get("oom_reduce_concurrency")
        args.oom_reduce_concurrency = bool(orc) if orc is not None else True

    if args.use_all_anchors is None:
        ua = pt.get("use_all_anchors")
        args.use_all_anchors = bool(ua) if ua is not None else True

    if args.clear_stuck is None:
        cs = pt.get("clear_stuck")
        args.clear_stuck = bool(cs) if cs is not None else False

    if args.sse_timeout is None:
        st = pt.get("sse_timeout_seconds")
        args.sse_timeout = float(st) if st is not None else None

    if args.sse_quiet is None:
        q = pt.get("quiet_sse_stream")
        args.sse_quiet = bool(q) if q is not None else False

    if args.idle_sleep_min is None:
        v = pt.get("idle_sleep_min_seconds")
        args.idle_sleep_min = float(v) if v is not None else 15.0

    if args.idle_sleep_max is None:
        v = pt.get("idle_sleep_max_seconds")
        args.idle_sleep_max = float(v) if v is not None else 120.0

    if args.idle_loops_before_exit is None:
        v = pt.get("idle_loops_before_exit")
        args.idle_loops_before_exit = int(v) if v is not None else 288

    return args


def _claim_next_video_update(
    claims_path: Path,
    cfg_path: Path,
    vid_allow: Optional[Set[str]],
    video_eligibility: Callable[..., Any],
    claimed_ref: list[str | None],
) -> None:

    def prune_and_try_claim(existing: Set[str]) -> Set[str]:
        p = _load_project_from_disk(cfg_path)[0]
        cur = set(existing)
        for vid in list(cur):
            vm = (p.get("videos") or {}).get(vid)
            if vm and vm.get("propagation_complete"):
                cur.discard(vid)

        ordered = sorted((p.get("videos") or {}).keys())
        for vid in ordered:
            if vid_allow is not None and vid not in vid_allow:
                continue
            vm = (p.get("videos") or {}).get(vid) or {}
            ok, _ = video_eligibility(vid, vm)
            if not ok:
                continue
            if vm.get("propagation_complete"):
                continue
            if vid in cur:
                continue
            claimed_ref[0] = vid
            cur.add(vid)
            break
        return cur

    claimed_ref[0] = None
    _locked_claims_update(claims_path, prune_and_try_claim)


def run_gpu_worker_process(payload: dict[str, Any]) -> int:
    """Entry point for multiprocessing (spawn-safe)."""

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "backend"))

    cfg_path = Path(payload["cfg_path"])
    project_dir = cfg_path.parent
    os.environ.setdefault("SAM3_PROJECTS_DIR", str(project_dir.parent))

    from project_manager import ProjectManager  # noqa: E402
    from run_pending_inference import propagation_sse, video_eligibility  # noqa: E402

    claims_path = Path(payload["claims_path"])
    pid = payload["project_id"]
    gpu_index = int(payload["gpu_index"])
    backend = str(payload["backend"]).strip().rstrip("/")
    api_base = backend if backend.endswith("/api") else backend + "/api"

    vid_allow = set(payload["videos"]) if payload.get("videos") else None
    hostname = socket.gethostname()

    propagate_kw = dict(
        use_all_anchors=bool(payload["use_all_anchors"]),
        timeout_s=payload.get("sse_timeout"),
        stream_log=not bool(payload.get("sse_quiet")),
    )
    idle_limit = int(payload["idle_loops_before_exit"])
    idle_min = float(payload["idle_sleep_min"])
    idle_max = float(payload["idle_sleep_max"])
    mx_conc = max(1, int(payload["max_concurrent_propagations_per_gpu"]))
    oom_shrink = bool(payload.get("oom_reduce_concurrency", True))
    clear_stuck = bool(payload.get("clear_stuck", False))

    pm = ProjectManager()
    if clear_stuck:
        proj0 = _load_project_from_disk(cfg_path)[0]
        for vid in list((proj0.get("videos") or {})):
            vm = proj0["videos"][vid]
            if (vm.get("whole_video_inference") or {}).get("status") == "running":
                pm.set_video_inference_status(pid, vid, "none", host=None)

    def reload_project() -> dict[str, Any]:
        return _load_project_from_disk(cfg_path)[0]

    def any_eligible_work(project: dict[str, Any]) -> bool:
        try:
            claimed = _parse_claim_ids(claims_path.read_text(encoding="utf-8")) if claims_path.exists() else set()
        except OSError:
            claimed = set()
        for vid, vm in sorted((project.get("videos") or {}).items()):
            if vid_allow is not None and vid not in vid_allow:
                continue
            if vid in claimed:
                continue
            ok, _ = video_eligibility(vid, vm)
            if ok:
                return True
        return False

    pr = lambda: _prefix(pid, gpu_index)

    if mx_conc <= 1:
        return _run_sequential_claim_loop(
            pr=pr,
            claims_path=claims_path,
            cfg_path=cfg_path,
            vid_allow=vid_allow,
            video_eligibility=video_eligibility,
            propagation_sse=propagation_sse,
            pid=pid,
            pm=pm,
            api_base=api_base,
            propagate_kw=propagate_kw,
            hostname=hostname,
            idle_limit=idle_limit,
            idle_min=idle_min,
            idle_max=idle_max,
            reload_project=reload_project,
            any_eligible_work=any_eligible_work,
        )

    return _run_pool_claim_loop(
        pr=pr,
        claims_path=claims_path,
        cfg_path=cfg_path,
        vid_allow=vid_allow,
        video_eligibility=video_eligibility,
        pid=pid,
        pm=pm,
        api_base=api_base,
        propagate_kw=propagate_kw,
        hostname=hostname,
        idle_limit=idle_limit,
        idle_min=idle_min,
        idle_max=idle_max,
        reload_project=reload_project,
        any_eligible_work=any_eligible_work,
        mx_conc=mx_conc,
        oom_shrink=oom_shrink,
    )


def _run_sequential_claim_loop(
    *,
    pr: Callable[[], str],
    claims_path: Path,
    cfg_path: Path,
    vid_allow: Optional[Set[str]],
    video_eligibility: Callable[..., Any],
    propagation_sse: Callable[..., Any],
    pid: str,
    pm: Any,
    api_base: str,
    propagate_kw: dict[str, Any],
    hostname: str,
    idle_limit: int,
    idle_min: float,
    idle_max: float,
    reload_project: Callable[[], dict[str, Any]],
    any_eligible_work: Callable[[dict[str, Any]], bool],
) -> int:

    idle_streak = 0
    exit_code = 0
    claimed_ref: list[str | None]

    while idle_streak < idle_limit:
        project = reload_project()
        claimed_ref = [None]
        _claim_next_video_update(claims_path, cfg_path, vid_allow, video_eligibility, claimed_ref)
        claimed_vid = claimed_ref[0]

        if claimed_vid is None:
            if not any_eligible_work(project):
                print(pr() + "No eligible propagation work left — exiting.", flush=True)
                break
            idle_streak += 1
            mn, mx = idle_min, idle_max
            if mn > mx:
                mn, mx = mx, mn
            delay = random.uniform(mn, mx)
            print(
                f"{pr()}Idle sweep #{idle_streak}: eligible work remains but none "
                f"claimable; sleeping {delay:.1f}s …",
                flush=True,
            )
            time.sleep(delay)
            continue

        idle_streak = 0
        vid_str = claimed_vid

        cv = vid_str

        def release_claim(ids: Set[str]) -> Set[str]:
            cur = set(ids)
            cur.discard(cv)
            return cur

        print(f"{pr()}Claimed {vid_str}; starting SSE propagation …", flush=True)
        ok = False
        last = None
        err = ""
        try:
            ok, last, err = propagation_sse(api_base, pid, vid_str, **propagate_kw)
        finally:
            _locked_claims_update(claims_path, release_claim)

        refreshed = pm.get_video(pid, vid_str)
        if ok and refreshed and refreshed.get("propagation_complete"):
            print(f"{pr()}Done {vid_str} last_sse={last!r}", flush=True)
        else:
            print(f"{pr()}Failed {vid_str}: {err}", file=sys.stderr, flush=True)
            try:
                pm.set_video_inference_status(pid, vid_str, "failed", host=hostname)
            except Exception:
                pass
            exit_code = 1

    return exit_code


def _run_pool_claim_loop(
    *,
    pr: Callable[[], str],
    claims_path: Path,
    cfg_path: Path,
    vid_allow: Optional[Set[str]],
    video_eligibility: Callable[..., Any],
    pid: str,
    pm: Any,
    api_base: str,
    propagate_kw: dict[str, Any],
    hostname: str,
    idle_limit: int,
    idle_min: float,
    idle_max: float,
    reload_project: Callable[[], dict[str, Any]],
    any_eligible_work: Callable[[dict[str, Any]], bool],
    mx_conc: int,
    oom_shrink: bool,
) -> int:
    from run_pending_inference import propagation_sse  # noqa: E402

    exit_code = 0
    idle_streak = 0
    dyn_limit_cell: list[int] = [mx_conc]
    lock = threading.Lock()
    futures: dict[str, Future[Any]] = {}

    def do_one(vid_str: str) -> tuple[str, bool, str]:
        cv = vid_str

        def release_claim(ids: Set[str]) -> Set[str]:
            cur = set(ids)
            cur.discard(cv)
            return cur

        ok = False
        err = ""
        try:
            print(f"{pr()}Claimed {vid_str}; starting SSE propagation …", flush=True)
            ok, _last, err = propagation_sse(api_base, pid, vid_str, **propagate_kw)
        finally:
            _locked_claims_update(claims_path, release_claim)

        if not ok and oom_shrink and _looks_like_cuda_oom(err):
            with lock:
                if dyn_limit_cell[0] > 1:
                    dyn_limit_cell[0] -= 1
                    print(
                        f"{pr()}CUDA OOM detected — reducing concurrent propagations to "
                        f"{dyn_limit_cell[0]} for this GPU worker.",
                        file=sys.stderr,
                        flush=True,
                    )
        return vid_str, ok, err

    executor = ThreadPoolExecutor(max_workers=mx_conc)

    try:
        while idle_streak < idle_limit:
            project = reload_project()

            done = [v for v, fut in list(futures.items()) if fut.done()]
            for v in done:
                fut = futures.pop(v)
                try:
                    vid_str, ok, err = fut.result()
                except Exception as ex:
                    print(f"{pr()}Task error {v}: {ex}", file=sys.stderr, flush=True)
                    exit_code = 1
                    continue
                refreshed = pm.get_video(pid, vid_str)
                if ok and refreshed and refreshed.get("propagation_complete"):
                    print(f"{pr()}Done {vid_str}", flush=True)
                else:
                    print(f"{pr()}Failed {vid_str}: {err}", file=sys.stderr, flush=True)
                    try:
                        pm.set_video_inference_status(pid, vid_str, "failed", host=hostname)
                    except Exception:
                        pass
                    exit_code = 1

            slot_cap = dyn_limit_cell[0]
            while len(futures) < slot_cap:
                claimed_ref = [None]
                _claim_next_video_update(claims_path, cfg_path, vid_allow, video_eligibility, claimed_ref)
                vid_new = claimed_ref[0]
                if vid_new is None:
                    break
                futures[vid_new] = executor.submit(do_one, vid_new)

            if len(futures) == 0:
                if not any_eligible_work(project):
                    print(pr() + "No eligible propagation work left — exiting.", flush=True)
                    break
                idle_streak += 1
                mn, mx = idle_min, idle_max
                if mn > mx:
                    mn, mx = mx, mn
                delay = random.uniform(mn, mx)
                print(
                    f"{pr()}Idle sweep #{idle_streak}: nothing running; sleeping {delay:.1f}s …",
                    flush=True,
                )
                time.sleep(delay)
            else:
                idle_streak = 0
                time.sleep(0.2)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    return exit_code


def _supervise_local_gpu_workers(
    *,
    args: argparse.Namespace,
    project_dir: Path,
    cfg_path: Path,
    repo: Path,
) -> int:
    n = max(1, int(args.local_gpu_workers))
    base = int(args.backend_port_base)
    bind = str(args.local_backend_bind_host).strip()
    timeout = float(args.backend_startup_timeout_seconds)

    if args.clear_stuck:
        sys.path.insert(0, str(repo / "backend"))
        from project_manager import ProjectManager  # noqa: E402

        pm = ProjectManager()
        project, pid = _load_project_from_disk(cfg_path)
        for vid in list((project.get("videos") or {})):
            vm = project["videos"][vid]
            if (vm.get("whole_video_inference") or {}).get("status") == "running":
                pm.set_video_inference_status(pid, vid, "none", host=None)

    uvicorn_procs: list[subprocess.Popen] = []
    if args.auto_start_local_backends:
        backend_dir = repo / "backend"
        # Kill any stale process on target ports before starting
        for i in range(n):
            port = base + i
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                result = s.connect_ex((bind, port))
                s.close()
                if result == 0:
                    print(f"WARNING: port {port} already in use — attempting to free it", flush=True)
                    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
                    time.sleep(1)
            except Exception:
                pass
        # Truncate large log files to prevent NFS I/O stalls
        log_dir = repo / "logs"
        for i in range(n):
            port = base + i
            for suffix in ("out", "err"):
                lf = log_dir / f"uvicorn_gpu{i}_{port}.{suffix}"
                try:
                    if lf.exists() and lf.stat().st_size > 10_000_000:
                        print(f"Truncating large log {lf.name} ({lf.stat().st_size // 1_000_000}MB)", flush=True)
                        lf.write_bytes(b"")
                except OSError:
                    pass
        print(f"Starting {n} uvicorn processes (ports {base}..{base + n - 1}) …", flush=True)
        uvicorn_procs = _start_uvicorn_per_gpu(n, base, bind, backend_dir)
        for i in range(n):
            port = base + i
            origin = f"http://{bind}:{port}"
            proc = uvicorn_procs[i] if i < len(uvicorn_procs) else None
            err_log = log_dir / f"uvicorn_gpu{i}_{port}.err"
            _wait_for_health(origin, timeout, proc=proc, err_log=err_log)
        print("All local backends healthy.", flush=True)

    project, pid = _load_project_from_disk(cfg_path)
    ctx = multiprocessing.get_context("spawn")
    children: list[multiprocessing.Process] = []

    for i in range(n):
        port = base + i
        origin = f"http://{bind}:{port}"
        payload = {
            "cfg_path": str(cfg_path),
            "claims_path": str(project_dir / args.claims_file),
            "project_id": pid,
            "gpu_index": i,
            "backend": origin,
            "use_all_anchors": args.use_all_anchors,
            "sse_timeout": args.sse_timeout,
            "sse_quiet": args.sse_quiet,
            "idle_loops_before_exit": args.idle_loops_before_exit,
            "idle_sleep_min": args.idle_sleep_min,
            "idle_sleep_max": args.idle_sleep_max,
            "videos": list(args.videos) if args.videos else None,
            "max_concurrent_propagations_per_gpu": args.max_concurrent_propagations_per_gpu,
            "oom_reduce_concurrency": args.oom_reduce_concurrency,
            "clear_stuck": False,
        }
        p = ctx.Process(target=_child_entry, args=(payload,))
        p.start()
        children.append(p)

    codes = []
    for p in children:
        p.join()
        codes.append(p.exitcode if p.exitcode is not None else 1)

    for p in uvicorn_procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()

    return max(codes) if codes else 1


def _child_entry(payload: dict[str, Any]) -> None:
    rc = run_gpu_worker_process(payload)
    if not isinstance(rc, int):
        rc = 1 if rc else 0
    os._exit(rc)


def main() -> int:
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--env-yaml",
        dest="env_yaml",
        type=str,
        default="",
        metavar="PATH",
        help="YAML config path (or set SAM3_ENV_YAML). Default: <repo>/configs/env.yaml if present.",
    )
    pf, argv_rest = pre.parse_known_args(sys.argv[1:])
    if pf.env_yaml.strip():
        os.environ["SAM3_ENV_YAML"] = pf.env_yaml.strip()

    from repo_env_yaml import load_env_yaml_raw, parallel_tracking_section  # noqa: E402

    pt_yaml = parallel_tracking_section(load_env_yaml_raw())

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[pre],
    )
    ap.set_defaults(backend=None)
    ap.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing config.json.",
    )
    ap.add_argument(
        "--claims-file",
        metavar="NAME",
        default=None,
        help="Claims filename inside project-dir (YAML default tracked_videos_in_progress.txt).",
    )
    ap.add_argument(
        "--backend",
        default=None,
        help="Backend HTTP origin when local_gpu_workers=0. With local GPU workers, port base is used instead.",
    )
    ap.add_argument(
        "--local-gpu-workers",
        type=int,
        default=None,
        metavar="N",
        help="Spawn N processes (one per GPU), each using backend_port_base + index. 0 = single-process mode.",
    )
    ap.add_argument(
        "--backend-port-base",
        type=int,
        default=None,
        metavar="PORT",
        help="First uvicorn port for GPU 0 (default from YAML or 8810).",
    )
    ap.add_argument(
        "--local-backend-bind-host",
        type=str,
        default=None,
        help="Host for auto-started uvicorn and worker backend URLs (default 127.0.0.1).",
    )
    auto = ap.add_mutually_exclusive_group()
    auto.add_argument(
        "--auto-start-local-backends",
        dest="auto_start_local_backends",
        action="store_true",
        help="Launch N uvicorn processes with CUDA_VISIBLE_DEVICES before workers.",
    )
    auto.add_argument(
        "--no-auto-start-local-backends",
        dest="auto_start_local_backends",
        action="store_false",
        help="Do not start uvicorn (you must run servers yourself on each port).",
    )
    ap.set_defaults(auto_start_local_backends=None)
    ap.add_argument(
        "--backend-startup-timeout",
        type=float,
        default=None,
        help="Seconds to wait for each /api/health after auto-start (YAML or 180).",
    )
    ap.add_argument(
        "--max-concurrent-per-gpu",
        type=int,
        default=None,
        metavar="K",
        help="Parallel propagations per GPU worker (experimental; default 1).",
    )
    oom_g = ap.add_mutually_exclusive_group()
    oom_g.add_argument(
        "--oom-reduce-concurrency",
        dest="oom_reduce_concurrency",
        action="store_true",
        help="On CUDA OOM, lower per-GPU concurrency for the rest of the run.",
    )
    oom_g.add_argument(
        "--no-oom-reduce-concurrency",
        dest="oom_reduce_concurrency",
        action="store_false",
        help="Disable OOM concurrency reduction.",
    )
    ap.set_defaults(oom_reduce_concurrency=None)

    ua = ap.add_mutually_exclusive_group()
    ua.add_argument(
        "--use-all-anchors",
        dest="use_all_anchors",
        action="store_true",
        help="Propagate with use_all_anchors=true.",
    )
    ua.add_argument(
        "--no-use-all-anchors",
        dest="use_all_anchors",
        action="store_false",
        help="Propagate without use_all_anchors.",
    )
    ap.set_defaults(use_all_anchors=None)
    cs = ap.add_mutually_exclusive_group()
    cs.add_argument(
        "--clear-stuck",
        dest="clear_stuck",
        action="store_true",
        help="Clear whole_video_inference running→none before looping (supervisor once if multi-GPU).",
    )
    cs.add_argument(
        "--no-clear-stuck",
        dest="clear_stuck",
        action="store_false",
        help="Do not reset stuck running status.",
    )
    ap.set_defaults(clear_stuck=None)
    qs = ap.add_mutually_exclusive_group()
    qs.add_argument(
        "--quiet-stream",
        dest="sse_quiet",
        action="store_true",
        help="Suppress SSE progress on stdout.",
    )
    qs.add_argument(
        "--verbose-stream",
        dest="sse_quiet",
        action="store_false",
        help="Mirror SSE progress (default unless YAML disables).",
    )
    ap.set_defaults(sse_quiet=None)
    ap.add_argument(
        "--videos",
        nargs="*",
        metavar="VIDEO_ID",
        help="Restrict to these video ids (default: all in project).",
    )
    ap.add_argument(
        "--sse-timeout",
        type=float,
        default=None,
        help="urllib urlopen timeout seconds (default YAML or infinite).",
    )
    ap.add_argument(
        "--idle-sleep-min",
        type=float,
        default=None,
        help="Idle sweep min backoff seconds.",
    )
    ap.add_argument(
        "--idle-sleep-max",
        type=float,
        default=None,
        help="Idle sweep max backoff seconds.",
    )
    ap.add_argument(
        "--idle-loops-before-exit",
        type=int,
        default=None,
        help="Consecutive idle sweeps before exiting.",
    )

    ap.set_defaults(
        local_gpu_workers=None,
        backend_port_base=None,
        local_backend_bind_host=None,
        backend_startup_timeout_seconds=None,
        max_concurrent_propagations_per_gpu=None,
        oom_reduce_concurrency=None,
    )

    args = ap.parse_args(argv_rest)
    args = _apply_parallel_tracking_yaml(args, pt_yaml)

    project_dir = args.project_dir.expanduser().resolve()
    cfg_path = project_dir / "config.json"
    if not cfg_path.is_file():
        print(f"No config.json: {cfg_path}", file=sys.stderr)
        return 2

    os.environ.setdefault("SAM3_PROJECTS_DIR", str(project_dir.parent))
    repo = Path(__file__).resolve().parents[1]

    if args.local_gpu_workers > 0:
        return _supervise_local_gpu_workers(args=args, project_dir=project_dir, cfg_path=cfg_path, repo=repo)

    sys.path.insert(0, str(repo / "backend"))
    from project_manager import ProjectManager  # noqa: E402
    from run_pending_inference import propagation_sse, video_eligibility  # noqa: E402

    claims_path = project_dir / args.claims_file
    vid_allow = set(args.videos) if args.videos else None

    pm = ProjectManager()
    project, pid = _load_project_from_disk(cfg_path)

    if args.clear_stuck:
        for vid in list((project.get("videos") or {})):
            vm = project["videos"][vid]
            if (vm.get("whole_video_inference") or {}).get("status") == "running":
                pm.set_video_inference_status(pid, vid, "none", host=None)

    def reload_project() -> dict[str, Any]:
        return _load_project_from_disk(cfg_path)[0]

    def any_eligible_work(project: dict[str, Any]) -> bool:
        try:
            claimed = _parse_claim_ids(claims_path.read_text(encoding="utf-8")) if claims_path.exists() else set()
        except OSError:
            claimed = set()
        for vid, vm in sorted((project.get("videos") or {}).items()):
            if vid_allow is not None and vid not in vid_allow:
                continue
            if vid in claimed:
                continue
            ok, _ = video_eligibility(vid, vm)
            if ok:
                return True
        return False

    backend = args.backend.strip().rstrip("/")
    api_base = backend if backend.endswith("/api") else backend + "/api"
    hostname = socket.gethostname()
    propagate_kw = dict(
        use_all_anchors=args.use_all_anchors,
        timeout_s=args.sse_timeout,
        stream_log=not args.sse_quiet,
    )

    pr = lambda: _prefix(pid, None)

    if args.max_concurrent_propagations_per_gpu <= 1:
        return _run_sequential_claim_loop(
            pr=pr,
            claims_path=claims_path,
            cfg_path=cfg_path,
            vid_allow=vid_allow,
            video_eligibility=video_eligibility,
            propagation_sse=propagation_sse,
            pid=pid,
            pm=pm,
            api_base=api_base,
            propagate_kw=propagate_kw,
            hostname=hostname,
            idle_limit=args.idle_loops_before_exit,
            idle_min=args.idle_sleep_min,
            idle_max=args.idle_sleep_max,
            reload_project=reload_project,
            any_eligible_work=any_eligible_work,
        )

    return _run_pool_claim_loop(
        pr=pr,
        claims_path=claims_path,
        cfg_path=cfg_path,
        vid_allow=vid_allow,
        video_eligibility=video_eligibility,
        pid=pid,
        pm=pm,
        api_base=api_base,
        propagate_kw=propagate_kw,
        hostname=hostname,
        idle_limit=args.idle_loops_before_exit,
        idle_min=args.idle_sleep_min,
        idle_max=args.idle_sleep_max,
        reload_project=reload_project,
        any_eligible_work=any_eligible_work,
        mx_conc=args.max_concurrent_propagations_per_gpu,
        oom_shrink=args.oom_reduce_concurrency,
    )


if __name__ == "__main__":
    raise SystemExit(main())
