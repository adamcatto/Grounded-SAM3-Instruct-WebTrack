#!/usr/bin/env python3
"""
Submit parallel_tracking_worker.py through IBM Spectrum LSF bsub, driven by
configs/env.yaml (copy from configs/env_template.yaml; gitignored per-repo copy).

The YAML `lsf_parallel_tracking` block supplies -J, -P, -q, -R, -gpu, -W, log files (-o/-e or
-oo/-eo), submit_count, conda python, etc.
The `parallel_tracking` block sets defaults (e.g. backend_url) that the worker merges with CLI.

Use `--local` to run the worker directly (no bsub). If env.yaml is absent, bsub is still
attempted with only the populated fields (you may need a minimal file for -P on your site).

Usage:
  parallel_tracking_launcher.py [--local] [--env-yaml PATH] [--submit-count N] PROJECT_DIR [-- WORKER_ARGS...]

Examples:
  parallel_tracking_launcher.py /opt/projects/segmentation_tracking_projects/ab12-demo
  parallel_tracking_launcher.py /path/proj -- --use-all-anchors --clear-stuck
  SAM3_ENV_YAML=/other/env.yaml parallel_tracking_launcher.py --local /path/proj
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts" / "parallel_tracking_worker.py"


def _split_at_ddash(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def _str(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _resolve_bsub_path(p: str, repo: Path) -> str:
    """Absolute paths unchanged; relative paths resolved from repo root."""

    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((repo / path).resolve())


def _build_bsub_argv(lsf: dict[str, Any], repo: Path) -> list[str]:
    out: list[str] = ["bsub"]

    jn = _str(lsf.get("job_name")) or "sam3-partrack"
    spec = _str(lsf.get("job_array_spec"))
    if spec:
        inner = spec.strip("[]")
        out += ["-J", f"{jn}[{inner}]"]
    else:
        out += ["-J", jn]

    qp = _str(lsf.get("queue_project"))
    if qp:
        out += ["-P", qp]

    qu = _str(lsf.get("queue"))
    if qu:
        out += ["-q", qu]

    n = lsf.get("num_process_slots")
    if n is not None and str(n).strip():
        out += ["-n", str(int(n))]

    for r in lsf.get("resources") or []:
        rs = _str(r)
        if rs:
            out += ["-R", rs]

    gpu = _str(lsf.get("gpu_allocation"))
    if gpu:
        out += ["-gpu", gpu]

    rl = _str(lsf.get("run_limit"))
    if rl:
        out += ["-W", rl]

    log_ow = bool(lsf.get("log_overwrite"))
    o_flag = "-oo" if log_ow else "-o"
    e_flag = "-eo" if log_ow else "-e"

    so = _str(lsf.get("stdout"))
    if so:
        out += [o_flag, _resolve_bsub_path(so, repo)]

    se = _str(lsf.get("stderr"))
    if se:
        out += [e_flag, _resolve_bsub_path(se, repo)]

    cwd = _str(lsf.get("working_directory"))
    if cwd:
        out += ["-cwd", str(Path(cwd).expanduser())]

    extras = lsf.get("extra_bsub_args") or []
    if isinstance(extras, list):
        for x in extras:
            sx = _str(x)
            if sx:
                out.append(sx)
    elif isinstance(extras, str) and extras.strip():
        out.append(extras.strip())

    return out


def _listish(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str):
        # Allow either a single line or multi-line string
        return [ln.strip() for ln in v.splitlines() if ln.strip()]
    return [str(v)]


def _build_job_shell_command(
    *,
    lsf: dict[str, Any],
    repo: Path,
    py_exe: str,
    worker: Path,
    project_dir: Path,
    worker_args: list[str],
) -> list[str]:
    """
    Build a command of the form:
      bash -lc 'cd <repo> && <preamble...> && python3 scripts/parallel_tracking_worker.py <project_dir> ...'
    so the job is self-contained (no manual backend startup needed when auto-start is enabled).
    """

    shell = _str(lsf.get("job_shell")) or "bash"
    login = bool(lsf.get("shell_login", True))
    run_in_repo = bool(lsf.get("run_in_repo", True))
    preamble = _listish(lsf.get("shell_preamble"))

    # Always ensure repo-relative log directories exist in the job (common failure mode).
    mkdirs: list[str] = []
    for k in ("stdout", "stderr"):
        p = _str(lsf.get(k))
        if not p:
            continue
        if Path(p).is_absolute():
            d = str(Path(p).expanduser().resolve().parent)
        else:
            d = str((repo / Path(p)).resolve().parent)
        mkdirs.append(f"mkdir -p {shlex.quote(d)}")

    worker_cmd = [py_exe, str(worker), str(project_dir), *worker_args]
    worker_cmd_q = " ".join(shlex.quote(x) for x in worker_cmd)

    parts: list[str] = []
    if run_in_repo:
        parts.append(f"cd {shlex.quote(str(repo))}")
    parts.extend(mkdirs)
    parts.extend([x for x in preamble if x.strip()])
    parts.append(worker_cmd_q)
    joined = " && ".join(parts)

    return [shell, "-lc" if login else "-c", joined]


def main() -> int:
    argv_la, argv_worker = _split_at_ddash(sys.argv[1:])

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--local", action="store_true", help="Run worker locally (skip bsub).")
    ap.add_argument(
        "--env-yaml",
        type=str,
        default="",
        help="Override path to env YAML (otherwise SAM3_ENV_YAML or configs/env.yaml).",
    )
    ap.add_argument(
        "--submit-count",
        type=int,
        default=0,
        help="Submit this many identical bsub jobs (0 = value from env.yaml submit_count).",
    )
    ap.add_argument("project_dir", type=Path, help="Project directory containing config.json.")
    args = ap.parse_args(argv_la)

    if args.env_yaml.strip():
        os.environ["SAM3_ENV_YAML"] = args.env_yaml.strip()

    from repo_env_yaml import (  # noqa: E402
        load_env_yaml_raw,
        lsf_parallel_tracking_section,
        parallel_tracking_section,
        resolve_env_yaml_path,
    )
    cfg_all = load_env_yaml_raw()

    yaml_path = resolve_env_yaml_path()
    pt = parallel_tracking_section(cfg_all)
    lsf = lsf_parallel_tracking_section(cfg_all)

    project_dir = args.project_dir.expanduser().resolve()
    if not (project_dir / "config.json").is_file():
        print(f"No config.json under {project_dir}", file=sys.stderr)
        return 2

    be = pt.get("backend_url")
    if be is not None and str(be).strip():
        os.environ.setdefault("SAM3_BACKEND_URL", str(be).strip())

    # Job environment block from YAML
    env_block = lsf.get("environment") if isinstance(lsf.get("environment"), dict) else {}
    child_env = os.environ.copy()
    for k, v in env_block.items():
        ks = _str(k)
        if ks and v is not None:
            child_env.setdefault(ks, str(v))

    py_exe = _str(lsf.get("worker_python"))
    if not py_exe:
        py_exe = shutil.which("python3") or "python3"
    worker_cmd_base = [py_exe, str(WORKER), str(project_dir), *argv_worker]

    submit_n = args.submit_count if args.submit_count > 0 else int(lsf.get("submit_count") or 1)
    submit_n = max(1, submit_n)

    if args.local:
        if yaml_path:
            print(f"Using env YAML (defaults already merged via worker CLI): {yaml_path}", flush=True)
        rc = subprocess.call(worker_cmd_base, env=child_env, cwd=str(REPO_ROOT))
        return int(rc)

    bsub_argv = _build_bsub_argv(lsf, repo=REPO_ROOT)
    # Submit a self-contained shell command so conda/module setup can happen
    # and so workers can auto-start local backends per GPU if configured.
    bsub_argv.extend(
        _build_job_shell_command(
            lsf=lsf,
            repo=REPO_ROOT,
            py_exe=py_exe,
            worker=WORKER,
            project_dir=project_dir,
            worker_args=argv_worker,
        )
    )

    if yaml_path:
        print(f"env YAML: {yaml_path}", flush=True)
    print("bsub:", " ".join(bsub_argv[:12]) + (" …" if len(bsub_argv) > 12 else ""), flush=True)

    codes: list[int] = []
    for i in range(submit_n):
        if submit_n > 1:
            print(f"--- submit {i + 1}/{submit_n} ---", flush=True)
        codes.append(subprocess.call(bsub_argv, env=child_env, cwd=str(REPO_ROOT)))

    return 1 if any(c != 0 for c in codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
