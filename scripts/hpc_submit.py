#!/usr/bin/env python3
"""
Submit SAM3 WebTrack batch jobs to Slurm or LSF.

Job types:
  tracking        Propagate every eligible video in a project (GPU). --count N
                  submits an N-element job array; elements share the work via
                  the project's claims file.
  features        One CPU job per tracked video: frame-feature extraction for
                  downstream clustering (cached under analysis_of_tracking_data/).
  behavior-quant  Clustering / behavior quantification for a project (CPU).

Cluster settings come from the `hpc:` block of configs/env.yaml (or
SAM3_ENV_YAML); see configs/env_template.yaml. The scheduler is taken from
`hpc.scheduler`, else auto-detected from `sbatch` / `bsub` on PATH.

Usage:
  hpc_submit.py tracking PROJECT_DIR [--count N] [--dry-run] [-- WORKER_ARGS...]
  hpc_submit.py features PROJECT_DIR [--skip-cached] [--dry-run]
  hpc_submit.py behavior-quant PROJECT_DIR [--dry-run] [-- CLUSTERING_ARGS...]

Common options: --scheduler {slurm,lsf}, --env-yaml PATH.
--dry-run prints the generated job script(s) without writing or submitting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULERS = ("slurm", "lsf")

# Per-job-type defaults; `hpc:` values override these, `hpc.jobs.<type>:` overrides both.
JOB_DEFAULTS: dict[str, dict[str, Any]] = {
    "tracking": {"cpus": 8, "mem_mb": 32000, "gpus": 1, "walltime": "24:00"},
    "features": {"cpus": 8, "mem_mb": 16000, "gpus": 0, "walltime": "08:00"},
    "behavior_quant": {"cpus": 32, "mem_mb": 64000, "gpus": 0, "walltime": "12:00"},
}


@dataclass
class JobSpec:
    name: str
    body: list[str]
    cpus: int
    mem_mb: int
    gpus: int
    walltime_min: int
    account: str = ""
    queue: str = ""
    array_size: int = 0
    extra_directives: list[str] = field(default_factory=list)
    preamble: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)


# ─── Config ──────────────────────────────────────────────────────────────────

def _listish(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [ln.strip() for ln in v.splitlines() if ln.strip()]
    return [str(x).strip() for x in v if str(x).strip()]


def parse_walltime(value: Any) -> int:
    """'HH:MM', 'HH:MM:SS' or 'D-HH:MM[:SS]' → whole minutes (rounded up)."""
    s = str(value).strip()
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{1,2})(?::(\d{1,2}))?", s)
    if not m:
        raise ValueError(f"Unrecognised walltime {value!r}; use HH:MM or HH:MM:SS")
    days, hours, mins, secs = (int(g) if g else 0 for g in m.groups())
    return days * 1440 + hours * 60 + mins + (1 if secs else 0)


def load_hpc_config(env_yaml: str) -> dict[str, Any]:
    if env_yaml:
        os.environ["SAM3_ENV_YAML"] = env_yaml
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from repo_env_yaml import load_env_yaml_raw

    hpc = load_env_yaml_raw().get("hpc")
    return dict(hpc) if isinstance(hpc, dict) else {}


def resolve_scheduler(cli: str, hpc: dict[str, Any]) -> str:
    choice = (cli or str(hpc.get("scheduler") or "")).strip().lower()
    if choice in ("", "auto"):
        if shutil.which("sbatch"):
            return "slurm"
        if shutil.which("bsub"):
            return "lsf"
        raise SystemExit(
            "No scheduler found (neither sbatch nor bsub on PATH). "
            "Set hpc.scheduler in configs/env.yaml or pass --scheduler."
        )
    if choice not in SCHEDULERS:
        raise SystemExit(f"hpc.scheduler must be one of {SCHEDULERS}, got {choice!r}")
    return choice


def default_preamble() -> list[str]:
    """Re-activate the submitting conda env on the compute node."""
    prefix = os.environ.get("CONDA_PREFIX", "").strip()
    conda = shutil.which("conda")
    if not prefix or not conda:
        return []
    base = subprocess.run([conda, "info", "--base"], capture_output=True, text=True).stdout.strip()
    if not base:
        return []
    return [
        f"source {shlex.quote(base)}/etc/profile.d/conda.sh",
        f"conda activate {shlex.quote(prefix)}",
    ]


def job_settings(hpc: dict[str, Any], job_type: str) -> dict[str, Any]:
    merged = dict(JOB_DEFAULTS[job_type])
    for k in ("cpus", "mem_mb", "gpus", "walltime", "account", "queue"):
        if hpc.get(k) not in (None, ""):
            merged[k] = hpc[k]
    per_job = (hpc.get("jobs") or {}).get(job_type) or {}
    merged.update({k: v for k, v in per_job.items() if v not in (None, "")})
    merged["extra_directives"] = _listish(hpc.get("extra_directives")) + _listish(
        per_job.get("extra_directives")
    )
    return merged


def make_spec(hpc: dict[str, Any], job_type: str, name: str, body: list[str], array_size: int = 0) -> JobSpec:
    s = job_settings(hpc, job_type)
    preamble = _listish(hpc.get("shell_preamble")) or default_preamble()
    env = {str(k): str(v) for k, v in (hpc.get("environment") or {}).items() if v is not None}
    python = str(hpc.get("python") or "").strip()
    if python:
        env.setdefault("SAM3_JOB_PYTHON", python)
    env["SAM3_JOB_GPUS"] = str(int(s["gpus"]))
    return JobSpec(
        name=name,
        body=body,
        cpus=int(s["cpus"]),
        mem_mb=int(s["mem_mb"]),
        gpus=int(s["gpus"]),
        walltime_min=parse_walltime(s["walltime"]),
        account=str(s.get("account") or ""),
        queue=str(s.get("queue") or ""),
        array_size=array_size,
        extra_directives=s["extra_directives"],
        preamble=preamble,
        environment=env,
    )


# ─── Rendering ───────────────────────────────────────────────────────────────

def _slurm_directives(spec: JobSpec, log_dir: Path) -> list[str]:
    stem = "%x_%A_%a" if spec.array_size > 1 else "%x_%j"
    d = [
        f"--job-name={spec.name}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={spec.cpus}",
        f"--mem={spec.mem_mb}M",
        f"--time={spec.walltime_min // 60}:{spec.walltime_min % 60:02d}:00",
        f"--output={log_dir / stem}.out",
        f"--error={log_dir / stem}.err",
    ]
    if spec.account:
        d.append(f"--account={spec.account}")
    if spec.queue:
        d.append(f"--partition={spec.queue}")
    if spec.gpus > 0:
        d.append(f"--gres=gpu:{spec.gpus}")
    if spec.array_size > 1:
        d.append(f"--array=1-{spec.array_size}")
    return ["#SBATCH " + x for x in d + spec.extra_directives]


def _lsf_directives(spec: JobSpec, log_dir: Path) -> list[str]:
    name = f"{spec.name}[1-{spec.array_size}]" if spec.array_size > 1 else spec.name
    stem = f"{spec.name}_%J_%I" if spec.array_size > 1 else f"{spec.name}_%J"
    d = [
        f'-J "{name}"',
        f"-n {spec.cpus}",
        # LSF memory units are site-configured (LSF_UNIT_FOR_LIMITS); MB is the common default.
        f'-R "rusage[mem={spec.mem_mb}]"',
        '-R "span[hosts=1]"',
        f"-W {spec.walltime_min // 60}:{spec.walltime_min % 60:02d}",
        f"-o {log_dir / stem}.out",
        f"-e {log_dir / stem}.err",
    ]
    if spec.account:
        d.append(f"-P {spec.account}")
    if spec.queue:
        d.append(f"-q {spec.queue}")
    if spec.gpus > 0:
        d.append(f'-gpu "num={spec.gpus}"')
    return ["#BSUB " + x for x in d + spec.extra_directives]


def render(spec: JobSpec, scheduler: str, log_dir: Path) -> str:
    directives = (_slurm_directives if scheduler == "slurm" else _lsf_directives)(spec, log_dir)
    lines = ["#!/bin/bash", *directives, "", "set -eo pipefail", ""]
    if spec.preamble:
        lines += ["# Environment setup", *spec.preamble, ""]
    for k, v in sorted(spec.environment.items()):
        lines.append(f"export {k}={shlex.quote(v)}")
    lines += ["", f"cd {shlex.quote(str(REPO_ROOT))}", *spec.body, ""]
    return "\n".join(lines)


def submit(script: Path, scheduler: str) -> int:
    if scheduler == "slurm":
        return subprocess.call(["sbatch", str(script)])
    with open(script) as f:
        return subprocess.call(["bsub"], stdin=f)


# ─── Job types ───────────────────────────────────────────────────────────────

def _project_config(project_dir: Path) -> dict[str, Any]:
    cfg = project_dir / "config.json"
    if not cfg.is_file():
        raise SystemExit(f"No config.json under {project_dir}")
    return json.loads(cfg.read_text(encoding="utf-8"))


def _find_video_dir(project_dir: Path, vid: str) -> Path | None:
    root = project_dir / "videos"
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if d.name == vid or d.name.startswith(vid + "_"):
            return d
    return None


def tracking_jobs(hpc, project_dir, args, extra) -> list[JobSpec]:
    _project_config(project_dir)
    body = [" ".join(shlex.quote(x) for x in
                     ["bash", "scripts/hpc/tracking_job.sh", str(project_dir), *extra])]
    return [make_spec(hpc, "tracking", args.job_name or "sam3-track", body, array_size=max(1, args.count))]


def behavior_quant_jobs(hpc, project_dir, args, extra) -> list[JobSpec]:
    _project_config(project_dir)
    body = [" ".join(shlex.quote(x) for x in
                     ["bash", "scripts/hpc/behavior_quant_job.sh", str(project_dir), *extra])]
    return [make_spec(hpc, "behavior_quant", args.job_name or "sam3-behavior", body)]


def feature_jobs(hpc, project_dir, args, extra) -> list[JobSpec]:
    videos = _project_config(project_dir).get("videos") or {}
    cache_dir = project_dir / "analysis_of_tracking_data" / "clustering" / "features"
    specs, n_ineligible, n_cached = [], 0, 0
    for vid, v in videos.items():
        vdir = _find_video_dir(project_dir, vid)
        if not v.get("propagation_complete") or vdir is None or not (vdir / "masks.sqlite").is_file():
            n_ineligible += 1
            continue
        if args.skip_cached and (cache_dir / f"frame_features_{vid}.npz").is_file():
            n_cached += 1
            continue
        body = ['"${SAM3_JOB_PYTHON:-python}" ' + " ".join(shlex.quote(x) for x in [
            "downstream_analysis/clustering/extract_single_video.py",
            "--project-dir", str(project_dir), "--video-id", vid, *extra,
        ])]
        specs.append(make_spec(hpc, "features", f"feat_{vid}", body))
    print(f"Videos: {len(videos)}  eligible jobs: {len(specs)}  "
          f"not tracked yet: {n_ineligible}  already cached: {n_cached}", file=sys.stderr)
    return specs


JOB_TYPES = {"tracking": tracking_jobs, "features": feature_jobs, "behavior-quant": behavior_quant_jobs}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_type", choices=sorted(JOB_TYPES))
    ap.add_argument("project_dir", type=Path)
    ap.add_argument("--scheduler", choices=SCHEDULERS, default="")
    ap.add_argument("--env-yaml", default="", help="Config file (default: SAM3_ENV_YAML or configs/env.yaml)")
    ap.add_argument("--count", type=int, default=1, help="tracking: number of parallel array jobs")
    ap.add_argument("--skip-cached", action="store_true", help="features: skip videos already extracted")
    ap.add_argument("--job-name", default="", help="Override the job name")
    ap.add_argument("--dry-run", action="store_true", help="Print job script(s); do not write or submit")
    args = ap.parse_args(argv)

    hpc = load_hpc_config(args.env_yaml)
    scheduler = resolve_scheduler(args.scheduler, hpc)
    project_dir = args.project_dir.expanduser().resolve()

    log_dir = Path(str(hpc.get("log_dir") or "logs/hpc")).expanduser()
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    script_dir = log_dir / "jobs"

    specs = JOB_TYPES[args.job_type](hpc, project_dir, args, extra)
    if not specs:
        print("Nothing to submit.", file=sys.stderr)
        return 0

    if args.dry_run:
        for spec in specs[:3]:
            print(f"# ---- {spec.name}.{scheduler} ----")
            print(render(spec, scheduler, log_dir))
        if len(specs) > 3:
            print(f"# ... and {len(specs) - 3} more job(s)")
        return 0

    script_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for spec in specs:
        path = script_dir / f"{spec.name}.{scheduler}.sh"
        path.write_text(render(spec, scheduler, log_dir), encoding="utf-8")
        rc = submit(path, scheduler)
        failures += rc != 0
        if rc != 0:
            print(f"FAILED to submit {path}", file=sys.stderr)
    print(f"Submitted {len(specs) - failures}/{len(specs)} job(s) via {scheduler}. "
          f"Scripts: {script_dir}  Logs: {log_dir}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
