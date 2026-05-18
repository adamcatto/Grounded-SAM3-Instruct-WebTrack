#!/usr/bin/env python3
"""Generate LSF (.lsf) job scripts for parallel frame feature extraction.

Creates one bsub job per video, plus convenience submit/check scripts.

Usage:
    python generate_bsub_jobs.py --project-dir /path/to/project
    python generate_bsub_jobs.py --project-dir /path/to/project --submit-all
    python generate_bsub_jobs.py --project-dir /path/to/project --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BSUB_ACCOUNT = "acc_KennyComputational"
BSUB_QUEUE = "premium"
BSUB_CORES = 32
BSUB_MEM_MB = 16000
BSUB_WALLTIME = "08:00"  # 8 hours (most videos ~55 min, some buffer)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_SCRIPT = REPO_ROOT / "downstream_analysis" / "clustering" / "extract_single_video.py"

LSF_TEMPLATE = dedent("""\
    #!/bin/bash
    #BSUB -J feat_{video_id}
    #BSUB -P {account}
    #BSUB -q {queue}
    #BSUB -n {cores}
    #BSUB -R "rusage[mem={mem_mb}]"
    #BSUB -W {walltime}
    #BSUB -oo {log_dir}/feat_{video_id}.out
    #BSUB -eo {log_dir}/feat_{video_id}.err

    set -euo pipefail

    ml purge

    set +u
    source /sc/arion/work/cattoa01/miniconda3/etc/profile.d/conda.sh
    conda activate /sc/arion/work/cattoa01/miniconda3/envs/sam3
    set -u

    echo "========================================="
    echo "Job: feat_{video_id}"
    echo "Video: {video_name}"
    echo "Host: $(hostname)"
    echo "Date: $(date)"
    echo "========================================="

    python {extract_script} \\
        --project-dir {project_dir} \\
        --video-id {video_id}

    echo "========================================="
    echo "Job complete: $(date)"
    echo "========================================="
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", type=Path, required=True,
                        help="Project folder with config.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing")
    parser.add_argument("--submit-all", action="store_true",
                        help="Generate and immediately bsub all jobs")
    parser.add_argument("--account", default=BSUB_ACCOUNT,
                        help=f"LSF account (default: {BSUB_ACCOUNT})")
    parser.add_argument("--queue", default=BSUB_QUEUE,
                        help=f"LSF queue (default: {BSUB_QUEUE})")
    parser.add_argument("--walltime", default=BSUB_WALLTIME,
                        help=f"Wall time limit (default: {BSUB_WALLTIME})")
    parser.add_argument("--mem", type=int, default=BSUB_MEM_MB,
                        help=f"Memory per job in MB (default: {BSUB_MEM_MB})")
    parser.add_argument("--skip-cached", action="store_true",
                        help="Skip videos that already have cached frame features")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    config_path = project_dir / "config.json"
    if not config_path.is_file():
        print(f"ERROR: No config.json in {project_dir}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    videos = config.get("videos", {})

    # Filter to propagation_complete videos with masks.sqlite
    eligible = []
    skipped_ineligible = []
    for vid, v in videos.items():
        vname = str(v.get("name") or vid)
        if not v.get("propagation_complete"):
            skipped_ineligible.append((vid, vname, "propagation_complete=false"))
            continue
        # Find video dir
        vdir = None
        root = project_dir / "videos"
        if root.is_dir():
            for d in root.iterdir():
                if d.name == vid or d.name.startswith(vid + "_"):
                    vdir = d
                    break
        if vdir is None or not (vdir / "masks.sqlite").is_file():
            skipped_ineligible.append((vid, vname, "no masks.sqlite"))
            continue
        eligible.append((vid, vname, v))

    print(f"Project: {project_dir.name}")
    print(f"Total videos: {len(videos)}")
    print(f"Eligible: {len(eligible)}")
    if skipped_ineligible:
        print(f"Skipped (ineligible): {len(skipped_ineligible)}")

    # Optionally skip already-cached
    cache_dir = project_dir / "analysis_of_tracking_data" / "clustering" / "features"
    skipped_cached = []
    to_process = []
    for vid, vname, v in eligible:
        cache_path = cache_dir / f"frame_features_{vid}.npz"
        if args.skip_cached and cache_path.is_file():
            skipped_cached.append((vid, vname))
        else:
            to_process.append((vid, vname, v))

    if skipped_cached:
        print(f"Skipped (already cached): {len(skipped_cached)}")
    print(f"Jobs to submit: {len(to_process)}")
    print()

    if not to_process:
        print("Nothing to do! All videos already cached.")
        return

    # Setup directories
    output_base = project_dir / "analysis_of_tracking_data" / "clustering"
    jobs_dir = output_base / "bsub_jobs"
    log_dir = output_base / "bsub_logs"

    if args.dry_run:
        print(f"Would create {len(to_process)} .lsf files in {jobs_dir}/")
        print(f"Logs would go to {log_dir}/")
        print()
        # Show one example
        vid, vname, v = to_process[0]
        content = LSF_TEMPLATE.format(
            video_id=vid,
            video_name=vname,
            account=args.account,
            queue=args.queue,
            cores=BSUB_CORES,
            mem_mb=args.mem,
            walltime=args.walltime,
            log_dir=log_dir,
            extract_script=EXTRACT_SCRIPT,
            project_dir=project_dir,
        )
        print(f"--- Example: feat_{vid}.lsf ---")
        print(content)
        return

    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate .lsf files
    lsf_paths = []
    for vid, vname, v in to_process:
        content = LSF_TEMPLATE.format(
            video_id=vid,
            video_name=vname,
            account=args.account,
            queue=args.queue,
            cores=BSUB_CORES,
            mem_mb=args.mem,
            walltime=args.walltime,
            log_dir=log_dir,
            extract_script=EXTRACT_SCRIPT,
            project_dir=project_dir,
        )
        lsf_path = jobs_dir / f"feat_{vid}.lsf"
        lsf_path.write_text(content)
        lsf_paths.append(lsf_path)

    print(f"Generated {len(lsf_paths)} .lsf files in {jobs_dir}/")
    print(f"Logs will go to {log_dir}/")

    # Generate submit_all.sh
    submit_path = jobs_dir / "submit_all.sh"
    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f'JOBS_DIR="$(cd "$(dirname "$0")" && pwd)"',
        "",
        f'echo "Submitting {len(lsf_paths)} feature extraction jobs..."',
        "",
        f'for lsf in "$JOBS_DIR"/feat_*.lsf; do',
        '  name=$(basename "$lsf" .lsf)',
        '  echo "  bsub < $lsf"',
        '  bsub < "$lsf"',
        "done",
        "",
        f'echo ""',
        f'echo "Done. {len(lsf_paths)} jobs submitted."',
        f'echo "Monitor with: bjobs -w | grep feat_"',
        f'echo "Check completion: python {Path(__file__).resolve()} --project-dir {project_dir} --skip-cached --dry-run"',
        "",
    ]
    submit_path.write_text("\n".join(submit_lines))
    st = os.stat(submit_path)
    os.chmod(submit_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"Submit script: {submit_path}")

    # Generate check_status.sh
    check_path = jobs_dir / "check_status.sh"
    check_lines = [
        "#!/bin/bash",
        f'CACHE_DIR="{cache_dir}"',
        f'TOTAL={len(to_process)}',
        "",
        'DONE=$(find "$CACHE_DIR" -name "frame_features_*.npz" 2>/dev/null | wc -l)',
        'echo "Cached: $DONE / $TOTAL videos"',
        "",
        "# Check running jobs",
        'RUNNING=$(bjobs -w 2>/dev/null | grep "feat_" | wc -l)',
        'echo "Running bsub jobs: $RUNNING"',
        "",
        "# Check for errors",
        f'ERR_DIR="{log_dir}"',
        'ERRORS=0',
        'for err in "$ERR_DIR"/feat_*.err; do',
        '  [ -f "$err" ] || continue',
        '  if [ -s "$err" ]; then',
        '    ERRORS=$((ERRORS + 1))',
        '  fi',
        'done',
        'echo "Jobs with stderr output: $ERRORS"',
        "",
        'if [ "$DONE" -eq "$TOTAL" ]; then',
        '  echo ""',
        '  echo "ALL VIDEOS COMPLETE! Ready to run clustering:"',
        f'  echo "  conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering --project-dir {project_dir} --skip-mask-verification"',
        "fi",
        "",
    ]
    check_path.write_text("\n".join(check_lines))
    os.chmod(check_path, os.stat(check_path).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"Status checker: {check_path}")

    # Submit if requested
    if args.submit_all:
        print(f"\nSubmitting {len(lsf_paths)} jobs...")
        n_ok = 0
        for lsf_path in lsf_paths:
            with open(lsf_path) as f:
                result = subprocess.run(["bsub"], stdin=f, capture_output=True, text=True)
            if result.returncode == 0:
                n_ok += 1
                print(f"  {lsf_path.stem}: {result.stdout.strip()}")
            else:
                print(f"  FAILED {lsf_path.stem}: {result.stderr.strip()}", file=sys.stderr)
        print(f"\nSubmitted {n_ok}/{len(lsf_paths)} jobs.")
        print(f"Monitor: bjobs -w | grep feat_")
        print(f"Check:   bash {check_path}")
    else:
        print(f"\nTo submit all jobs:")
        print(f"  bash {submit_path}")
        print(f"  # or: python {Path(__file__).resolve()} --project-dir {project_dir} --submit-all")


if __name__ == "__main__":
    main()
