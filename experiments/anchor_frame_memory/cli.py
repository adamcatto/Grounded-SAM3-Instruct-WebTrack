"""CLI for the anchor-frame memory ablation.

Run from the repo root:

  python -m experiments.anchor_frame_memory backup --source-project <project>
  python -m experiments.anchor_frame_memory build --source-project <project>
  python -m experiments.anchor_frame_memory track --project <queue-project>
  python -m experiments.anchor_frame_memory evaluate --project <queue-project>
  python -m experiments.anchor_frame_memory figures --project <queue-project>
  python -m experiments.anchor_frame_memory all --source-project <project>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import DEFAULT_IOU_THRESHOLD, RESULTS_DIRNAME


def _resolve(ref: str) -> Path:
    from .common import resolve_project_dir

    return resolve_project_dir(ref)


def _out_dir(args, project_dir: Path) -> Path:
    return Path(args.out_dir).resolve() if args.out_dir else project_dir / RESULTS_DIRNAME


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.anchor_frame_memory",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pbk = sub.add_parser("backup", help="Create a timestamped copy of the source project.")
    pbk.add_argument("--source-project", required=True)
    pbk.add_argument("--backup-parent", default=None)

    pb = sub.add_parser("build", help="Create recent_queue_memory_<project> from labels only.")
    pb.add_argument("--source-project", required=True)
    pb.add_argument("--videos", nargs="*", default=None)
    pb.add_argument("--output-parent", default=None)
    pb.add_argument("--overwrite", action="store_true")
    pb.add_argument("--emit-path-file", default=None)

    pt = sub.add_parser("track", help="Track the fork with standard recent-queue memory.")
    pt.add_argument("--project", required=True)
    pt.add_argument("--backend", default="http://127.0.0.1:8000")
    pt.add_argument("--videos", nargs="*", default=None)
    pt.add_argument("--sse-timeout", type=float, default=None)
    pt.add_argument("--quiet-stream", action="store_true")
    pt.add_argument("--force", action="store_true")

    pe = sub.add_parser("evaluate", help="Compare queue tracking to anchor-memory tracking.")
    pe.add_argument("--project", required=True)
    pe.add_argument("--out-dir", default=None)
    pe.add_argument("--threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    pe.add_argument("--min-absence-frames", type=int, default=30)
    pe.add_argument("--reentry-window", type=int, default=90)

    pf = sub.add_parser("figures", help="Render PNG/PDF/SVG figures.")
    pf.add_argument("--project", required=True)
    pf.add_argument("--out-dir", default=None)

    pa = sub.add_parser("all", help="backup -> build -> track -> evaluate -> figures.")
    pa.add_argument("--source-project", required=True)
    pa.add_argument("--videos", nargs="*", default=None)
    pa.add_argument("--output-parent", default=None)
    pa.add_argument("--backup-parent", default=None)
    pa.add_argument("--overwrite", action="store_true")
    pa.add_argument("--backend", default="http://127.0.0.1:8000")
    pa.add_argument("--quiet-stream", action="store_true")
    pa.add_argument("--threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    pa.add_argument("--min-absence-frames", type=int, default=30)
    pa.add_argument("--reentry-window", type=int, default=90)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "backup":
        from .common import backup_project

        dst = backup_project(
            _resolve(args.source_project),
            backup_parent=Path(args.backup_parent) if args.backup_parent else None,
        )
        print(f"[backup] {dst}")
        return 0

    if args.cmd == "build":
        from .project_builder import build_recent_queue_project

        project_dir = build_recent_queue_project(
            args.source_project,
            output_parent=Path(args.output_parent) if args.output_parent else None,
            only_videos=args.videos,
            overwrite=args.overwrite,
        )
        if args.emit_path_file:
            Path(args.emit_path_file).write_text(str(project_dir) + "\n")
        return 0

    if args.cmd == "track":
        from .tracking import run_queue_tracking

        return run_queue_tracking(
            _resolve(args.project),
            backend=args.backend,
            only_videos=args.videos,
            sse_timeout=args.sse_timeout,
            quiet_stream=args.quiet_stream,
            force=args.force,
        )

    if args.cmd == "evaluate":
        from .evaluation import evaluate_project

        project_dir = _resolve(args.project)
        evaluate_project(
            project_dir,
            out_dir=_out_dir(args, project_dir),
            threshold=args.threshold,
            min_absence_frames=args.min_absence_frames,
            reentry_window=args.reentry_window,
        )
        return 0

    if args.cmd == "figures":
        from .figures import make_figures

        project_dir = _resolve(args.project)
        make_figures(_out_dir(args, project_dir))
        return 0

    if args.cmd == "all":
        from .common import backup_project
        from .evaluation import evaluate_project
        from .figures import make_figures
        from .project_builder import build_recent_queue_project
        from .tracking import run_queue_tracking

        src = _resolve(args.source_project)
        backup = backup_project(
            src, backup_parent=Path(args.backup_parent) if args.backup_parent else None
        )
        print(f"[all] backup: {backup}")
        project_dir = build_recent_queue_project(
            str(src),
            output_parent=Path(args.output_parent) if args.output_parent else None,
            only_videos=args.videos,
            overwrite=args.overwrite,
        )
        rc = run_queue_tracking(
            project_dir,
            backend=args.backend,
            quiet_stream=args.quiet_stream,
            force=True,
        )
        if rc != 0:
            print(f"[all] tracking exited with {rc}; evaluating completed frames")
        out_dir = project_dir / RESULTS_DIRNAME
        evaluate_project(
            project_dir,
            out_dir=out_dir,
            threshold=args.threshold,
            min_absence_frames=args.min_absence_frames,
            reentry_window=args.reentry_window,
        )
        make_figures(out_dir)
        return rc

    return 2
