"""Command-line entry point for the VOS comparison experiment.

Run from the repo root:

  python -m experiments.vos_comparison build    --source-project <ref> [--batch-size inherit|single|N]
  python -m experiments.vos_comparison track    --project <single-shot ref> [--backend URL]
  python -m experiments.vos_comparison evaluate  --project <single-shot ref> [--threshold 0.5]
  python -m experiments.vos_comparison figures   --project <single-shot ref>
  python -m experiments.vos_comparison all       --source-project <ref> [--backend URL]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import DEFAULT_IOU_THRESHOLD
from .common import resolve_project_dir
from .evaluation import evaluate_project
from .figures import make_figures
from .project_builder import build_single_shot_project
from .tracking import run_tracking


def _add_threshold_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
                   help="IoU/Jaccard collapse threshold τ (default %(default)s).")
    p.add_argument("--window", type=int, default=None,
                   help="Smoothing window (frames) for collapse detection "
                        "(default ~1%% of video, clamped to [15,300]).")
    p.add_argument("--persist-frac", type=float, default=0.8,
                   help="Fraction of the remaining video that must stay below τ "
                        "to call a collapse sustained (default %(default)s).")
    p.add_argument("--out-dir", default=None,
                   help="Results directory (default <project>/vos_comparison_results).")


def _out_dir(args, project_dir: Path) -> Path:
    return Path(args.out_dir).resolve() if args.out_dir else (project_dir / "vos_comparison_results")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.vos_comparison",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Emit single_shot_vos_<project> (symlinks, first anchor only).")
    pb.add_argument("--source-project", required=True,
                    help="Source anchor-based project: path, folder name, or short id.")
    pb.add_argument("--batch-size", default="inherit",
                    help="'inherit' (default), 'single' (whole video in one batch), or an int.")
    pb.add_argument("--videos", nargs="*", default=None, help="Restrict to source video ids.")
    pb.add_argument("--output-parent", default=None,
                    help="Where to create the project (default: alongside the source project).")
    pb.add_argument("--overwrite", action="store_true", help="Replace an existing output project.")

    pt = sub.add_parser("track", help="Run SAM3 propagation on the single-shot project.")
    pt.add_argument("--project", required=True, help="Single-shot project: path, name, or id.")
    pt.add_argument("--backend", default="http://127.0.0.1:8000", help="Backend origin (default %(default)s).")
    pt.add_argument("--videos", nargs="*", default=None, help="Restrict to single-shot video ids.")
    pt.add_argument("--sse-timeout", type=float, default=None, help="urlopen timeout (s).")
    pt.add_argument("--quiet-stream", action="store_true", help="Suppress per-frame SSE progress.")
    pt.add_argument("--force", action="store_true", help="Re-propagate even if already complete.")

    pe = sub.add_parser("evaluate", help="Score single-shot tracks vs. anchor ground truth.")
    pe.add_argument("--project", required=True, help="Single-shot project: path, name, or id.")
    _add_threshold_args(pe)

    pf = sub.add_parser("figures", help="Render figures + LaTeX tables from results.")
    pf.add_argument("--project", required=True, help="Single-shot project: path, name, or id.")
    pf.add_argument("--out-dir", default=None, help="Results directory (default <project>/vos_comparison_results).")
    pf.add_argument("--threshold", type=float, default=None, help="Override τ used for figure annotations.")

    pa = sub.add_parser("all", help="build → track → evaluate → figures end-to-end.")
    pa.add_argument("--source-project", required=True, help="Source anchor-based project.")
    pa.add_argument("--batch-size", default="inherit", help="'inherit', 'single', or an int.")
    pa.add_argument("--videos", nargs="*", default=None, help="Restrict to source video ids.")
    pa.add_argument("--output-parent", default=None)
    pa.add_argument("--overwrite", action="store_true")
    pa.add_argument("--backend", default="http://127.0.0.1:8000", help="Backend origin (default %(default)s).")
    pa.add_argument("--quiet-stream", action="store_true")
    _add_threshold_args(pa)
    return ap


def _parse_batch_size(raw: str):
    if raw in ("inherit", "single"):
        return raw
    return int(raw)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "build":
        build_single_shot_project(
            args.source_project,
            output_parent=Path(args.output_parent) if args.output_parent else None,
            batch_size=_parse_batch_size(args.batch_size),
            only_videos=args.videos,
            overwrite=args.overwrite,
        )
        return 0

    if args.cmd == "track":
        return run_tracking(
            resolve_project_dir(args.project),
            backend=args.backend,
            only_videos=args.videos,
            sse_timeout=args.sse_timeout,
            quiet_stream=args.quiet_stream,
            force=args.force,
        )

    if args.cmd == "evaluate":
        project_dir = resolve_project_dir(args.project)
        evaluate_project(
            project_dir,
            out_dir=_out_dir(args, project_dir),
            threshold=args.threshold,
            window=args.window,
            persist_frac=args.persist_frac,
        )
        return 0

    if args.cmd == "figures":
        project_dir = resolve_project_dir(args.project)
        make_figures(_out_dir(args, project_dir), threshold=args.threshold)
        return 0

    if args.cmd == "all":
        project_dir = build_single_shot_project(
            args.source_project,
            output_parent=Path(args.output_parent) if args.output_parent else None,
            batch_size=_parse_batch_size(args.batch_size),
            only_videos=args.videos,
            overwrite=args.overwrite,
        )
        rc = run_tracking(
            project_dir, backend=args.backend, quiet_stream=args.quiet_stream
        )
        if rc != 0:
            print(f"[all] tracking exited with code {rc}; evaluating what completed.")
        out_dir = _out_dir(args, project_dir)
        evaluate_project(
            project_dir,
            out_dir=out_dir,
            threshold=args.threshold,
            window=args.window,
            persist_frac=args.persist_frac,
        )
        make_figures(out_dir, threshold=args.threshold)
        return rc

    return 2
