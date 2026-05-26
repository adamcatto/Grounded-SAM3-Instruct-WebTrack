"""CLI entry point for the multi-project clustering pipeline.

Usage::

    conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering.multi_project_cli \
        --project-dirs \
            /path/to/Hab \
            /path/to/test-day \
            /path/to/SH-Intruder \
        --output-dir /path/to/multi_project_analysis \
        --registry-csv downstream_analysis/clustering/experiment_registry.csv \
        --skip-mask-verification \
        --batch-correction combat

    # With locomotion comparison:
    conda run --no-capture-output -n sam3 python -m downstream_analysis.clustering.multi_project_cli \
        --project-dirs ... \
        --locomotion-dir /path/to/Locomotion \
        --output-dir /path/to/multi_project_analysis
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Multi-project behavioral clustering pipeline.",
    )
    p.add_argument(
        "--project-dirs", nargs="+", type=Path, required=True,
        help="Interaction project directories (with config.json)",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for merged results",
    )
    p.add_argument(
        "--registry-csv", type=Path, default=None,
        help="Path to experiment_registry.csv (default: bundled)",
    )
    p.add_argument(
        "--batch-correction", choices=["combat", "zscore_per_batch", "none"],
        default="combat", help="Batch correction method (default: combat)",
    )
    p.add_argument(
        "--locomotion-dir", type=Path, default=None,
        help="Locomotion project directory (for alone vs paired comparison)",
    )
    p.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution")
    p.add_argument("--n-neighbors", type=int, default=30, help="k-NN neighbors")
    p.add_argument("--window-size", type=int, default=90, help="Window size in frames")
    p.add_argument("--stride", type=int, default=30, help="Stride in frames")
    p.add_argument(
        "--skip-mask-verification", action="store_true",
        help="Skip per-frame mask existence check",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.verbose and args.quiet:
        print("Cannot use both --verbose and --quiet", file=sys.stderr)
        return 2

    # Logging setup (same pattern as single-project CLI)
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    handler.flush = lambda: sys.stdout.flush()
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    sys.stdout.reconfigure(line_buffering=True)

    # Validate project dirs
    for d in args.project_dirs:
        if not (d / "config.json").is_file():
            print(f"No config.json in {d}", file=sys.stderr)
            return 1

    from .config import ClusteringConfig
    cfg = ClusteringConfig(
        window_size=args.window_size,
        stride=args.stride,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.resolution,
    )

    # Validate locomotion dir if provided
    if args.locomotion_dir is not None:
        if not (args.locomotion_dir / "config.json").is_file():
            print(f"No config.json in {args.locomotion_dir}", file=sys.stderr)
            return 1

    from .multi_project import MultiProjectPipeline

    pipeline = MultiProjectPipeline(
        project_dirs=args.project_dirs,
        output_dir=args.output_dir,
        registry_csv=args.registry_csv,
        cfg=cfg,
        batch_correction_method=args.batch_correction,
        skip_mask_verification=args.skip_mask_verification,
        locomotion_dir=args.locomotion_dir,
    )
    result = pipeline.run()

    if not args.quiet:
        print(f"\nDone: {result.n_clusters} clusters from {len(result.metadata)} windows")
        print(f"Output: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
