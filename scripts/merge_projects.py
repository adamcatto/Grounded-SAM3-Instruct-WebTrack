#!/usr/bin/env python3
"""
Merge two SAM3 WebTrack projects into a new project (source projects are not deleted).

Examples:
  python scripts/merge_projects.py abc12345 def67890 --name "Combined study"
  python scripts/merge_projects.py --left abc12345 --right def67890 -n "Combined study"
  python scripts/merge_projects.py /path/to/proj-a /path/to/proj-b -n "Merged" -o /path/to/parent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from project_manager import ProjectManager  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge two projects into a new project (originals are kept).",
    )
    p.add_argument(
        "left",
        nargs="?",
        help="Left project: short id, folder name, or path to project directory",
    )
    p.add_argument(
        "right",
        nargs="?",
        help="Right project: short id, folder name, or path to project directory",
    )
    p.add_argument(
        "--left",
        dest="left_named",
        metavar="REF",
        help="Left project (named alternative to positional)",
    )
    p.add_argument(
        "--right",
        dest="right_named",
        metavar="REF",
        help="Right project (named alternative to positional)",
    )
    p.add_argument(
        "--name",
        "-n",
        required=True,
        help="Display name for the merged project",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help=(
            "Parent directory for the new project folder "
            "(default: shared parent of the two sources, or the active projects root)"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    left_ref = args.left_named or args.left
    right_ref = args.right_named or args.right
    if not left_ref or not right_ref:
        _build_parser().print_help()
        print("\nerror: provide left and right projects (positional or --left/--right)", file=sys.stderr)
        return 2

    pm = ProjectManager()
    output_parent = Path(args.output_dir).expanduser() if args.output_dir else None
    try:
        merged = pm.merge_projects(
            name=args.name.strip(),
            left_ref=left_ref,
            right_ref=right_ref,
            output_parent=output_parent,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    project_dir = pm._find_project_dir(merged["id"])
    print(json.dumps(merged, indent=2))
    if project_dir is not None:
        print(f"\nCreated merged project at: {project_dir}", flush=True)
        if project_dir.parent.resolve() != pm.projects_base_dir().resolve():
            print(
                f"Note: set the app projects folder to {project_dir.parent} "
                "if the merged project does not appear in the UI.",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
