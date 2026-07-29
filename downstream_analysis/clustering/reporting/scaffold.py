#!/usr/bin/env python3
"""Draft a ReportSpec JSON from a completed clustering run.

Pre-fills numbers, discovers the best headline figure, and drafts finding stubs
from the strongest feature effects -- all TODO-marked so the agent knows to
replace them with real, plain-language reads of the data. Nothing here invents a
scientific claim; it only lays out the skeleton for the agent to curate.

Usage:
    python -m downstream_analysis.clustering.reporting.scaffold \
        --project-dir /path/to/project [--output-dir /merged/out] --out spec.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .spec import scaffold_spec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--output-dir", default=None,
                    help="Explicit clustering output dir (for merged runs)")
    ap.add_argument("--out", required=True, help="Where to write the draft spec JSON")
    args = ap.parse_args()

    spec = scaffold_spec(args.project_dir, args.output_dir)
    spec.to_json(args.out)
    n_findings = len(spec.findings)
    print(f"Draft spec written: {args.out}")
    print(f"  stats: {spec.stats}")
    print(f"  key figure: {spec.key_figure.get('image') or '(none found)'}")
    print(f"  drafted {n_findings} finding stub(s) — replace all TODO text before rendering.")


if __name__ == "__main__":
    main()
