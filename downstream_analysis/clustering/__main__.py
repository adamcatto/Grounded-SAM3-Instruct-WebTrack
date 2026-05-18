"""CLI entry point: python -m downstream_analysis.clustering ..."""

import sys

# Force unbuffered stdout so output appears immediately under conda run
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("[clustering] Loading modules...", flush=True)

from .clustering_pipeline import main

print("[clustering] Modules loaded. Starting pipeline.", flush=True)

if __name__ == "__main__":
    raise SystemExit(main())
