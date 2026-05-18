"""CLI entry point: python -m downstream_analysis.clustering ..."""

from .clustering_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
