"""Run locomotion histogram pipeline on a finished SAM3 Web Tracker project."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import paths as pathutil
from .bimodal import fit_bimodal_gmm, moving_fraction
from .logging_setup import configure_logging
from .locomotion import LocomotionConfig, locomotion_chunks_for_series, normalize_by_diagonal
from .metrics import distribution_snapshot
from .plots import plot_bimodal_moving_fraction, plot_overlay_cdf, plot_overlay_histogram
from .pooling import NamedSampleBundle, append_video_quantiles
from .tqdm_optional import try_tqdm
from .tracking_io import (
    build_centroid_timelines,
    load_project_config,
    npz_key_base_id,
    object_display_name,
    partition_complete_videos,
)

logger = logging.getLogger(__name__)


class LocomotionAnalysisPipeline:
    """
    Computes 90-frame / 30-frame-stride centroid path-length samples,
    writes per-video and aggregate plots under ``analysis_of_tracking_data/``.
    """

    def __init__(
        self,
        project_dir: Path,
        *,
        loc_cfg: LocomotionConfig | None = None,
        show_progress: bool = True,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.cfg = loc_cfg or LocomotionConfig()
        self.config = load_project_config(self.project_dir)
        self.pid = str(self.config.get("id") or self.project_dir.name[:8])
        self.show_progress = show_progress

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "analysis_of_tracking_data"

    def run(self) -> Path:
        out = self.output_dir
        logger.info(
            "Starting locomotion analysis | project_id=%s | dir=%s",
            self.pid,
            self.project_dir,
        )
        logger.info(
            "Locomotion windows: chunk_size=%d frames, stride=%d | metric=sum of centroid step lengths (px) within each chunk.",
            self.cfg.chunk_size,
            self.cfg.stride,
        )

        out.mkdir(parents=True, exist_ok=True)
        per_vid_dir = out / "locomotion" / "per_video"
        agg_dir = out / "locomotion" / "aggregate"
        csv_dir = out / "locomotion" / "samples"
        per_vid_dir.mkdir(parents=True, exist_ok=True)
        agg_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Output directories ready under %s (per_video / aggregate / samples).",
            out,
        )

        bundles: dict[str, NamedSampleBundle] = {}
        summary: dict[str, Any] = {
            "project_id": self.pid,
            "project_dir": str(self.project_dir),
            "locomotion": {
                "chunk_size": self.cfg.chunk_size,
                "stride": self.cfg.stride,
                "metric": "sum of centroid displacements over chunk (pixels)",
                "videos_included": [],
                "videos_skipped": [],
                "per_video": {},
                "aggregate_by_name": {},
                "bimodal_by_name": {},
            },
        }

        complete, skipped = partition_complete_videos(
            self.project_dir,
            self.config,
            progress=self.show_progress,
        )
        summary["locomotion"]["videos_skipped"] = [
            {"video_id": vid, "video_name": vname, "reason": reason}
            for vid, vname, reason in skipped
        ]

        if not complete:
            summary["locomotion"]["note"] = (
                "No videos qualify (requires propagation_complete and mask npz for every frame)."
            )
            logger.warning(
                "No fully tracked videos — wrote summary only to %s/summary.json",
                out,
            )
            with (out / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            return out

        logger.info(
            "Processing %d video(s): locomotion chunks, per-video plots, CSV exports, then aggregates.",
            len(complete),
        )

        color_by_name: dict[str, str] = {}

        vid_enum = try_tqdm(
            list(enumerate(complete, start=1)),
            total=len(complete),
            desc="Process videos",
            unit="video",
            leave=True,
            disable=not self.show_progress,
        )

        for vi, ctx in vid_enum:
            summary["locomotion"]["videos_included"].append(ctx.video_id)
            vid_slug = pathutil.safe_filename_fragment(ctx.video_name)
            logger.info(
                '[%d/%d] Video "%s" (id=%s) — loading centroids and computing sliding-window locomotion.',
                vi,
                len(complete),
                ctx.video_name,
                ctx.video_id,
            )

            series, start, n, w, h = build_centroid_timelines(
                ctx,
                progress=self.show_progress,
            )

            per_video_series_abs: dict[str, list[np.ndarray]] = defaultdict(list)
            per_video_series_norm: dict[str, list[np.ndarray]] = defaultdict(list)

            vid_entry: dict[str, Any] = {
                "video_name": ctx.video_name,
                "frame_range": [start, n],
                "objects": {},
            }

            keys_iter = try_tqdm(
                series.items(),
                desc=f'Objects [{ctx.video_name[:18]}]',
                unit="track",
                leave=False,
                disable=not self.show_progress,
            )
            for npz_key, xy in keys_iter:
                disp_name = object_display_name(ctx.config, npz_key)
                if disp_name is None:
                    logger.debug("Skip npz key %r (no config object).", npz_key)
                    continue
                obj_meta = (ctx.config.get("objects") or {}).get(npz_key_base_id(npz_key), {})
                color = str(obj_meta.get("color") or "#888888")
                color_by_name.setdefault(disp_name, color)

                _, vals = locomotion_chunks_for_series(xy, cfg=self.cfg)
                abs_clean = vals.copy()
                norm_clean = normalize_by_diagonal(vals, w, h)

                n_chunks = int(np.sum(np.isfinite(abs_clean)))
                vid_entry["objects"][f"{disp_name}::{npz_key}"] = {
                    "chunk_count": n_chunks,
                    "mean_abs_px": float(np.nanmean(abs_clean)),
                    "mean_norm_diag": float(np.nanmean(norm_clean)),
                }

                per_video_series_abs[disp_name].append(abs_clean)
                per_video_series_norm[disp_name].append(norm_clean)

                append_video_quantiles(bundles, disp_name, ctx.video_id, abs_clean, norm_clean)

            pv_abs = {k: np.concatenate(v) for k, v in per_video_series_abs.items()}
            pv_norm = {k: np.concatenate(v) for k, v in per_video_series_norm.items()}

            summary["locomotion"]["per_video"][ctx.video_id] = vid_entry

            pnorm_path = per_vid_dir / f"{ctx.video_id}_{vid_slug}_normalized.png"
            pabs_path = per_vid_dir / f"{ctx.video_id}_{vid_slug}_absolute_px.png"
            pnorm_cdf = per_vid_dir / f"{ctx.video_id}_{vid_slug}_normalized_cdf.png"
            pabs_cdf = per_vid_dir / f"{ctx.video_id}_{vid_slug}_absolute_px_cdf.png"
            logger.info(
                '[%d/%d] Saving histograms + CDFs for "%s": %s, %s, %s, %s',
                vi,
                len(complete),
                ctx.video_name,
                pnorm_path.name,
                pnorm_cdf.name,
                pabs_path.name,
                pabs_cdf.name,
            )
            plot_overlay_histogram(
                pv_norm,
                title=f"Locomotion (diag-normalized) · {ctx.video_name}",
                xlabel="chunk path length / √(w²+h²)",
                outfile=pnorm_path,
                colors=color_by_name,
            )
            plot_overlay_cdf(
                pv_norm,
                title=f"Locomotion (diag-normalized) · {ctx.video_name}",
                xlabel="chunk path length / √(w²+h²)",
                outfile=pnorm_cdf,
                colors=color_by_name,
            )
            plot_overlay_histogram(
                pv_abs,
                title=f"Locomotion (pixels) · {ctx.video_name}",
                xlabel="chunk path length (px)",
                outfile=pabs_path,
                colors=color_by_name,
                log_x=True,
            )
            plot_overlay_cdf(
                pv_abs,
                title=f"Locomotion (pixels) · {ctx.video_name}",
                xlabel="chunk path length (px)",
                outfile=pabs_cdf,
                colors=color_by_name,
                log_x=True,
            )

            n_csv = 0
            for name, arr in pv_norm.items():
                vfin = arr[np.isfinite(arr)]
                if vfin.size == 0:
                    continue
                p_csv = csv_dir / f"{ctx.video_id}_{vid_slug}_{pathutil.safe_filename_fragment(name)}.csv"
                np.savetxt(
                    p_csv,
                    np.column_stack([vfin]),
                    delimiter=",",
                    header="normalized_chunk_path_length",
                    comments="",
                )
                n_csv += 1
            logger.info(
                '[%d/%d] Wrote %d CSV sample file(s) for "%s".',
                vi,
                len(complete),
                n_csv,
                ctx.video_name,
            )

        logger.info(
            "Building aggregate distributions for %d object name group(s): %s",
            len(bundles),
            ", ".join(sorted(bundles.keys(), key=str.lower)),
        )

        abs_concat = {name: np.array(b.absolute_pixels, dtype=np.float64) for name, b in bundles.items()}
        norm_concat = {name: np.array(b.normalized, dtype=np.float64) for name, b in bundles.items()}
        quant_concat = {name: np.array(b.quantile_ranks, dtype=np.float64) for name, b in bundles.items()}

        for name, b in bundles.items():
            summary["locomotion"]["aggregate_by_name"][name] = {
                "n_chunks": len(b.absolute_pixels),
                "videos": sorted(set(b.video_ids)),
                "normalized_diag_pool": distribution_snapshot(norm_concat[name]),
                "absolute_px_pool": distribution_snapshot(abs_concat[name]),
                "quantile_pool": distribution_snapshot(quant_concat[name]),
            }

        logger.info("Saving pooled histograms + CDFs (normalized, absolute px, quantile ranks).")
        plot_overlay_histogram(
            norm_concat,
            title="Pooled locomotion by object name (diag-normalized)",
            xlabel="chunk path length / √(w²+h²)",
            outfile=agg_dir / "by_name_normalized.png",
            colors=color_by_name,
        )
        plot_overlay_cdf(
            norm_concat,
            title="Pooled locomotion by object name (diag-normalized)",
            xlabel="chunk path length / √(w²+h²)",
            outfile=agg_dir / "by_name_normalized_cdf.png",
            colors=color_by_name,
        )
        plot_overlay_histogram(
            abs_concat,
            title="Pooled locomotion by object name (raw pixels)",
            xlabel="chunk path length (px)",
            outfile=agg_dir / "by_name_absolute_px.png",
            colors=color_by_name,
            log_x=True,
        )
        plot_overlay_cdf(
            abs_concat,
            title="Pooled locomotion by object name (raw pixels)",
            xlabel="chunk path length (px)",
            outfile=agg_dir / "by_name_absolute_px_cdf.png",
            colors=color_by_name,
            log_x=True,
        )
        plot_overlay_histogram(
            quant_concat,
            title="Pooled within-video quantile ranks",
            xlabel="quantile rank ∈ [0,1]",
            outfile=agg_dir / "by_name_quantile_pool.png",
            colors=color_by_name,
            bins=40,
        )
        plot_overlay_cdf(
            quant_concat,
            title="Pooled within-video quantile ranks",
            xlabel="quantile rank ∈ [0,1]",
            outfile=agg_dir / "by_name_quantile_pool_cdf.png",
            colors=color_by_name,
        )

        names_sorted = sorted(norm_concat.keys(), key=lambda s: s.lower())
        frac_list: list[float | None] = []
        bimodal_rows: list[dict[str, Any]] = []

        logger.info(
            "Fitting 2-component GMM on log(metric) for bimodal / moving-vs-quiet comparison (%d groups).",
            len(names_sorted),
        )
        for name in try_tqdm(
            names_sorted,
            desc="Bimodal GMM",
            unit="name",
            leave=True,
            disable=not self.show_progress,
        ):
            arr = norm_concat[name]
            fit = fit_bimodal_gmm(arr)
            frac = moving_fraction(fit, arr)
            frac_list.append(frac)
            row: dict[str, Any] = {"object_name": name, "moving_fraction": frac}
            if fit is not None:
                row.update(
                    {
                        "threshold_normalized": fit.threshold,
                        "component_low_mean": fit.low_mean,
                        "component_high_mean": fit.high_mean,
                        "weights": [fit.low_weight, fit.high_weight],
                        "bic": fit.bic,
                    },
                )
                logger.debug(
                    "Bimodal %r: threshold_norm=%.6f P(moving)=%s BIC=%s",
                    name,
                    fit.threshold,
                    frac,
                    fit.bic,
                )
            else:
                logger.debug(
                    "Bimodal %r: skipped (need scikit-learn + ≥10 finite samples).",
                    name,
                )
            summary["locomotion"]["bimodal_by_name"][name] = row
            bimodal_rows.append(row)

        bimodal_png = agg_dir / "bimodal_moving_fraction.png"
        logger.info("Saving bimodal summary bar chart: %s", bimodal_png)
        plot_bimodal_moving_fraction(
            names_sorted,
            frac_list,
            bimodal_png,
        )

        summary_path = out / "summary.json"
        logger.info("Writing %s", summary_path)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        bimodal_json = agg_dir / "bimodal_table.json"
        with bimodal_json.open("w", encoding="utf-8") as f:
            json.dump(bimodal_rows, f, indent=2, default=str)

        logger.info("Finished locomotion analysis — all outputs under %s", out)
        return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Downstream locomotion analysis for SAM3 Web Tracker projects.")
    p.add_argument("--project-dir", type=Path, help="Folder containing config.json")
    p.add_argument("--project-id", type=str, help="Short project uuid")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    p.add_argument("-q", "--quiet", action="store_true", help="WARNING only; no progress bars")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Keep INFO logs but disable tqdm bars",
    )
    args = p.parse_args(argv)

    if args.verbose and args.quiet:
        print("Cannot use both --verbose and --quiet", file=sys.stderr)
        return 2

    configure_logging(verbose=args.verbose, quiet=args.quiet)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    show_progress = not args.no_progress and not args.quiet

    if args.project_dir:
        proj = Path(args.project_dir).resolve()
        if not (proj / "config.json").is_file():
            print("No config.json in", proj, file=sys.stderr)
            return 1
    elif args.project_id:
        found = pathutil.find_project_dir(args.project_id.strip())
        if found is None:
            print("Project id not found:", args.project_id, file=sys.stderr)
            return 1
        proj = found
    else:
        p.print_help()
        return 2

    pipe = LocomotionAnalysisPipeline(proj, show_progress=show_progress)
    out = pipe.run()
    if not args.quiet:
        print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
