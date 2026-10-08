"""
SQLite-backed consolidated mask storage per video (`masks.sqlite`), with dual-write
to legacy `masks/*.npz` + `bboxes/*.json` for compatibility.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

from mask_seg_codec import decode_masks_blob, encode_masks_blob, swap_keys_in_masks_blob
from video_processor import (
    bbox_norm_xywh_score_from_mask,
    load_masks_npz,
    save_bboxes_json,
    save_masks_npz,
)

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS mask_store_meta (
  key TEXT PRIMARY KEY NOT NULL,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS frame_segmentation (
  frame_idx INTEGER NOT NULL PRIMARY KEY,
  seg_blob BLOB NOT NULL,
  bbox_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_frame_seg_updated ON frame_segmentation(updated_at);
"""


class VideoMaskStorage:
    """One masks.sqlite under ``video_dir``, dual-written with NPZ/JSON."""

    def __init__(self, video_dir: str | Path) -> None:
        self.video_dir = Path(video_dir)

    @property
    def sqlite_path(self) -> Path:
        return self.video_dir / "masks.sqlite"

    @property
    def masks_dir(self) -> Path:
        return self.video_dir / "masks"

    @property
    def bboxes_dir(self) -> Path:
        return self.video_dir / "bboxes"

    def _connect(self) -> sqlite3.Connection:
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.bboxes_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema(conn)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_DDL)
        conn.execute(
            "INSERT OR IGNORE INTO mask_store_meta(key,value) VALUES ('schema_version','1')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO mask_store_meta(key,value) VALUES ('seg_codec','rle_picklezlib_v1')"
        )
        # Seed the revision from the clock (µs, still JSON-safe) so a store that
        # is wiped and recreated never reuses a revision — and therefore an
        # ETag — that a browser may still hold for the old masks.
        conn.execute(
            "INSERT OR IGNORE INTO mask_store_meta(key,value) VALUES ('cache_revision', ?)",
            (str(time.time_ns() // 1000),),
        )
        conn.commit()

    def _bump_revision_locked(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT value FROM mask_store_meta WHERE key='cache_revision'"
        ).fetchone()
        v = int(row[0]) + 1 if row else 1
        conn.execute(
            "INSERT OR REPLACE INTO mask_store_meta(key,value) VALUES ('cache_revision', ?)",
            (str(v),),
        )

    def cache_revision_int(self) -> int:
        if not self.sqlite_path.is_file():
            return 0
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value FROM mask_store_meta WHERE key='cache_revision'"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def png_cache_revision_part(self, frame_idx: int) -> tuple[str, int]:
        """Token segment for PNG encode cache keys."""
        if self.sqlite_path.is_file():
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT 1 FROM frame_segmentation WHERE frame_idx=?",
                    (frame_idx,),
                ).fetchone()
                if row is not None:
                    r = conn.execute(
                        "SELECT value FROM mask_store_meta WHERE key='cache_revision'"
                    ).fetchone()
                    return ("sqlite", int(r[0]) if r else 0)
            finally:
                conn.close()
        p = self.masks_dir / f"{frame_idx:06d}.npz"
        if p.is_file():
            return ("legacy", int(p.stat().st_mtime_ns))
        return ("none", 0)

    def has_masks(self, frame_idx: int) -> bool:
        if self.sqlite_path.is_file():
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT 1 FROM frame_segmentation WHERE frame_idx=?",
                    (frame_idx,),
                ).fetchone()
                if row is not None:
                    return True
            finally:
                conn.close()
        return (self.masks_dir / f"{frame_idx:06d}.npz").is_file()

    def load_masks_dense(self, frame_idx: int) -> dict[str, np.ndarray]:
        if self.sqlite_path.is_file():
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT seg_blob FROM frame_segmentation WHERE frame_idx=?",
                    (frame_idx,),
                ).fetchone()
                if row is not None:
                    return decode_masks_blob(row[0])
            finally:
                conn.close()
        return {
            str(k): np.asarray(v).astype(np.uint8)
            for k, v in load_masks_npz(
                str(self.masks_dir / f"{frame_idx:06d}.npz")
            ).items()
        }

    def load_bboxes(self, frame_idx: int) -> dict[str, list]:
        if self.sqlite_path.is_file():
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT bbox_json FROM frame_segmentation WHERE frame_idx=?",
                    (frame_idx,),
                ).fetchone()
                if row is not None:
                    try:
                        return json.loads(row[0])
                    except json.JSONDecodeError:
                        return {}
            finally:
                conn.close()
        p = self.bboxes_dir / f"{frame_idx:06d}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def mask_object_count(self, frame_idx: int) -> int:
        return len(self.load_masks_dense(frame_idx))

    def iter_saved_frame_indices_sorted(self) -> list[int]:
        seen: set[int] = set()
        if self.sqlite_path.is_file():
            conn = self._connect()
            try:
                for (fi,) in conn.execute(
                    "SELECT frame_idx FROM frame_segmentation ORDER BY frame_idx"
                ):
                    seen.add(int(fi))
            finally:
                conn.close()
        if self.masks_dir.is_dir():
            for npz in self.masks_dir.glob("*.npz"):
                try:
                    seen.add(int(npz.stem))
                except ValueError:
                    continue
        return sorted(seen)

    def iter_frame_indices_in_range(self, from_frame: int, to_frame: int) -> list[int]:
        if self.sqlite_path.is_file():
            return self._sql_frame_indices_in_range(from_frame, to_frame)
        out: list[int] = []
        for fidx in self.iter_saved_frame_indices_sorted():
            if from_frame >= 0 and fidx < from_frame:
                continue
            if to_frame >= 0 and fidx > to_frame:
                continue
            out.append(fidx)
        return out

    def _sql_frame_indices_in_range(self, from_frame: int, to_frame: int) -> list[int]:
        clauses: list[str] = []
        args: list[int] = []
        if from_frame >= 0:
            clauses.append("frame_idx >= ?")
            args.append(from_frame)
        if to_frame >= 0:
            clauses.append("frame_idx <= ?")
            args.append(to_frame)
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._connect()
        try:
            return [
                int(fi)
                for (fi,) in conn.execute(
                    f"SELECT frame_idx FROM frame_segmentation{where_sql} ORDER BY frame_idx",
                    args,
                )
            ]
        finally:
            conn.close()

    @staticmethod
    def _swap_bbox_json_keys(bbox_json: str, obj_a: str, obj_b: str) -> tuple[str, bool]:
        try:
            bboxes = json.loads(bbox_json or "{}")
        except json.JSONDecodeError:
            bboxes = {}
        has_a = obj_a in bboxes
        has_b = obj_b in bboxes
        if not has_a and not has_b:
            return bbox_json or "{}", False
        a_bbox = bboxes.pop(obj_a, None)
        b_bbox = bboxes.pop(obj_b, None)
        if b_bbox is not None:
            bboxes[obj_a] = b_bbox
        if a_bbox is not None:
            bboxes[obj_b] = a_bbox
        return json.dumps(bboxes, separators=(",", ":")), True

    def swap_object_masks_in_range(
        self,
        obj_a: str,
        obj_b: str,
        from_frame: int = -1,
        to_frame: int = -1,
        *,
        dual_write_legacy: bool = False,
    ) -> int:
        """
        Swap mask (and bbox) assignments between two object ids across a frame range.

        Uses in-blob RLE key swaps in SQLite — no dense decode/encode, no NPZ writes
        unless ``dual_write_legacy`` is True.
        """
        obj_a, obj_b = str(obj_a), str(obj_b)
        if obj_a == obj_b:
            return 0

        if self.sqlite_path.is_file():
            return self._swap_object_masks_sqlite(
                obj_a, obj_b, from_frame, to_frame, dual_write_legacy=dual_write_legacy
            )

        swapped = 0
        for fidx in self.iter_frame_indices_in_range(from_frame, to_frame):
            masks = self.load_masks_dense(fidx)
            has_a = obj_a in masks
            has_b = obj_b in masks
            if not has_a and not has_b:
                continue
            a_mask = masks.pop(obj_a, None)
            b_mask = masks.pop(obj_b, None)
            if b_mask is not None:
                masks[obj_a] = b_mask
            if a_mask is not None:
                masks[obj_b] = a_mask
            bboxes = self.load_bboxes(fidx)
            a_bbox = bboxes.pop(obj_a, None)
            b_bbox = bboxes.pop(obj_b, None)
            if b_bbox is not None:
                bboxes[obj_a] = b_bbox
            if a_bbox is not None:
                bboxes[obj_b] = a_bbox
            self.save_frame(fidx, masks, bboxes)
            swapped += 1
        return swapped

    def _swap_object_masks_sqlite(
        self,
        obj_a: str,
        obj_b: str,
        from_frame: int,
        to_frame: int,
        *,
        dual_write_legacy: bool,
    ) -> int:
        clauses: list[str] = []
        args: list[int] = []
        if from_frame >= 0:
            clauses.append("frame_idx >= ?")
            args.append(from_frame)
        if to_frame >= 0:
            clauses.append("frame_idx <= ?")
            args.append(to_frame)
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        swapped = 0
        touched_frames: list[int] = []
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"SELECT frame_idx, seg_blob, bbox_json FROM frame_segmentation{where_sql} ORDER BY frame_idx",
                args,
            ).fetchall()
            for frame_idx, seg_blob, bbox_json in rows:
                new_blob, changed_blob = swap_keys_in_masks_blob(seg_blob, obj_a, obj_b)
                new_bbox_json, changed_bbox = self._swap_bbox_json_keys(
                    bbox_json or "{}", obj_a, obj_b
                )
                if not changed_blob and not changed_bbox:
                    continue
                conn.execute(
                    """UPDATE frame_segmentation
                       SET seg_blob=?, bbox_json=?, updated_at=datetime('now')
                       WHERE frame_idx=?""",
                    (new_blob, new_bbox_json, frame_idx),
                )
                swapped += 1
                touched_frames.append(int(frame_idx))
            if swapped:
                self._bump_revision_locked(conn)
            conn.commit()
        finally:
            conn.close()

        if dual_write_legacy and touched_frames:
            for fidx in touched_frames:
                masks = self.load_masks_dense(fidx)
                bboxes = self.load_bboxes(fidx)
                save_masks_npz(str(self.masks_dir / f"{fidx:06d}.npz"), masks)
                save_bboxes_json(str(self.bboxes_dir / f"{fidx:06d}.json"), bboxes)

        return swapped

    def save_frame(
        self,
        frame_idx: int,
        masks: dict[str | int, np.ndarray],
        bboxes: dict[str | int, list] | None = None,
    ) -> None:
        masks_norm = {
            str(k): np.squeeze(np.asarray(v)).astype(np.uint8) for k, v in masks.items()
        }
        if not masks_norm:
            self.delete_frame(frame_idx)
            return

        bbox_merged: dict[str, list] = {}
        if bboxes:
            for k, v in bboxes.items():
                oid = str(k)
                bbox_merged[oid] = [float(x) for x in v] if isinstance(v, (list, tuple)) else v  # type: ignore[assignment]

        prev = self.load_bboxes(frame_idx)
        for oid in masks_norm:
            if oid not in bbox_merged:
                if oid in prev:
                    bbox_merged[oid] = prev[oid]
                else:
                    bx, sc = bbox_norm_xywh_score_from_mask(masks_norm[oid])
                    bbox_merged[oid] = list(bx) + [float(sc)]
        bbox_merged = {k: v for k, v in bbox_merged.items() if k in masks_norm}

        blob = encode_masks_blob(masks_norm)
        bbox_json_str = json.dumps(bbox_merged, separators=(",", ":"))

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT OR REPLACE INTO frame_segmentation
                   (frame_idx, seg_blob, bbox_json, updated_at)
                   VALUES (?,?,?,datetime('now'))""",
                (frame_idx, blob, bbox_json_str),
            )
            self._bump_revision_locked(conn)
            conn.commit()
        finally:
            conn.close()

        save_masks_npz(str(self.masks_dir / f"{frame_idx:06d}.npz"), masks_norm)
        save_bboxes_json(str(self.bboxes_dir / f"{frame_idx:06d}.json"), bbox_merged)

    def delete_frame(self, frame_idx: int) -> None:
        if self.sqlite_path.is_file():
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "DELETE FROM frame_segmentation WHERE frame_idx=?", (frame_idx,)
                )
                self._bump_revision_locked(conn)
                conn.commit()
            finally:
                conn.close()
        mp = self.masks_dir / f"{frame_idx:06d}.npz"
        if mp.exists():
            mp.unlink()
        jp = self.bboxes_dir / f"{frame_idx:06d}.json"
        if jp.exists():
            jp.unlink()

    def delete_masks_range(
        self,
        from_frame: int | None = None,
        to_frame: int | None = None,
    ) -> int:
        clauses: list[str] = []
        args: list[int] = []
        if from_frame is not None:
            clauses.append("frame_idx >= ?")
            args.append(from_frame)
        if to_frame is not None:
            clauses.append("frame_idx <= ?")
            args.append(to_frame)
        where_sql = " AND ".join(clauses) if clauses else "1=1"

        touched: set[int] = set()

        if self.sqlite_path.is_file():
            conn = self._connect()
            try:
                for (fi,) in conn.execute(
                    f"SELECT frame_idx FROM frame_segmentation WHERE {where_sql}", args
                ):
                    touched.add(int(fi))
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(f"DELETE FROM frame_segmentation WHERE {where_sql}", args)
                self._bump_revision_locked(conn)
                conn.commit()
            finally:
                conn.close()

        if self.masks_dir.exists():
            for npz in list(self.masks_dir.glob("*.npz")):
                try:
                    fidx = int(npz.stem)
                except ValueError:
                    continue
                if from_frame is not None and fidx < from_frame:
                    continue
                if to_frame is not None and fidx > to_frame:
                    continue
                touched.add(fidx)
                npz.unlink()
                jp = self.bboxes_dir / f"{fidx:06d}.json"
                if jp.exists():
                    jp.unlink()

        return len(touched)

    def wipe_sqlite_file(self) -> None:
        """Remove masks.sqlite (e.g. on full video reset)."""
        if self.sqlite_path.is_file():
            try:
                self.sqlite_path.unlink()
            except OSError as e:
                logger.warning("Could not remove %s: %s", self.sqlite_path, e)
        wal = Path(str(self.sqlite_path) + "-wal")
        shm = Path(str(self.sqlite_path) + "-shm")
        for p in (wal, shm):
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass


def legacy_npz_frame_indices(masks_dir: Path) -> list[int]:
    if not masks_dir.is_dir():
        return []
    out: list[int] = []
    for npz in masks_dir.glob("*.npz"):
        try:
            out.append(int(npz.stem))
        except ValueError:
            continue
    return sorted(out)
