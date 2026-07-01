#!/usr/bin/env python3
"""
Reversible verification of mask swap/clear at API, disk (SQLite + NPZ), and config layers.

Sandbox: 9f8a7b6c / f09434ac — restores all mutations before exit.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from mask_store import VideoMaskStorage  # noqa: E402

API = "http://localhost:8000/api"
PID = "9f8a7b6c"
VID = "f09434ac"
ANCHOR = 160
NON_ANCHOR = 500
OBJ_A, OBJ_B = "1", "2"
VIDEO_DIR = Path(
    "/opt/projects/segmentation_tracking_projects/"
    "9f8a7b6c-sandbox_Home-Cage-Interactions-0126-test-day/"
    "videos/f09434ac_1A_video_test1a_20260130_090255.mp4"
)


class Check:
    def __init__(self, name: str):
        self.name = name
        self.ok = True
        self.notes: list[str] = []

    def fail(self, msg: str):
        self.ok = False
        self.notes.append(f"FAIL: {msg}")

    def info(self, msg: str):
        self.notes.append(msg)


def mask_fingerprint(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def dense_meta(ms: VideoMaskStorage, frame: int) -> dict:
    masks = ms.load_masks_dense(frame)
    bboxes = ms.load_bboxes(frame)
    return {
        "areas": {k: int(v.sum()) for k, v in masks.items()},
        "fps": {k: mask_fingerprint(v) for k, v in masks.items()},
        "bboxes": bboxes,
        "objects": sorted(masks.keys()),
    }


def npz_matches_sqlite(ms: VideoMaskStorage, frame: int) -> bool:
    npz_path = ms.masks_dir / f"{frame:06d}.npz"
    if not npz_path.is_file():
        return True
    sqlite_masks = ms.load_masks_dense(frame)
    import numpy as np

    from video_processor import load_masks_npz

    npz_masks = {str(k): np.asarray(v).astype(np.uint8) for k, v in load_masks_npz(str(npz_path)).items()}
    if set(sqlite_masks.keys()) != set(npz_masks.keys()):
        return False
    for k in sqlite_masks:
        if not np.array_equal(sqlite_masks[k], npz_masks[k]):
            return False
    return True


def _api(method: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
    url = f"{API}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} → {e.code}: {e.read().decode()}") from e
    except TimeoutError as e:
        raise RuntimeError(f"{method} {path} timed out after {timeout}s") from e


def get_project() -> dict:
    return _api("GET", f"/projects/{PID}")


def get_masks_api(frame: int) -> dict:
    return _api("GET", f"/projects/{PID}/videos/{VID}/masks/{frame}").get("masks") or {}


def get_session_state() -> dict | None:
    try:
        return _api("GET", f"/projects/{PID}/videos/{VID}/session/state", timeout=8)
    except RuntimeError:
        return None


def swap(frame: int) -> int:
    return _api(
        "POST",
        f"/projects/{PID}/videos/{VID}/masks/swap",
        {"obj_a": OBJ_A, "obj_b": OBJ_B, "from_frame": frame, "to_frame": frame},
        timeout=300,
    )["frames_swapped"]


def clear_object_frame(oid: str, frame: int):
    _api("DELETE", f"/projects/{PID}/videos/{VID}/objects/{oid}/frames/{frame}/points")


def clear_frame_masks(frame: int):
    _api("DELETE", f"/projects/{PID}/videos/{VID}/masks/{frame}")


def restore_masks(frames: dict[str, dict]):
    _api("POST", f"/projects/{PID}/videos/{VID}/masks/restore_frames", {"frames": frames})


def restore_masks_disk(ms: VideoMaskStorage, frame: int, before_masks: dict, before_bboxes: dict):
    ms.save_frame(frame, before_masks, before_bboxes)


def restore_object_frame_prompt(oid: str, frame: int, prompt: dict | None):
    """Restore or clear one object/frame prompt (avoids rewriting all anchor frames)."""
    body = (
        {"points": prompt["points"], "labels": prompt["labels"]}
        if prompt
        else {"points": [], "labels": []}
    )
    _api(
        "PUT",
        f"/projects/{PID}/videos/{VID}/objects/{oid}/frames/{frame}/prompts",
        body,
        timeout=30,
    )


def get_frame_prompt(oid: str, frame: int) -> dict | None:
    pp = get_project()["videos"][VID].get("point_prompts", {})
    return pp.get(oid, {}).get(str(frame))


def swap_disk(ms: VideoMaskStorage, frame: int) -> None:
    """Mirror server swap logic on disk (fast — no full-frame index scan)."""
    masks = ms.load_masks_dense(frame)
    has_a = OBJ_A in masks
    has_b = OBJ_B in masks
    if not has_a and not has_b:
        return
    a_mask = masks.pop(OBJ_A, None)
    b_mask = masks.pop(OBJ_B, None)
    if b_mask is not None:
        masks[OBJ_A] = b_mask
    if a_mask is not None:
        masks[OBJ_B] = a_mask
    bboxes = ms.load_bboxes(frame)
    a_bbox = bboxes.pop(OBJ_A, None)
    b_bbox = bboxes.pop(OBJ_B, None)
    if b_bbox is not None:
        bboxes[OBJ_A] = b_bbox
    if a_bbox is not None:
        bboxes[OBJ_B] = a_bbox
    ms.save_frame(frame, masks, bboxes)


def test_swap_disk(ms: VideoMaskStorage, frame: int, label: str) -> Check:
    c = Check(f"swap disk @ frame {frame} ({label})")
    print(f"  swap disk test frame {frame}...", flush=True)
    before = dense_meta(ms, frame)
    swap_disk(ms, frame)
    after = dense_meta(ms, frame)
    if after["fps"].get(OBJ_A) != before["fps"].get(OBJ_B):
        c.fail(f"obj {OBJ_A} fingerprint did not take obj {OBJ_B}'s mask")
    if after["fps"].get(OBJ_B) != before["fps"].get(OBJ_A):
        c.fail(f"obj {OBJ_B} fingerprint did not take obj {OBJ_A}'s mask")
    if not npz_matches_sqlite(ms, frame):
        c.fail("SQLite and legacy NPZ disagree after disk swap")
    swap_disk(ms, frame)
    restored = dense_meta(ms, frame)
    if restored["fps"] != before["fps"]:
        c.fail("double-swap did not restore disk state")
    else:
        c.info("disk swap + restore OK")
    return c


def test_swap_api(ms: VideoMaskStorage, frame: int, label: str) -> Check:
    c = Check(f"swap API @ frame {frame} ({label})")
    print(f"  swap API test frame {frame} (may take ~2 min on large projects)...", flush=True)
    before = dense_meta(ms, frame)
    if OBJ_A not in before["fps"] or OBJ_B not in before["fps"]:
        c.fail(f"precondition: both objects must have masks (got {before['objects']})")
        return c
    prompt_snap = {
        oid: get_frame_prompt(oid, frame)
        for oid in (OBJ_A, OBJ_B)
    }

    n = swap(frame)
    if n != 1:
        c.fail(f"expected frames_swapped=1, got {n}")

    after = dense_meta(ms, frame)
    if after["fps"].get(OBJ_A) != before["fps"].get(OBJ_B):
        c.fail(f"obj {OBJ_A} fingerprint did not take obj {OBJ_B}'s mask")
    if after["fps"].get(OBJ_B) != before["fps"].get(OBJ_A):
        c.fail(f"obj {OBJ_B} fingerprint did not take obj {OBJ_A}'s mask")

    if not npz_matches_sqlite(ms, frame):
        c.fail("SQLite and legacy NPZ disagree after swap")

    api_masks = get_masks_api(frame)
    if len(api_masks) != len(before["objects"]):
        c.fail(f"API mask count changed: {len(api_masks)} vs {len(before['objects'])}")

    for oid, pr in prompt_snap.items():
        after_pr = get_frame_prompt(oid, frame)
        if json.dumps(pr, sort_keys=True) != json.dumps(after_pr, sort_keys=True):
            c.fail(f"point_prompts changed for obj {oid} frame {frame} after swap")

    swap(frame)  # restore
    restored = dense_meta(ms, frame)
    if restored["fps"] != before["fps"]:
        c.fail("double-swap did not restore original masks")
    else:
        c.info("double-swap restored disk state OK")

    return c


def test_clear_object(ms: VideoMaskStorage, frame: int, label: str) -> Check:
    c = Check(f"clear object {OBJ_A} @ frame {frame} ({label})")
    print(f"  clear object test frame {frame}...", flush=True)
    before = dense_meta(ms, frame)
    before_masks = ms.load_masks_dense(frame)
    before_bboxes = ms.load_bboxes(frame)
    prompt_before = get_frame_prompt(OBJ_A, frame)

    clear_object_frame(OBJ_A, frame)

    after = dense_meta(ms, frame)
    if OBJ_A in after["fps"]:
        c.fail(f"obj {OBJ_A} mask still on disk")
    if OBJ_B not in after["fps"]:
        c.fail(f"obj {OBJ_B} mask missing after partial clear")

    if not npz_matches_sqlite(ms, frame):
        c.fail("SQLite/NPZ mismatch after object clear")

    api_after = get_masks_api(frame)
    if OBJ_A in api_after:
        c.fail(f"API still returns mask for obj {OBJ_A}")
    if OBJ_B not in api_after:
        c.fail(f"API missing mask for obj {OBJ_B}")

    has_prompt = get_frame_prompt(OBJ_A, frame) is not None
    if prompt_before and has_prompt:
        c.fail(f"frame {frame}: point prompt for obj {OBJ_A} should be cleared")
    if not prompt_before and has_prompt:
        c.fail("non-anchor: unexpected prompt added")

    restore_masks_disk(ms, frame, before_masks, before_bboxes)
    if prompt_before is not None:
        restore_object_frame_prompt(OBJ_A, frame, prompt_before)
    final = dense_meta(ms, frame)
    if final["fps"] != before["fps"]:
        c.fail("restore did not recover masks")
    c.info("restore OK")

    return c


def test_clear_frame(ms: VideoMaskStorage, frame: int, label: str) -> Check:
    c = Check(f"clear all masks @ frame {frame} ({label})")
    print(f"  clear frame test {frame}...", flush=True)
    before = dense_meta(ms, frame)
    before_masks = ms.load_masks_dense(frame)
    before_bboxes = ms.load_bboxes(frame)
    prompt_snaps = {oid: get_frame_prompt(oid, frame) for oid in (OBJ_A, OBJ_B)}

    clear_frame_masks(frame)

    if ms.has_masks(frame):
        c.fail("masks still present on disk after clear_frame_masks")
    api_after = get_masks_api(frame)
    if api_after:
        c.fail(f"API still returns masks: {list(api_after.keys())}")

    for oid, pr in prompt_snaps.items():
        after_pr = get_frame_prompt(oid, frame)
        if json.dumps(pr, sort_keys=True) != json.dumps(after_pr, sort_keys=True):
            c.fail(f"bulk frame mask clear changed prompt obj {oid} frame {frame}")

    restore_masks_disk(ms, frame, before_masks, before_bboxes)
    if not ms.has_masks(frame):
        c.fail("restore did not bring masks back")
    c.info("restore OK")
    return c


def test_inference_gaps() -> Check:
    """Document inference-state behavior when no active SAM session."""
    c = Check("inference state introspection")
    print("  inference gap check...", flush=True)
    state = get_session_state()
    if state is None:
        c.info("session/state timed out or unavailable without active session")
        c.info("GAP: swap does not update SAM cached outputs (by design — disk only)")
        c.info("GAP: clear_object_frame_points does not call SAM (config + disk only)")
        return c
    c.info(f"session_active={state.get('session_active')}")
    if not state.get("session_active"):
        c.info("No active session — inference cache not inspectable via API")
    return c


def main() -> int:
    print("Mask ops reversible test starting...", flush=True)
    ms = VideoMaskStorage(VIDEO_DIR)
    checks: list[Check] = []
    checks.append(test_inference_gaps())

    for frame, label in [(ANCHOR, "anchor"), (NON_ANCHOR, "non-anchor")]:
        meta = dense_meta(ms, frame)
        if OBJ_A not in meta["fps"] or OBJ_B not in meta["fps"]:
            print(f"SKIP: frame {frame} missing both object masks")
            continue
        if not npz_matches_sqlite(ms, frame):
            pre = Check(f"pre-check sqlite/npz frame {frame}")
            pre.fail("SQLite and NPZ disagree before tests")
            checks.append(pre)
        checks.append(test_swap_disk(ms, frame, label))

    # One API swap round-trip (slow on 59k-frame projects — scans all saved indices)
    checks.append(test_swap_api(ms, NON_ANCHOR, "non-anchor API"))

    checks.append(test_clear_object(ms, ANCHOR, "anchor"))
    checks.append(test_clear_object(ms, NON_ANCHOR, "non-anchor"))
    checks.append(test_clear_frame(ms, NON_ANCHOR, "non-anchor"))

    # Final sanity: baseline frames still intact
    for frame in (ANCHOR, NON_ANCHOR):
        if not ms.has_masks(frame):
            final = Check(f"post-check frame {frame}")
            final.fail("masks missing after all tests — restore failed")
            checks.append(final)

    print("\n=== Mask ops reversible test report ===\n")
    failures = 0
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        print(f"[{status}] {c.name}")
        for n in c.notes:
            print(f"       {n}")
        if not c.ok:
            failures += 1

    print(f"\n{len(checks) - failures}/{len(checks)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
