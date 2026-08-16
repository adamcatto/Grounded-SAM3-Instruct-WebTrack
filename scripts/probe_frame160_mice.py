#!/usr/bin/env python3
"""Locate the two dark mice on sandbox frame 160 and probe SAM3 text vs points."""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

API = "http://127.0.0.1:8000/api"
PID = "9f8a7b6c"
VID = "f09434ac"
FRAME = 160
HEAD = "1"
NO = "2"
OUT = Path("/opt/cursor/artifacts/screenshots")
REPO = Path("/opt/software/Grounded-SAM3-Instruct-WebTrack/reports/agent-e2e-a100-shared/gt")
OUT.mkdir(parents=True, exist_ok=True)
REPO.mkdir(parents=True, exist_ok=True)


def api(method: str, path: str, body=None, timeout=180):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            ctype = r.headers.get("content-type", "")
            if "json" in ctype or raw[:1] in (b"{", b"["):
                return json.loads(raw.decode())
            return raw
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} -> {e.code}: {err[:800]}") from e


def connected_components(binary: np.ndarray):
    h, w = binary.shape
    vis = np.zeros((h, w), dtype=np.int32)
    blobs = []
    bid = 0
    for y in range(h):
        xs = np.where(binary[y])[0]
        for x in xs:
            if vis[y, x]:
                continue
            bid += 1
            q = deque([(y, x)])
            vis[y, x] = bid
            pix = []
            while q:
                cy, cx = q.popleft()
                pix.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and vis[ny, nx] == 0:
                        vis[ny, nx] = bid
                        q.append((ny, nx))
            ys = np.array([p[0] for p in pix])
            xs2 = np.array([p[1] for p in pix])
            blobs.append({
                "id": bid,
                "area": len(pix),
                "cx": float(xs2.mean()) / w,
                "cy": float(ys.mean()) / h,
                "x0": int(xs2.min()),
                "x1": int(xs2.max()),
                "y0": int(ys.min()),
                "y1": int(ys.max()),
            })
    return vis, blobs


def decode_png_mask(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    a = np.asarray(im)
    return a[:, :, 3] > 16


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    if pred.sum() == 0:
        return 0.0
    return float(np.logical_and(pred, gt).sum() / pred.sum())


def overlay(frame: Image.Image, masks: dict[str, np.ndarray], colors: dict[str, tuple], path: Path):
    base = frame.convert("RGBA")
    h, w = np.asarray(frame).shape[:2]
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    for name, m in masks.items():
        if m is None:
            continue
        mm = m
        if mm.shape != (h, w):
            mm = np.array(Image.fromarray(mm.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        c = colors[name]
        layer[mm] = (*c, 110)
    out = Image.alpha_composite(base, Image.fromarray(layer, "RGBA")).convert("RGB")
    out.save(path)
    print("saved", path)


def mask_stats(m: np.ndarray, name: str):
    h, w = m.shape
    ys, xs = np.where(m)
    if xs.size == 0:
        print(f"  {name}: EMPTY")
        return
    cx = float(xs.mean()) / w
    cy = float(ys.mean()) / h
    print(f"  {name}: area={int(m.sum())} cx={cx:.3f} cy={cy:.3f} "
          f"bbox=[{xs.min()/w:.3f},{ys.min()/h:.3f}]-[{xs.max()/w:.3f},{ys.max()/h:.3f}]")


def build_gt(frame_path: Path):
    img = Image.open(frame_path).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    inset = (xx > 0.12 * w) & (xx < 0.92 * w) & (yy > 0.18 * h) & (yy < 0.88 * h)
    binary = (lum < 50) & inset
    vis, blobs = connected_components(binary)
    blobs = sorted(blobs, key=lambda b: -b["area"])
    top = [b for b in blobs if b["area"] >= 2000][:4]
    print("GT blobs lum<50:")
    for b in top:
        print(f"  area={b['area']} cx={b['cx']:.3f} cy={b['cy']:.3f}")
    if len(top) < 2:
        raise SystemExit("expected two dark mice")
    a, b = sorted(top[:2], key=lambda x: x["cx"])
    left = vis == a["id"]
    right = vis == b["id"]
    gt = {
        "left_noshave": {
            "mask": left,
            "cx": a["cx"],
            "cy": a["cy"],
            "area": a["area"],
        },
        "right_headshave": {
            "mask": right,
            "cx": b["cx"],
            "cy": b["cy"],
            "area": b["area"],
        },
    }
    overlay(
        img,
        {"left": left, "right": right},
        {"left": (232, 164, 69), "right": (91, 141, 217)},
        OUT / "gt-two-mice.png",
    )
    overlay(img, {"left": left, "right": right}, {"left": (232, 164, 69), "right": (91, 141, 217)}, REPO / "gt-two-mice.png")
    np.savez_compressed(REPO / "gt_masks.npz", left=left.astype(np.uint8), right=right.astype(np.uint8))
    json.dump(
        {
            "left_noshave": {"cx": a["cx"], "cy": a["cy"], "area": a["area"]},
            "right_headshave": {"cx": b["cx"], "cy": b["cy"], "area": b["area"]},
        },
        open(REPO / "gt.json", "w"),
        indent=2,
    )
    return img, gt


def reset_and_prep():
    print("reset...")
    print(api("POST", f"/projects/{PID}/videos/{VID}/reset?keep_objects=true"))
    print("extract annotated frame...")
    print(api("POST", f"/projects/{PID}/videos/{VID}/extract_frame/{FRAME}"))
    urllib.request.urlopen(API + f"/projects/{PID}/videos/{VID}/frames/{FRAME}", timeout=60).read()
    print("init session...")
    print(api("POST", f"/projects/{PID}/videos/{VID}/session"))


def score_against_gt(pred: np.ndarray, gt):
    left = gt["left_noshave"]["mask"]
    right = gt["right_headshave"]["mask"]
    if pred.shape != left.shape:
        pred = np.array(Image.fromarray(pred.astype(np.uint8) * 255).resize(
            (left.shape[1], left.shape[0]), Image.NEAREST)) > 0
    i_l, i_r = iou(pred, left), iou(pred, right)
    p_l, p_r = precision(pred, left), precision(pred, right)
    return {
        "iou_left": round(i_l, 4),
        "iou_right": round(i_r, 4),
        "prec_left": round(p_l, 4),
        "prec_right": round(p_r, 4),
        "match": "left" if i_l >= i_r else "right",
        "best_iou": round(max(i_l, i_r), 4),
        "best_prec": round(max(p_l, p_r), 4),
        "ok": max(i_l, i_r) >= 0.25 and max(p_l, p_r) >= 0.35,
    }


def pngs_from_text_result(result: dict) -> dict[str, np.ndarray]:
    out = {}
    for oid, b64 in (result.get("masks") or {}).items():
        if isinstance(b64, str) and len(b64) > 80:
            out[str(oid)] = decode_png_mask(b64)
    return out


def run_text(gt, img):
    print("\n=== TEXT SEGMENT ===")
    results = {}
    for oid, name, text in (
        (NO, "NoShave", "dark mouse"),
        (HEAD, "HeadShave", "dark mouse"),
    ):
        print(f"-- text_segment obj {oid} {name!r} text={text!r}")
        r = api("POST", f"/projects/{PID}/videos/{VID}/text_segment", {
            "frame_idx": FRAME,
            "text": text,
            "object_id": oid,
            "object_name": name,
            "min_score": 0.0,
        })
        slim = {k: r.get(k) for k in (
            "ok", "method", "quality", "picked_side", "picked_center_xy",
            "detection_count", "all_detections", "suggestion", "error",
        )}
        print(json.dumps(slim, indent=2, default=str)[:2000])
        masks = pngs_from_text_result(r)
        scored = {}
        for k, m in masks.items():
            mask_stats(m, f"pred {k}")
            scored[k] = score_against_gt(m, gt)
        print("scores", scored)
        results[oid] = {"resp": slim, "scores": scored, "masks": masks}
    merged = {}
    colors = {NO: (232, 164, 69), HEAD: (91, 141, 217)}
    for oid, pack in results.items():
        if pack["masks"]:
            merged[oid] = next(iter(pack["masks"].values()))
    overlay(img, merged, colors, OUT / "probe-text-dark-mouse.png")
    overlay(img, merged, colors, REPO / "probe-text-dark-mouse.png")
    return results


def run_points(gt, img, tag: str, clicks: list[tuple[str, str, list[list[float]], list[int]]]):
    print(f"\n=== POINTS {tag} ===")
    reset_and_prep()
    merged = {}
    colors = {NO: (232, 164, 69), HEAD: (91, 141, 217)}
    scores = {}
    for oid, name, pts, labels in clicks:
        print(f"-- add_point obj {oid} {name} pts={pts} labels={labels}")
        r = api("POST", f"/projects/{PID}/videos/{VID}/objects/{oid}/points", {
            "frame_idx": FRAME,
            "points": pts,
            "labels": labels,
            "anchor_mode": True,
        })
        masks = {}
        for k, b64 in (r.get("masks") or {}).items():
            if isinstance(b64, str) and len(b64) > 80:
                masks[str(k)] = decode_png_mask(b64)
        print(" returned mask ids", list(masks))
        if oid in masks:
            merged[oid] = masks[oid]
            mask_stats(masks[oid], f"{name}")
            scores[oid] = score_against_gt(masks[oid], gt)
            print(" score", scores[oid])
        elif masks:
            k = next(iter(masks))
            merged[oid] = masks[k]
            mask_stats(masks[k], f"{name}/{k}")
            scores[oid] = score_against_gt(masks[k], gt)
            print(" score", scores[oid])
    overlay(img, merged, colors, OUT / f"probe-points-{tag}.png")
    overlay(img, merged, colors, REPO / f"probe-points-{tag}.png")
    return scores


def main():
    frame_path = Path("/tmp/frame160.jpg")
    if not frame_path.exists():
        raw = urllib.request.urlopen(API + f"/projects/{PID}/videos/{VID}/frames/{FRAME}", timeout=60).read()
        frame_path.write_bytes(raw)
    img, gt = build_gt(frame_path)
    print("GT centers: left", gt["left_noshave"]["cx"], gt["left_noshave"]["cy"],
          "right", gt["right_headshave"]["cx"], gt["right_headshave"]["cy"])

    reset_and_prep()
    text_scores = run_text(gt, img)

    lx, ly = gt["left_noshave"]["cx"], gt["left_noshave"]["cy"]
    rx, ry = gt["right_headshave"]["cx"], gt["right_headshave"]["cy"]
    point_scores = run_points(gt, img, "torso", [
        (NO, "NoShave", [[lx, ly]], [1]),
        (HEAD, "HeadShave", [[rx, ry]], [1]),
    ])
    extra = run_points(gt, img, "torso-plus-neg", [
        (NO, "NoShave", [[lx, ly], [rx, ry]], [1, 0]),
        (HEAD, "HeadShave", [[rx, ry], [lx, ly]], [1, 0]),
    ])

    summary = {
        "gt": {
            "left_noshave": {k: gt["left_noshave"][k] for k in ("cx", "cy", "area")},
            "right_headshave": {k: gt["right_headshave"][k] for k in ("cx", "cy", "area")},
        },
        "text": {k: v["scores"] for k, v in text_scores.items()},
        "points_torso": point_scores,
        "points_torso_neg": extra,
    }
    json.dump(summary, open(OUT / "probe-summary.json", "w"), indent=2)
    json.dump(summary, open(REPO / "probe-summary.json", "w"), indent=2)
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
