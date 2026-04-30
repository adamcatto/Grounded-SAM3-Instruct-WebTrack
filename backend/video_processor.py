"""
Video frame extraction, mask encoding, and output persistence.
"""

import base64
import io
import json
import logging
import shutil
import subprocess
import struct
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─── MP4 web-optimization ─────────────────────────────────────────────────────

def _needs_faststart(video_path: str) -> bool:
    """
    Check if an MP4 file has its moov atom at the end (not web-optimized).
    Returns True if moov comes after mdat, meaning faststart is needed.
    """
    try:
        with open(video_path, "rb") as f:
            pos = 0
            found_mdat = False
            for _ in range(50):  # scan up to 50 top-level atoms
                f.seek(pos)
                header = f.read(8)
                if len(header) < 8:
                    break
                size = struct.unpack(">I", header[:4])[0]
                atom_type = header[4:8]
                if size == 1:
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    size = struct.unpack(">Q", ext)[0]
                if size == 0:
                    break
                if atom_type == b"mdat":
                    found_mdat = True
                if atom_type == b"moov":
                    # moov found — if mdat was already seen, we need faststart
                    return found_mdat
                pos += size
    except Exception:
        pass
    # If we couldn't parse it, assume faststart is needed for safety
    return True


def ensure_faststart(video_path: str) -> str:
    """
    If the video is an MP4 without faststart (moov atom at the end),
    re-mux it with ffmpeg -movflags +faststart.  This is a copy (no
    re-encoding) and takes just a few seconds even for large files.

    Returns the path to the web-optimized file (may be the same path
    if no work was needed, or a new path if it was re-muxed in place).
    """
    path = Path(video_path)
    if path.suffix.lower() not in (".mp4", ".mov"):
        return video_path

    if not _needs_faststart(video_path):
        logger.info(f"Video already has faststart: {path.name}")
        return video_path

    logger.info(f"Applying faststart to {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")
    tmp_out = path.with_suffix(".faststart.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(path),
                "-c", "copy",                # no re-encoding
                "-movflags", "+faststart",    # move moov to front
                str(tmp_out),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for very large files
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg faststart failed: {result.stderr[-500:]}")
            tmp_out.unlink(missing_ok=True)
            return video_path  # fall back to original

        # Replace original with faststart version
        tmp_out.replace(path)
        logger.info(f"Faststart applied successfully: {path.name}")
        return str(path)
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg faststart timed out")
        tmp_out.unlink(missing_ok=True)
        return video_path
    except FileNotFoundError:
        logger.warning("ffmpeg not found — skipping faststart optimization")
        return video_path


# ─── Frame Extraction ─────────────────────────────────────────────────────────

def compute_ds_dims(
    w: int, h: int,
    max_dim: Optional[int] = None,
    scale_factor: Optional[float] = None,
) -> tuple[int, int]:
    """Return (new_w, new_h) after applying lazy-downsample parameters."""
    if scale_factor is not None and scale_factor > 1.0:
        return max(1, int(w / scale_factor)), max(1, int(h / scale_factor))
    if max_dim is not None and max_dim > 0 and max(w, h) > max_dim:
        if w >= h:
            return max_dim, max(1, int(round(h * max_dim / w)))
        else:
            return max(1, int(round(w * max_dim / h))), max_dim
    return w, h


def _ds_resize(
    frame: np.ndarray,
    max_dim: Optional[int] = None,
    scale_factor: Optional[float] = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    new_w, new_h = compute_ds_dims(w, h, max_dim, scale_factor)
    if (new_w, new_h) == (w, h):
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_video_info(video_path: str) -> dict:
    """Return num_frames, fps, width, height for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"num_frames": num_frames, "fps": fps, "width": width, "height": height}


def extract_frames(
    video_path: str, out_dir: str, progress_callback=None,
    max_dim: Optional[int] = None, scale_factor: Optional[float] = None,
) -> dict:
    """
    Extract all frames from a video as 000000.jpg, 000001.jpg, ...
    Returns {num_frames, fps, width, height}.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _ds_resize(frame, max_dim, scale_factor)
        out_file = out_path / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_idx += 1
        if progress_callback and frame_idx % 100 == 0:
            progress_callback(frame_idx, total)

    cap.release()
    eff_w, eff_h = compute_ds_dims(width, height, max_dim, scale_factor)
    return {"num_frames": frame_idx, "fps": fps, "width": eff_w, "height": eff_h}


def extract_preview_frames(
    video_path: str, out_dir: str, max_preview: int = 80,
    max_dim: Optional[int] = None, scale_factor: Optional[float] = None,
) -> dict:
    """
    Extract a small set of evenly-spaced preview frames for the video player.
    These are named by their real frame index (e.g. 000000.jpg, 000050.jpg, ...).
    Returns {num_frames (total), preview_count, preview_indices, fps, width, height}.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError("Video has 0 frames")

    # Compute evenly spaced indices
    if total <= max_preview:
        indices = list(range(total))
    else:
        step = total / max_preview
        indices = [int(round(i * step)) for i in range(max_preview)]
        # Ensure last frame is included and no duplicates
        if indices[-1] != total - 1:
            indices[-1] = total - 1
        indices = sorted(set(indices))

    extracted = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = _ds_resize(frame, max_dim, scale_factor)
        out_file = out_path / f"{idx:06d}.jpg"
        cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        extracted.append(idx)

    cap.release()
    eff_w, eff_h = compute_ds_dims(width, height, max_dim, scale_factor)
    return {
        "num_frames": total,
        "preview_count": len(extracted),
        "preview_indices": extracted,
        "fps": fps,
        "width": eff_w,
        "height": eff_h,
    }


def extract_frame_range(
    video_path: str, out_dir: str, start: int, end: int,
    progress_callback=None,
    max_dim: Optional[int] = None, scale_factor: Optional[float] = None,
) -> dict:
    """
    Extract frames [start, end) from a video.  Skips frames that already exist
    on disk.  Returns {extracted_count, start, end}.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end, total)
    if start >= end:
        cap.release()
        return {"extracted_count": 0, "start": start, "end": end}

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    extracted = 0
    for idx in range(start, end):
        out_file = out_path / f"{idx:06d}.jpg"
        if out_file.exists():
            # Already on disk – still need to advance the capture
            cap.read()
            continue
        ret, frame = cap.read()
        if not ret:
            break
        frame = _ds_resize(frame, max_dim, scale_factor)
        cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        extracted += 1
        if progress_callback and extracted % 10 == 0:
            progress_callback(extracted, end - start)

    cap.release()
    return {"extracted_count": extracted, "start": start, "end": end}


def extract_frame_indices(
    video_path: str,
    out_dir: str,
    indices: list[int],
    progress_callback=None,
    max_dim: Optional[int] = None,
    scale_factor: Optional[float] = None,
) -> dict:
    """
    Extract only the listed frame indices into out_dir as %06d.jpg.
    Skips indices whose files already exist. Indices may be non-contiguous.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    uniq = sorted(set(indices))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted = 0
    todo = sum(
        1
        for idx in uniq
        if idx < total and not (out_path / f"{idx:06d}.jpg").exists()
    )
    done_missing = 0
    for idx in uniq:
        if idx >= total:
            continue
        out_file = out_path / f"{idx:06d}.jpg"
        if out_file.exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = _ds_resize(frame, max_dim, scale_factor)
        cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        extracted += 1
        done_missing += 1
        if progress_callback and todo > 0 and done_missing % max(1, todo // 10) == 0:
            progress_callback(done_missing, todo)

    cap.release()
    return {"extracted_count": extracted, "indices_requested": uniq}


def get_frame_path(frames_dir: str, frame_idx: int) -> Path:
    return Path(frames_dir) / f"{frame_idx:06d}.jpg"


# ─── Mask Visualization ───────────────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def encode_mask_as_png(
    mask_np: np.ndarray,
    color_hex: str,
    alpha_fill: float = 0.45,
    border_thickness: Optional[int] = None,
) -> str:
    """
    Convert a binary (H, W) mask to a base64-encoded RGBA PNG.
    Visual style: semi-transparent fill + glowing bright border.

    border_thickness defaults to a value proportional to the mask resolution
    (roughly 3px at 1080p, 1px at ~360p) so the border looks consistent
    regardless of whether the video has been spatially downsampled.
    """
    H, W = mask_np.shape[:2]
    if border_thickness is None:
        # Scale with the shorter dimension: ~3px at 1080p, ~1px at 360p.
        border_thickness = max(1, round(min(H, W) / 360))
    r, g, b = hex_to_rgb(color_hex)

    # Create RGBA overlay
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Semi-transparent fill
    mask_bool = mask_np.astype(bool)
    rgba[mask_bool, 0] = r
    rgba[mask_bool, 1] = g
    rgba[mask_bool, 2] = b
    rgba[mask_bool, 3] = int(255 * alpha_fill)

    # Bright border (find contours, draw thick outline)
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw border on a separate layer
    border_layer = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(border_layer, contours, -1, 255, border_thickness)

    # Outer glow: dilate contour slightly, lower alpha
    kernel = np.ones((border_thickness + 2, border_thickness + 2), np.uint8)
    glow_layer = cv2.dilate(border_layer, kernel, iterations=1)
    glow_bool = glow_layer.astype(bool) & ~border_layer.astype(bool)
    rgba[glow_bool, 0] = min(255, r + 60)
    rgba[glow_bool, 1] = min(255, g + 60)
    rgba[glow_bool, 2] = min(255, b + 60)
    rgba[glow_bool, 3] = 80

    # Solid bright border
    border_bool = border_layer.astype(bool)
    rgba[border_bool, 0] = min(255, r + 80)
    rgba[border_bool, 1] = min(255, g + 80)
    rgba[border_bool, 2] = min(255, b + 80)
    rgba[border_bool, 3] = 255

    # Encode as PNG
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def composite_masks_as_png(
    masks: dict,  # {obj_id: binary_mask_HW}
    colors: dict,  # {obj_id: hex_color}
    frame_w: int,
    frame_h: int,
) -> str:
    """
    Composite all object masks into a single RGBA PNG for a frame.
    Returns base64-encoded PNG.
    """
    rgba = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    for obj_id, mask in masks.items():
        color_hex = colors.get(str(obj_id), "#5B8DD9")
        r, g, b = hex_to_rgb(color_hex)
        mask_bool = mask.astype(bool)
        # Fill
        rgba[mask_bool, 0] = r
        rgba[mask_bool, 1] = g
        rgba[mask_bool, 2] = b
        rgba[mask_bool, 3] = int(255 * 0.45)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def bbox_norm_xywh_score_from_mask(mask: np.ndarray) -> tuple[list[float], float]:
    """Normalized xywh bbox [0,1] + score from a binary mask (matches server semantics)."""
    m = np.squeeze(mask).astype(bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    h, w = m.shape[:2]
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    nw = max(1, w)
    nh = max(1, h)
    return [
        float(x0) / nw,
        float(y0) / nh,
        float(x1 - x0 + 1) / nw,
        float(y1 - y0 + 1) / nh,
    ], 1.0


# ─── Output Persistence ───────────────────────────────────────────────────────

def save_masks_npz(out_path: str, masks: dict):
    """
    Save {obj_id: binary_mask_HW (bool/uint8)} to compressed npz.
    Key format: 'obj_{obj_id}'
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **{f"obj_{k}": v.astype(np.uint8) for k, v in masks.items()})


def save_bboxes_json(out_path: str, bboxes: dict):
    """
    Save {obj_id: [x, y, w, h, score]} as json.
    Coordinates are normalized [0, 1].
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({str(k): v for k, v in bboxes.items()}, indent=2))


def load_masks_npz(npz_path: str) -> dict:
    """Load masks from npz. Returns {obj_id: binary_mask_HW}. Corrupt/missing reads as {}."""
    path = Path(npz_path)
    if not path.is_file():
        return {}
    try:
        with np.load(str(path)) as data:
            return {k.replace("obj_", ""): data[k].astype(bool) for k in data.files}
    except (OSError, zipfile.BadZipFile, EOFError, ValueError) as e:
        logger.warning("Unreadable mask npz (truncated or corrupt): %s — %s", path, e)
        return {}


def load_bboxes_json(json_path: str) -> dict:
    """Load bboxes from json. Returns {obj_id: [x, y, w, h, score]}."""
    return json.loads(Path(json_path).read_text())


# ─── Video Export ─────────────────────────────────────────────────────────────

def overlay_masks_on_frame(
    frame_bgr: np.ndarray,
    masks: dict,        # {obj_id_str: binary H×W uint8/bool}
    colors: dict,       # {obj_id_str: hex_color}
    labels: dict,       # {obj_id_str: label_text}
    alpha_fill: float = 0.4,
    border_thickness: int = 2,
) -> np.ndarray:
    """
    Composite object masks with labels onto a BGR video frame.
    Returns a new frame; does not modify the input.
    """
    out = frame_bgr.copy()
    H, W = out.shape[:2]
    overlay = out.copy()

    for obj_id_str, mask in masks.items():
        if mask is None:
            continue
        mask_bool = np.squeeze(np.asarray(mask)).astype(bool)
        if mask_bool.ndim != 2:
            continue
        if mask_bool.shape != (H, W):
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        r, g, b = hex_to_rgb(colors.get(str(obj_id_str), "#5B8DD9"))
        overlay[mask_bool] = (b, g, r)  # OpenCV BGR

    # Blend semi-transparent fill
    cv2.addWeighted(overlay, alpha_fill, out, 1.0 - alpha_fill, 0.0, out)

    # Draw borders and labels on top (fully opaque)
    for obj_id_str, mask in masks.items():
        if mask is None:
            continue
        mask_bool = np.squeeze(np.asarray(mask)).astype(bool)
        if mask_bool.ndim != 2:
            continue
        if mask_bool.shape != (H, W):
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        r, g, b = hex_to_rgb(colors.get(str(obj_id_str), "#5B8DD9"))
        bright = (min(255, b + 80), min(255, g + 80), min(255, r + 80))
        mask_u8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, bright, border_thickness)

        # Label at centroid
        label = labels.get(str(obj_id_str), f"Object {obj_id_str}")
        ys, xs = np.where(mask_bool)
        if len(ys) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.35, min(0.65, W / 1920.0))
        thick = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        tx = max(2, min(cx - tw // 2, W - tw - 4))
        ty = max(th + baseline + 4, min(cy, H - baseline - 4))
        pad = 3
        cv2.rectangle(
            out,
            (tx - pad, ty - th - pad),
            (tx + tw + pad, ty + baseline + pad),
            (20, 20, 20),
            -1,
        )
        cv2.putText(out, label, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    return out


def export_video_with_masks(
    source_path: str,
    out_path: str,
    video_dir: str,
    colors: dict,       # {obj_id_str: hex_color}
    labels: dict,       # {obj_id_str: label_text}
    num_frames_hint: int = 0,
    progress_callback=None,  # fn(frame_idx: int, total: int)
) -> dict:
    """
    Write source_path to out_path as H.264 MP4, overlaying masks from the video's
    mask storage (`masks.sqlite` and/or legacy masks/*.npz).

    Does NOT read or modify any SAM session state, annotated_frames, or
    inference data — only reads the source video and persisted masks.

    Returns {"total_frames": N, "fps": fps}.
    """
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open source video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or num_frames_hint or 1000

    from mask_store import VideoMaskStorage

    ms = VideoMaskStorage(video_dir)
    emit_every = max(30, total // 200)

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Drain ffmpeg stderr in a background thread to prevent the OS pipe
    # buffer (64 KB) from filling up and deadlocking the encode on long videos.
    import threading
    _stderr_chunks: list[bytes] = []
    def _drain_stderr():
        for chunk in iter(lambda: ffmpeg_proc.stderr.read(4096), b""):
            _stderr_chunks.append(chunk)
    _stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    _stderr_thread.start()

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                raw_masks = ms.load_masks_dense(frame_idx)
                if raw_masks:
                    frame = overlay_masks_on_frame(frame, raw_masks, colors, labels)
            except Exception as e:
                logger.warning(f"Mask overlay failed for frame {frame_idx}: {e}")
            ffmpeg_proc.stdin.write(frame.tobytes())
            frame_idx += 1
            if progress_callback and frame_idx % emit_every == 0:
                progress_callback(frame_idx, total)
    except Exception:
        cap.release()
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.wait(timeout=30)
        raise
    finally:
        cap.release()

    # Signal end of input and wait for ffmpeg to finish
    try:
        ffmpeg_proc.stdin.close()
    except Exception:
        pass
    ret_code = ffmpeg_proc.wait(timeout=300)
    _stderr_thread.join(timeout=10)
    if ret_code != 0:
        stderr_out = b"".join(_stderr_chunks).decode(errors="replace")
        raise RuntimeError(f"ffmpeg encoding failed (exit {ret_code}): {stderr_out[-500:]}")

    # Emit final progress at 100%
    if progress_callback:
        progress_callback(frame_idx, frame_idx or 1)

    return {"total_frames": frame_idx, "fps": fps}


# ─── Spatial Downsampling ─────────────────────────────────────────────────────

def downsample_video(
    src_path: str,
    dst_path: str,
    max_dim: Optional[int] = None,
    scale_factor: Optional[float] = None,
) -> dict:
    """
    Re-encode src_path → dst_path at a lower resolution.
    Exactly one of max_dim or scale_factor must be provided.
      max_dim     – longest side clamped to this pixel count
      scale_factor – each dimension divided by this value (e.g. 2.0 → half size)
    Uses libx264 with faststart. Returns {num_frames, fps, width, height}.
    Raises RuntimeError if ffmpeg fails.
    """
    if scale_factor is not None:
        # Divide both dimensions by scale_factor; round down to nearest even number
        vf = (
            f"scale=trunc(iw/{scale_factor}/2)*2:trunc(ih/{scale_factor}/2)*2"
        )
    elif max_dim is not None:
        # Scale so longer dimension ≤ max_dim, maintain AR, ensure even dims for libx264
        vf = (
            f"scale=w='min(iw,{max_dim})':h='min(ih,{max_dim})'"
            ":force_original_aspect_ratio=decrease"
            ",scale=trunc(iw/2)*2:trunc(ih/2)*2"
        )
    else:
        raise ValueError("Either max_dim or scale_factor must be provided")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src_path),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-movflags", "+faststart",
            str(dst_path),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg downsample failed: {result.stderr[-500:]}")
    info = get_video_info(str(dst_path))
    if info["num_frames"] <= 0 or info["width"] <= 0 or info["height"] <= 0:
        raise RuntimeError(
            f"ffmpeg produced a broken output (no frames/dimensions): {result.stderr[-300:]}"
        )
    return info


# ─── Thumbnail Generation ──────────────────────────────────────────────────────

def generate_thumbnail(frame_path: str, height: int = 60) -> Optional[bytes]:
    """Generate a small JPEG thumbnail for the frame strip."""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    new_w = int(w * height / h)
    thumb = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return bytes(buf)
