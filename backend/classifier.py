"""
Per-pixel object identity classifier using a DINOv2 backbone.

Architecture
------------
  DINOv2-small (frozen) → patch features [B, D, n_h, n_w]
  → Conv2d(D, 256, 3×3) + BN + GELU
  → Conv2d(256, N+1, 1×1)          N objects + 1 background class
  → bilinear upsample to INPUT_SIZE

Training
--------
  Backbone features are pre-extracted once (no grad) so training epochs
  only update the lightweight head — usually fast even on CPU.

  Loss: per-pixel cross-entropy supervised by SAM3 propagated masks.
  Background class = pixels covered by no mask.

Inference
---------
  Per-mask average of object class probabilities → greedy argmax assignment.
"""

import logging
import threading
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mask_store import VideoMaskStorage

logger = logging.getLogger(__name__)

BACKBONE_NAME = "facebook/dinov2-small"
PATCH_SIZE    = 14
INPUT_SIZE    = 448   # must be a multiple of PATCH_SIZE (32 × 14)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Per-(project, video) cancellation signals
_cancel_flags: dict[tuple[str, str], threading.Event] = {}


# ── Model ─────────────────────────────────────────────────────────────────────

class ObjectClassifier(nn.Module):
    """DINOv2-backed per-pixel classifier with N+1 output channels."""

    def __init__(self, n_objects: int) -> None:
        super().__init__()
        # Lazy import so the server starts without pulling in transformers
        from transformers import Dinov2Model
        self.backbone = Dinov2Model.from_pretrained(BACKBONE_NAME)
        hidden = self.backbone.config.hidden_size   # 384 for dinov2-small
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, n_objects + 1, kernel_size=1),  # 0=bg, 1..N=objects
        )
        # Freeze backbone — only the head is trained
        for p in self.backbone.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run frozen backbone, reshape patch tokens to spatial grid.
        Args:
            pixel_values: [B, 3, H, W]  (must be INPUT_SIZE × INPUT_SIZE)
        Returns:
            [B, D, n_h, n_w]  where n_h = H // PATCH_SIZE
        """
        B, _, H, W = pixel_values.shape
        with torch.no_grad():
            out = self.backbone(pixel_values=pixel_values)
        patch = out.last_hidden_state[:, 1:]        # drop CLS  [B, N, D]
        n_h, n_w = H // PATCH_SIZE, W // PATCH_SIZE
        feat = patch.reshape(B, n_h, n_w, -1).permute(0, 3, 1, 2).contiguous()
        return feat                                 # [B, D, n_h, n_w]

    def head_forward(
        self, feat: torch.Tensor, out_hw: tuple[int, int]
    ) -> torch.Tensor:
        """Apply head and upsample."""
        logits = self.head(feat)
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(pixel_values)
        return self.head_forward(feat, (INPUT_SIZE, INPUT_SIZE))


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxWx3 → normalized float tensor [1, 3, INPUT_SIZE, INPUT_SIZE]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rs  = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = img_rs.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # [1, 3, H, W]


def _build_label(npz: dict[str, np.ndarray], obj_ids: list[str]) -> torch.Tensor:
    """
    Convert {obj_id: binary_mask} to [1, H, W] int64 label tensor.
    Background = 0; object at index i → class i+1 (1-based, ordered by obj_ids).
    """
    label = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.int64)
    for idx, obj_id in enumerate(obj_ids, start=1):
        mask = npz.get(obj_id)
        if mask is None:
            continue
        m_rs = cv2.resize(
            mask.astype(np.uint8), (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        label[m_rs > 0] = idx
    return torch.from_numpy(label).unsqueeze(0)   # [1, H, W]


def _load_frame(
    fidx: int,
    frames_dir: Path,
    cap: "cv2.VideoCapture | None",
) -> "np.ndarray | None":
    """
    Load a single video frame as BGR.
    Prefers a pre-extracted JPEG from frames_dir; falls back to VideoCapture seek.
    """
    jpg = frames_dir / f"{fidx:06d}.jpg"
    if jpg.exists():
        img = cv2.imread(str(jpg))
        if img is not None:
            return img
    if cap is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fidx))
        ret, frame = cap.read()
        if ret:
            return frame
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def cancel(pid: str, vid: str) -> None:
    """Signal a running training job to stop."""
    ev = _cancel_flags.get((pid, vid))
    if ev:
        ev.set()


def train_and_infer(
    pid: str,
    vid: str,
    frames_dir: Path,
    video_dir: Path,
    video_source: str,
    frame_range: range,
    obj_ids: list[str],
    epochs: int,
    lr: float,
    train_ratio: float = 0.75,
    eval_ratio: float = 0.20,
) -> Iterator[dict]:
    """
    Main entry point.  Yields SSE-style event dicts with a 'type' key.
    Designed to run in a background thread; the caller forwards events
    through an asyncio Queue to the SSE response generator.

    Event types
    -----------
    split             – train/eval frame lists
    status            – human-readable progress string
    epoch             – per-epoch loss & accuracy
    inference_progress– inference progress (done / total)
    done              – final results payload
    cancelled         – training was cancelled
    error             – unrecoverable error with message
    """
    cancel_ev = threading.Event()
    _cancel_flags[(pid, vid)] = cancel_ev
    try:
        yield from _run(
            cancel_ev, frames_dir, video_dir, video_source,
            list(frame_range), obj_ids, epochs, lr, train_ratio, eval_ratio,
        )
    finally:
        _cancel_flags.pop((pid, vid), None)


# ── Internal implementation ───────────────────────────────────────────────────

def _run(
    cancel_ev: threading.Event,
    frames_dir: Path,
    video_dir: Path,
    video_source: str,
    frame_indices: list[int],
    obj_ids: list[str],
    epochs: int,
    lr: float,
    train_ratio: float,
    eval_ratio: float,
) -> Iterator[dict]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Classifier: training on {device}, backbone={BACKBONE_NAME}")

    ms = VideoMaskStorage(video_dir)

    # ── 1. Find frames that have saved masks in the requested range ────────────
    frames_with_masks = [
        f for f in frame_indices
        if ms.has_masks(f)
    ]
    if len(frames_with_masks) < 4:
        yield {
            "type": "error",
            "message": (
                f"Only {len(frames_with_masks)} frames with saved masks in the selected range. "
                "Run propagation first."
            ),
        }
        return

    n        = len(frames_with_masks)
    n_train  = max(1, int(n * train_ratio))
    n_skip   = max(0, int(n * (1.0 - train_ratio - eval_ratio)))
    train_frames = frames_with_masks[:n_train]
    eval_frames  = frames_with_masks[n_train + n_skip:]

    yield {"type": "split", "train_frames": train_frames, "eval_frames": eval_frames}

    # ── 2. Load training frames ────────────────────────────────────────────────
    yield {"type": "status", "message": "Loading training frames…"}
    cap = cv2.VideoCapture(str(video_source)) if video_source else None

    train_data: list[tuple[torch.Tensor, torch.Tensor]] = []
    for fidx in train_frames:
        if cancel_ev.is_set():
            if cap: cap.release()
            yield {"type": "cancelled"}
            return
        img = _load_frame(fidx, frames_dir, cap)
        if img is None:
            logger.warning(f"Classifier: cannot load frame {fidx}")
            continue
        masks_dict = ms.load_masks_dense(fidx)
        label = _build_label(masks_dict, obj_ids)
        if int(label.max()) == 0:
            continue   # no object pixels in this frame — skip
        train_data.append((_preprocess(img), label))

    if cap:
        cap.release()

    if not train_data:
        yield {
            "type": "error",
            "message": (
                "No usable training frames. "
                "Frames must have been viewed (so they exist in frames/) "
                "and have propagated masks."
            ),
        }
        return

    yield {
        "type": "status",
        "message": (
            f"Loaded {len(train_data)} training frames — "
            "initialising model (may download DINOv2 on first run)…"
        ),
    }

    # ── 3. Build model ─────────────────────────────────────────────────────────
    try:
        model = ObjectClassifier(n_objects=len(obj_ids)).to(device)
    except Exception as exc:
        yield {"type": "error", "message": f"Model init failed: {exc}"}
        return

    # ── 4. Pre-extract backbone features (once, no grad) ──────────────────────
    yield {"type": "status", "message": "Extracting backbone features…"}
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for img_t, lbl_t in train_data:
        if cancel_ev.is_set():
            yield {"type": "cancelled"}
            return
        feat = model.extract_features(img_t.to(device))  # [1, D, n_h, n_w]
        cached.append((feat, lbl_t.to(device)))

    # ── 5. Train the head ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    model.head.train()

    for epoch in range(epochs):
        if cancel_ev.is_set():
            yield {"type": "cancelled"}
            return
        ep_loss = ep_correct = ep_total = 0
        for feat, lbl_t in cached:
            optimizer.zero_grad()
            logits = model.head_forward(feat, (INPUT_SIZE, INPUT_SIZE))
            loss = F.cross_entropy(logits, lbl_t)
            loss.backward()
            optimizer.step()
            ep_loss   += loss.item()
            preds      = logits.argmax(1)
            ep_correct += int((preds == lbl_t).sum().item())
            ep_total   += int(lbl_t.numel())
        scheduler.step()
        yield {
            "type": "epoch",
            "epoch": epoch + 1,
            "epochs": epochs,
            "loss": round(ep_loss / max(len(cached), 1), 4),
            "train_acc": round(ep_correct / max(ep_total, 1), 4),
        }

    # ── 6. Inference on train + eval frames ───────────────────────────────────
    yield {"type": "status", "message": "Running per-frame inference…"}
    model.eval()
    frame_assignments: dict[int, dict] = {}
    all_infer = sorted(set(train_frames) | set(eval_frames))

    cap2 = cv2.VideoCapture(str(video_source)) if video_source else None
    with torch.no_grad():
        for i, fidx in enumerate(all_infer):
            if cancel_ev.is_set():
                if cap2: cap2.release()
                yield {"type": "cancelled"}
                return

            img = _load_frame(fidx, frames_dir, cap2)
            if img is None:
                continue
            masks_dict = ms.load_masks_dense(fidx)

            img_t    = _preprocess(img).to(device)
            logits   = model(img_t)[0]           # [N+1, H, W]
            probs    = F.softmax(logits, dim=0)  # [N+1, H, W]
            obj_prob = probs[1:]                 # [N, H, W] skip background

            frame_asgn: dict[str, dict] = {}
            for sam_id in obj_ids:
                mask_np = masks_dict.get(sam_id)
                if mask_np is None:
                    continue
                m_rs = cv2.resize(
                    mask_np.astype(np.uint8), (INPUT_SIZE, INPUT_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_bool = torch.from_numpy(m_rs > 0).to(device)
                if not mask_bool.any():
                    continue
                # Average class probs within this mask region
                avg = obj_prob[:, mask_bool].mean(dim=1)   # [N]
                pred_idx  = int(avg.argmax().item())
                frame_asgn[sam_id] = {
                    "predictedClass": obj_ids[pred_idx],
                    "confidence":     round(float(avg[pred_idx]), 4),
                    "scores":         {
                        oid: round(float(avg[j]), 4)
                        for j, oid in enumerate(obj_ids)
                    },
                }
            if frame_asgn:
                frame_assignments[fidx] = frame_asgn

            yield {"type": "inference_progress", "done": i + 1, "total": len(all_infer)}

    if cap2:
        cap2.release()

    # ── 7. Compute metrics ─────────────────────────────────────────────────────
    def _agreement(flist: list[int]) -> tuple[float, int]:
        correct = total = 0
        for f in flist:
            for sam_id, a in frame_assignments.get(f, {}).items():
                if a["predictedClass"] == sam_id:
                    correct += 1
                total += 1
        return round(correct / max(total, 1), 4), total

    train_acc, _ = _agreement(train_frames)
    eval_agr,  _ = _agreement(eval_frames)

    yield {
        "type": "done",
        "object_ids":     obj_ids,
        "frame_assignments": {str(k): v for k, v in frame_assignments.items()},
        "train_frames":   train_frames,
        "eval_frames":    eval_frames,
        "train_accuracy": train_acc,
        "eval_agreement": eval_agr,
    }
