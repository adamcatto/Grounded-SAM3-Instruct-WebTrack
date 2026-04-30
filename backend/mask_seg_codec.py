"""
Lossless segmentation blob codec for masks.sqlite (v1: zlib + pickle + COCO RLE).
"""

from __future__ import annotations

import pickle
import zlib
from typing import Any

import numpy as np
from pycocotools import mask as mask_util

MAGIC = b"SAM3WT01"
VERSION_U32 = 1


def encode_masks_blob(masks: dict[str, np.ndarray]) -> bytes:
    """Dense uint8/bool masks keyed by obj id string → compressed blob."""
    if not masks:
        payload = {"h": 0, "w": 0, "rles": {}}
    else:
        first = np.squeeze(next(iter(masks.values())))
        h, w = int(first.shape[0]), int(first.shape[1])
        rles: dict[str, Any] = {}
        for oid, m in masks.items():
            arr = np.squeeze(m.astype(np.uint8))
            if arr.shape != (h, w):
                raise ValueError(f"mask shape mismatch for {oid!r}: {arr.shape} vs ({h},{w})")
            rles[str(oid)] = mask_util.encode(np.asfortranarray(arr))
        payload = {"h": h, "w": w, "rles": rles}
    blob = zlib.compress(pickle.dumps(payload, protocol=4))
    return MAGIC + VERSION_U32.to_bytes(4, "big") + blob


def decode_masks_blob(blob: bytes) -> dict[str, np.ndarray]:
    """Blob → dense bool masks keyed by obj id string."""
    if len(blob) < len(MAGIC) + 4:
        raise ValueError("seg_blob too short")
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("invalid seg_blob magic")
    ver = int.from_bytes(blob[len(MAGIC) : len(MAGIC) + 4], "big")
    if ver != VERSION_U32:
        raise ValueError(f"unsupported seg_blob version {ver}")
    raw = zlib.decompress(blob[len(MAGIC) + 4 :])
    payload = pickle.loads(raw)
    h: int = int(payload["h"])
    w: int = int(payload["w"])
    rles: dict[str, Any] = payload.get("rles") or {}
    if h == 0 and w == 0:
        return {}
    out: dict[str, np.ndarray] = {}
    for oid, rle in rles.items():
        m = mask_util.decode(rle)
        out[str(oid)] = (np.squeeze(m) > 0).astype(np.uint8)
    return out
