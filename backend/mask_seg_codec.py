"""
Lossless segmentation blob codec for masks.sqlite (v1: zlib + pickle + COCO RLE).
"""

from __future__ import annotations

import io
import pickle
import zlib
from typing import Any

import numpy as np
from pycocotools import mask as mask_util

MAGIC = b"SAM3WT01"
VERSION_U32 = 1


class _PlainDataUnpickler(pickle.Unpickler):
    """Unpickler that refuses every global lookup.

    Payloads only hold dicts/lists/str/bytes/ints, which pickle without any
    GLOBAL opcode. Rejecting find_class means a crafted masks.sqlite (e.g. from
    a project folder shared by someone else) cannot execute code on load.
    """

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"seg_blob may not reference {module}.{name}")


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


def _unpack_rle_payload(blob: bytes) -> dict:
    if len(blob) < len(MAGIC) + 4:
        raise ValueError("seg_blob too short")
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("invalid seg_blob magic")
    ver = int.from_bytes(blob[len(MAGIC) : len(MAGIC) + 4], "big")
    if ver != VERSION_U32:
        raise ValueError(f"unsupported seg_blob version {ver}")
    raw = zlib.decompress(blob[len(MAGIC) + 4 :])
    return _PlainDataUnpickler(io.BytesIO(raw)).load()


def _pack_rle_payload(payload: dict) -> bytes:
    blob = zlib.compress(pickle.dumps(payload, protocol=4))
    return MAGIC + VERSION_U32.to_bytes(4, "big") + blob


def swap_keys_in_masks_blob(blob: bytes, obj_a: str, obj_b: str) -> tuple[bytes, bool]:
    """Swap two object-id keys in the RLE dict without decoding masks to dense arrays."""
    payload = _unpack_rle_payload(blob)
    rles: dict[str, Any] = dict(payload.get("rles") or {})
    has_a = obj_a in rles
    has_b = obj_b in rles
    if not has_a and not has_b:
        return blob, False
    a_rle = rles.pop(obj_a, None)
    b_rle = rles.pop(obj_b, None)
    if b_rle is not None:
        rles[obj_a] = b_rle
    if a_rle is not None:
        rles[obj_b] = a_rle
    payload["rles"] = rles
    return _pack_rle_payload(payload), True


def decode_masks_blob(blob: bytes) -> dict[str, np.ndarray]:
    """Blob → dense bool masks keyed by obj id string."""
    payload = _unpack_rle_payload(blob)
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
