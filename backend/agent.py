"""
Agentic text-prompting loop for SAM3 Web Tracker.

The agent sees project/video/frame context, calls tools (load frames, SAM3
text or point prompts, propagation, nearby-frame fallback), and streams
reasoning + action events over SSE so the UI can render them live.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 36
MAX_HISTORY_TURNS = 16
INSPECT_MAX_EDGE = 640
DEFAULT_NEARBY_OFFSET = 20
CONFIDENCE_RETRY_THRESHOLD = 0.45
# Inspect JPEGs stay in the chat until pruned. Keep the last two so the model can
# compare the current frame with the previous one without blowing the context.
KEEP_INSPECT_IMAGES = 2
# Tool calls are short; 4096 reserved tokens would eat half of an 8k window.
AGENT_MAX_TOKENS = 2048


# ─── LLM configuration (local Ollama / vLLM first, same idea as SAM 3 Agent) ──

# SAM 3 Agent serves Qwen-VL via vLLM at :8001 with a dummy OpenAI key.
# Ollama exposes the same /v1/chat/completions surface on :11434.
#
# Hardware profiles (AGENT_LLM_PROFILE / `scripts/serve_agent_llm.sh --profile`):
#   a100         1× 80GB A100 dedicated to the LLM → Qwen3-VL-32B-Instruct
#   h100x4       4× 80GB H100 NVL → Qwen2.5-VL-72B-Instruct (tensor parallel 4)
#   a100-shared  same 80GB GPU as SAM3 → Qwen3-VL-8B (leave VRAM for tracking)
#   demo         SAM 3 Agent notebook default (8B Thinking)
VLLM_DEFAULT_BASE = "http://127.0.0.1:8001/v1"
OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434/v1"
OLLAMA_NATIVE_BASE = "http://127.0.0.1:11434"
SAM3_AGENT_DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
VLLM_DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
OLLAMA_DEFAULT_MODEL = "qwen2.5vl:32b"
LOCAL_DUMMY_KEY = "DUMMY_API_KEY"

LLM_PROFILES: dict[str, dict[str, Any]] = {
    "a100": {
        "vllm_model": "Qwen/Qwen3-VL-32B-Instruct",
        "vllm_thinking_model": "Qwen/Qwen3-VL-32B-Thinking",
        "ollama_model": "qwen2.5vl:32b",
        "tensor_parallel": 1,
        "blurb": "1× 80GB A100 dedicated to the vision LLM (do not co-locate with SAM3)",
    },
    "h100x4": {
        "vllm_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "vllm_thinking_model": "Qwen/Qwen3-VL-32B-Thinking",
        "ollama_model": "qwen2.5vl:72b",
        "tensor_parallel": 4,
        "blurb": "4× 80GB H100 NVL; 72B bf16 at TP=4. Point the tracker at this server.",
    },
    "a100-shared": {
        "vllm_model": "Qwen/Qwen3-VL-8B-Instruct",
        "vllm_thinking_model": "Qwen/Qwen3-VL-8B-Thinking",
        "ollama_model": "qwen2.5vl",
        "tensor_parallel": 1,
        "blurb": "Same 80GB GPU as SAM3 — 8B VL so tracking still fits",
    },
    "demo": {
        "vllm_model": SAM3_AGENT_DEFAULT_MODEL,
        "vllm_thinking_model": SAM3_AGENT_DEFAULT_MODEL,
        "ollama_model": "qwen2.5vl",
        "tensor_parallel": 1,
        "blurb": "SAM 3 Agent notebook default (Qwen3-VL-8B-Thinking)",
    },
}
_PROFILE_ALIASES = {
    "workstation": "a100",
    "a100-80": "a100",
    "80gb": "a100",
    "cluster": "h100x4",
    "h100": "h100x4",
    "4xh100": "h100x4",
    "h100nvl": "h100x4",
    "shared": "a100-shared",
    "sam3": "demo",
    "8b": "demo",
}
VISION_MODEL_HINTS = (
    "vl", "vision", "llava", "minicpm", "pixtral", "gemma3", "qwen2.5-vl",
    "qwen2.5vl", "qwen3-vl", "qwen3_vl", "internvl", "phi-4-multimodal",
)

LOCAL_PROVIDERS = frozenset({"ollama", "vllm", "openai_compatible", "local", "lmstudio"})


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str]
    configured: bool
    missing_reason: str = ""
    reachable: bool = False
    available_models: list[str] = field(default_factory=list)
    local: bool = False


def _env_true(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def normalize_llm_profile(raw: Optional[str] = None) -> str:
    name = (raw if raw is not None else os.environ.get("AGENT_LLM_PROFILE") or "").strip().lower()
    name = _PROFILE_ALIASES.get(name, name)
    if not name:
        return "a100"
    if name not in LLM_PROFILES:
        return "a100"
    return name


def default_vllm_model(profile: Optional[str] = None, thinking: Optional[bool] = None) -> str:
    spec = LLM_PROFILES[normalize_llm_profile(profile)]
    think = _env_true("AGENT_LLM_THINKING", False) if thinking is None else bool(thinking)
    return str(spec["vllm_thinking_model"] if think else spec["vllm_model"])


def default_ollama_model(profile: Optional[str] = None) -> str:
    return str(LLM_PROFILES[normalize_llm_profile(profile)]["ollama_model"])


def _serve_hint() -> str:
    profile = normalize_llm_profile()
    model = default_vllm_model()
    return (
        f"bash scripts/serve_agent_llm.sh vllm --profile {profile}  "
        f"(serves {model} on :8001). "
        "Profiles: a100 (32B on dedicated 80GB), h100x4 (72B TP=4), "
        "a100-shared / demo (8B, can share a GPU with SAM3)."
    )


def _normalize_openai_base(url: str, provider: str = "") -> str:
    u = (url or "").strip().rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")].rstrip("/")
    if u.endswith("/v1"):
        return u
    if provider in ("ollama", "vllm", "openai_compatible", "local", "lmstudio") or _looks_local_url(u):
        return u + "/v1"
    return u


def _looks_local_url(url: Optional[str]) -> bool:
    if not url:
        return False
    low = url.lower()
    return any(h in low for h in ("127.0.0.1", "localhost", "0.0.0.0", "::1"))


def _is_local_provider(provider: str, base_url: Optional[str] = None) -> bool:
    if provider in LOCAL_PROVIDERS:
        return True
    return _looks_local_url(base_url)


def _prefer_vision_model(names: list[str], fallback: str = "") -> str:
    if not names:
        return fallback
    lowered = [(n, n.lower()) for n in names]
    for hint in VISION_MODEL_HINTS:
        for orig, low in lowered:
            if hint in low:
                return orig
    return names[0]


def _http_get_json(url: str, timeout: float = 1.2) -> Optional[dict]:
    try:
        import httpx
        r = httpx.get(url, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r.json()
    except Exception:
        try:
            import urllib.request
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None


def _models_from_openai_list(payload: Optional[dict]) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") or payload.get("models") or []
    names: list[str] = []
    for item in data:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("id") or item.get("name") or item.get("model")
            if name:
                names.append(str(name))
    return names


def probe_openai_compatible(base_url: str) -> list[str]:
    base = _normalize_openai_base(base_url)
    payload = _http_get_json(base + "/models")
    return _models_from_openai_list(payload)


def probe_ollama(native_base: str = OLLAMA_NATIVE_BASE) -> list[str]:
    payload = _http_get_json(native_base.rstrip("/") + "/api/tags")
    names: list[str] = []
    if isinstance(payload, dict):
        for item in payload.get("models") or []:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
            elif isinstance(item, str):
                names.append(item)
    if names:
        return names
    return probe_openai_compatible(native_base.rstrip("/") + "/v1")


def discover_local_llm() -> Optional[dict]:
    """Find a running Ollama or vLLM OpenAI-compatible server (SAM 3 Agent style)."""
    explicit = (os.environ.get("AGENT_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip()
    provider_hint = (os.environ.get("AGENT_LLM_PROVIDER") or "").strip().lower()

    if explicit:
        models = probe_openai_compatible(explicit)
        prov = provider_hint if provider_hint in LOCAL_PROVIDERS else (
            "ollama" if "11434" in explicit else "vllm" if "8001" in explicit else "openai_compatible"
        )
        if models or _looks_local_url(explicit):
            return {
                "provider": prov,
                "base_url": _normalize_openai_base(explicit, prov),
                "models": models,
                "reachable": bool(models),
            }

    order = ["vllm", "ollama"]
    if provider_hint == "ollama":
        order = ["ollama", "vllm"]
    elif provider_hint == "vllm":
        order = ["vllm", "ollama"]

    for prov in order:
        if prov == "vllm":
            models = probe_openai_compatible(VLLM_DEFAULT_BASE)
            if models:
                return {"provider": "vllm", "base_url": VLLM_DEFAULT_BASE, "models": models, "reachable": True}
        elif prov == "ollama":
            models = probe_ollama()
            if models:
                return {
                    "provider": "ollama",
                    "base_url": OLLAMA_DEFAULT_BASE,
                    "models": models,
                    "reachable": True,
                }
    return None


def load_llm_config(*, probe: bool = True) -> LLMConfig:
    provider = (os.environ.get("AGENT_LLM_PROVIDER") or "").strip().lower()
    api_key = (
        os.environ.get("AGENT_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ""
    ).strip()
    model = (os.environ.get("AGENT_LLM_MODEL") or "").strip()
    base_url = (os.environ.get("AGENT_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip() or None

    if provider in ("openai-compatible", "compatible", "azure"):
        provider = "openai_compatible"
    if provider == "lm-studio":
        provider = "lmstudio"

    discovered = None
    if probe and _env_true("AGENT_LLM_AUTODETECT", True) and provider not in ("anthropic",):
        if provider in ("", "auto", "local") or _is_local_provider(provider, base_url):
            discovered = discover_local_llm()

    if not provider or provider in ("auto", "local"):
        if discovered:
            provider = discovered["provider"]
        elif (os.environ.get("ANTHROPIC_API_KEY") or "").strip() and not (
            os.environ.get("OPENAI_API_KEY") or ""
        ).strip():
            provider = "anthropic"
        elif api_key and not _looks_local_url(base_url):
            provider = "openai"
        else:
            provider = "vllm"

    if provider not in ("openai", "anthropic", "openai_compatible", "ollama", "vllm", "lmstudio"):
        provider = "openai_compatible" if base_url else "openai"

    local = _is_local_provider(provider, base_url)
    available: list[str] = list((discovered or {}).get("models") or [])
    reachable = bool((discovered or {}).get("reachable"))

    if local:
        if not base_url:
            base_url = (discovered or {}).get("base_url") or (
                OLLAMA_DEFAULT_BASE if provider == "ollama" else VLLM_DEFAULT_BASE
            )
        else:
            base_url = _normalize_openai_base(base_url, provider)
        if not model:
            default_model = default_ollama_model() if provider == "ollama" else default_vllm_model()
            model = _prefer_vision_model(available, default_model)
        if not api_key:
            api_key = LOCAL_DUMMY_KEY
        if not reachable and probe:
            available = probe_openai_compatible(base_url) if provider != "ollama" else (
                probe_ollama() or probe_openai_compatible(base_url)
            )
            reachable = bool(available)
        configured = True
        missing = ""
        if probe and not reachable:
            missing = (
                f"No {provider} server answered at {base_url}. "
                f"{_serve_hint()} "
                "Or Ollama: `ollama pull qwen2.5vl:32b && ollama serve`. "
                "Same OpenAI-compat pattern as SAM 3 Agent (dummy API key)."
            )
            configured = False
        return LLMConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            configured=configured,
            missing_reason=missing,
            reachable=reachable,
            available_models=available,
            local=True,
        )

    if not model:
        model = "claude-sonnet-4-20250514" if provider == "anthropic" else "gpt-4o"

    if not api_key:
        return LLMConfig(
            provider=provider,
            api_key="",
            model=model,
            base_url=base_url,
            configured=False,
            missing_reason=(
                "No local Ollama/vLLM server detected and no cloud API key. "
                f"{_serve_hint()} "
                "Or set AGENT_LLM_BASE_URL / AGENT_LLM_PROVIDER=ollama|vllm. "
                "Cloud fallback: AGENT_LLM_API_KEY or OPENAI_API_KEY / ANTHROPIC_API_KEY."
            ),
            reachable=False,
            local=False,
        )
    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        configured=True,
        reachable=True,
        local=False,
    )


def llm_status_dict() -> dict:
    cfg = load_llm_config()
    profile = normalize_llm_profile()
    return {
        "configured": cfg.configured,
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "missing_reason": cfg.missing_reason,
        "reachable": cfg.reachable,
        "local": cfg.local,
        "available_models": cfg.available_models,
        "profile": profile,
        "recommended_model": default_vllm_model(profile),
        "profile_blurb": LLM_PROFILES[profile]["blurb"],
    }


# ─── Pure planning / evaluation helpers ───────────────────────────────────────


def plan_sample_frames(
    start_frame: int,
    num_frames: int,
    interval: int,
    include_last: bool = True,
) -> list[int]:
    """Every `interval` frames from start through the video (optionally include last)."""
    if num_frames <= 0:
        return []
    start = max(0, int(start_frame))
    last = num_frames - 1
    step = max(1, int(interval))
    frames: list[int] = []
    f = start
    while f <= last:
        frames.append(f)
        f += step
    if include_last and frames and frames[-1] != last:
        frames.append(last)
    if not frames:
        frames.append(min(start, last))
    return frames


def nearby_frames(
    frame_idx: int,
    num_frames: int,
    offset: int = DEFAULT_NEARBY_OFFSET,
    extras: Optional[list[int]] = None,
) -> list[int]:
    """Candidate frames to try when the current one is unclear."""
    last = max(0, num_frames - 1)
    off = max(1, int(offset))
    raw = [frame_idx + off, frame_idx - off, frame_idx + 2 * off, frame_idx - 2 * off]
    if extras:
        raw.extend(extras)
    out: list[int] = []
    seen = {frame_idx}
    for f in raw:
        f = max(0, min(last, int(f)))
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def evaluate_mask_quality(
    mask,
    score: Optional[float] = None,
    frame_area: Optional[int] = None,
) -> dict:
    """Heuristic confidence for a binary mask (area, compactness, model score)."""
    m = np.squeeze(np.asarray(mask))
    if m.ndim != 2:
        return {
            "ok": False,
            "confidence": 0.0,
            "reason": "invalid_mask_shape",
            "area_px": 0,
            "coverage": 0.0,
        }
    binary = m.astype(bool)
    area = int(binary.sum())
    h, w = binary.shape
    total = int(h * w) if frame_area is None else max(1, int(frame_area))
    coverage = area / max(1, total)
    if area == 0:
        return {
            "ok": False,
            "confidence": 0.0,
            "reason": "empty_mask",
            "area_px": 0,
            "coverage": 0.0,
            "bbox_xywh_norm": [0, 0, 0, 0],
            "retry_nearby": True,
        }

    ys, xs = np.where(binary)
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    bbox_area = max(1, bw * bh)
    fill = area / bbox_area
    bbox = [
        float(xs.min()) / w,
        float(ys.min()) / h,
        float(bw) / w,
        float(bh) / h,
    ]

    reasons: list[str] = []
    conf = 0.72
    if score is not None:
        conf = 0.35 * conf + 0.65 * float(np.clip(score, 0.0, 1.0))
    if coverage < 0.0004:
        conf *= 0.35
        reasons.append("mask_very_small")
    elif coverage < 0.0015:
        conf *= 0.7
        reasons.append("mask_small")
    if coverage > 0.55:
        conf *= 0.45
        reasons.append("mask_covers_most_of_frame")
    if fill < 0.12:
        conf *= 0.65
        reasons.append("fragmented_or_thin")
    if bbox[2] < 0.01 or bbox[3] < 0.01:
        conf *= 0.4
        reasons.append("tiny_bbox")

    conf = float(np.clip(conf, 0.0, 1.0))
    ok = conf >= CONFIDENCE_RETRY_THRESHOLD and "empty_mask" not in reasons
    return {
        "ok": ok,
        "confidence": round(conf, 4),
        "reason": ",".join(reasons) if reasons else "ok",
        "area_px": area,
        "coverage": round(coverage, 6),
        "bbox_xywh_norm": [round(v, 5) for v in bbox],
        "bbox_fill": round(fill, 4),
        "model_score": None if score is None else round(float(score), 4),
        "retry_nearby": (not ok) or conf < CONFIDENCE_RETRY_THRESHOLD,
    }


SAME_LOCATION_TOL = 0.08


def _compact_label(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("_", "").replace("-", "")


def _bbox_center_xy(det: dict) -> tuple[float, float]:
    bbox = det.get("bbox") or det.get("bbox_xywh_norm")
    if bbox is not None:
        try:
            x, y, w, h = [float(v) for v in list(bbox)[:4]]
            return (x + w / 2.0, y + h / 2.0)
        except (TypeError, ValueError):
            pass
    mask = det.get("mask")
    if mask is None:
        return (0.5, 0.5)
    m = np.squeeze(np.asarray(mask))
    if m.ndim != 2:
        return (0.5, 0.5)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return (0.5, 0.5)
    h, w = m.shape
    return (float(xs.mean()) / w, float(ys.mean()) / h)


def _prefer_side_from_text_and_object(text: str, obj: Optional[dict]) -> Optional[str]:
    """Map shave-state / side language onto leftmost vs rightmost detections."""
    name = _compact_label((obj or {}).get("name") or "")
    desc = (obj or {}).get("description") or ""
    blob = f"{text or ''} {desc}".lower()
    compact_blob = _compact_label(blob)
    if any(k in name for k in ("noshave", "unshaved", "unclipped")):
        return "left"
    if any(k in name for k in ("headshave", "backshave")):
        return "right"
    left_phrases = (
        "on the left", "left mouse", "left side", "left of the cage",
        "unshaved", "unclipped", "no shave", "noshave", "fully furred",
    )
    right_phrases = (
        "on the right", "right mouse", "right side", "far right",
        "water bottle", "water port", "clipped patch", "head shave", "shaved",
    )
    if any(p in blob or _compact_label(p) in compact_blob for p in left_phrases):
        return "left"
    if any(p in blob or _compact_label(p) in compact_blob for p in right_phrases):
        return "right"
    return None


def _text_implies_far_right(text: str, obj: Optional[dict]) -> bool:
    blob = f"{text or ''} {(obj or {}).get('description') or ''}".lower()
    return any(k in blob for k in ("far right", "water bottle", "water port", "right wall", "right of the cage"))


def _pick_detection(detections: list[dict], text: str, obj: Optional[dict]) -> dict:
    if not detections:
        raise ValueError("no detections")
    if len(detections) == 1:
        return detections[0]
    side = _prefer_side_from_text_and_object(text, obj)
    ranked = sorted(detections, key=lambda d: _bbox_center_xy(d)[0])
    if side == "left":
        return ranked[0]
    if side == "right":
        return ranked[-1]
    return detections[0]


def _detection_summary(det: dict) -> dict:
    cx, cy = _bbox_center_xy(det)
    bbox = det.get("bbox") or det.get("bbox_xywh_norm")
    bbox_out = None
    if bbox is not None:
        try:
            bbox_out = [round(float(x), 5) for x in list(bbox)[:4]]
        except (TypeError, ValueError):
            bbox_out = None
    score = det.get("score")
    try:
        score = None if score is None else round(float(score), 4)
    except (TypeError, ValueError):
        score = None
    return {
        "score": score,
        "bbox_xywh_norm": bbox_out,
        "center_xy": [round(cx, 4), round(cy, 4)],
    }


def finalize_segmentation_eval(objects: dict, per_obj: list[dict]) -> dict:
    """Fill in missing objects and flag two masks sitting on the same spot."""
    out = [dict(p) for p in per_obj]
    seen = {str(p.get("object_id")) for p in out}
    expected = len(objects or {})
    for oid, obj in (objects or {}).items():
        if str(oid) in seen:
            continue
        out.append({
            "ok": False,
            "confidence": 0.0,
            "reason": "missing_mask",
            "area_px": 0,
            "coverage": 0.0,
            "object_id": str(oid),
            "name": (obj or {}).get("name"),
            "overlap_px": 0,
            "retry_nearby": False,
        })
    with_bbox = [p for p in out if p.get("area_px") and p.get("bbox_xywh_norm")]
    for i, a in enumerate(with_bbox):
        for b in with_bbox[i + 1:]:
            ba = a["bbox_xywh_norm"]
            bb = b["bbox_xywh_norm"]
            ca = (ba[0] + ba[2] / 2.0, ba[1] + ba[3] / 2.0)
            cb = (bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0)
            if abs(ca[0] - cb[0]) < SAME_LOCATION_TOL and abs(ca[1] - cb[1]) < SAME_LOCATION_TOL:
                extra = "same_location_as_other_object"
                for p in (a, b):
                    p["ok"] = False
                    reason = p.get("reason") or "ok"
                    if extra not in reason:
                        p["reason"] = extra if reason in ("ok", "") else f"{reason},{extra}"
    masked = sum(1 for p in out if (p.get("area_px") or 0) > 0)
    incomplete = expected > 0 and masked < expected
    any_retry = any(p.get("retry_nearby") for p in out)
    return {
        "objects": out,
        "expected_object_count": expected,
        "masked_object_count": masked,
        "incomplete": incomplete,
        "ok": bool(out) and all(p.get("ok") for p in out) and not incomplete,
        "retry_nearby": any_retry,
        "note": (
            f"Expected {expected} object masks on this frame, found {masked}. "
            "Stay on this frame and segment each remaining object."
            if incomplete else None
        ),
    }


def project_overview(project: dict, current_video_id: Optional[str], current_frame: int) -> dict:
    """Compact project snapshot the LLM can reason over."""
    videos = project.get("videos") or {}
    video_rows = []
    for vid, v in videos.items():
        objects = v.get("objects") or {}
        prompts = v.get("point_prompts") or {}
        prompted_frames: set[int] = set()
        for fmap in prompts.values():
            if isinstance(fmap, dict):
                for fk in fmap:
                    try:
                        prompted_frames.add(int(fk))
                    except (TypeError, ValueError):
                        pass
        video_rows.append({
            "id": vid,
            "name": v.get("name"),
            "num_frames": v.get("num_frames"),
            "object_count": len(objects),
            "is_current": vid == current_video_id,
        } if vid != current_video_id else {
            "id": vid,
            "name": v.get("name"),
            "num_frames": v.get("num_frames"),
            "fps": v.get("fps"),
            "width": v.get("width"),
            "height": v.get("height"),
            "start_frame": v.get("start_frame", 0),
            "anchor_batch_size": v.get("anchor_batch_size"),
            "object_count": len(objects),
            "objects": [
                {
                    "id": oid,
                    "name": obj.get("name"),
                    "description": obj.get("description") or "",
                    "color": obj.get("color"),
                }
                for oid, obj in objects.items()
            ],
            "prompted_frame_count": len(prompted_frames),
            "prompted_frames_sample": sorted(prompted_frames)[:24],
            "annotated_anchors": v.get("annotated_anchors") or [],
            "anchor_labeling_complete": bool(v.get("anchor_labeling_complete")),
            "propagation_complete": bool(v.get("propagation_complete")),
            "propagated_frame_count": len(v.get("propagated_frames") or []),
            "is_current": True,
        })
    return {
        "project_id": project.get("id"),
        "project_name": project.get("name"),
        "tracking_mode": project.get("tracking_mode") or "segmentation_tracking",
        "video_count": len(videos),
        "current_video_id": current_video_id,
        "current_frame": current_frame,
        "videos": video_rows,
    }


# ─── Tool schemas ─────────────────────────────────────────────────────────────


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "think",
        "description": (
            "Record a short reasoning trace before acting. Use this to plan the next "
            "few steps, decide whether a frame looks usable, or explain a fallback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Concise working reasoning."},
            },
            "required": ["thought"],
        },
    },
    {
        "name": "get_project_overview",
        "description": (
            "List videos in the project (ids, names, frame counts, objects, "
            "annotation/propagation status) and which video/frame is current."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_video_details",
        "description": "Detailed context for one video: objects, prompts, anchors, masks.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "Video id. Omit to use the current video.",
                },
            },
        },
    },
    {
        "name": "select_video",
        "description": "Switch the working video (updates the UI).",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "goto_frame",
        "description": (
            "Extract and load a frame for viewing/annotation, and move the UI there. "
            "Always do this before segmenting a frame that may not be extracted yet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "frame_idx": {"type": "integer"},
                "video_id": {"type": "string"},
            },
            "required": ["frame_idx"],
        },
    },
    {
        "name": "inspect_frame",
        "description": (
            "Load a frame and attach the image so you can see it. Returns mask/prompt "
            "stats plus a JPEG. Use this to judge occlusion, blur, or identity (e.g. "
            "shaved vs unshaved mouse) before choosing text vs point prompts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "frame_idx": {"type": "integer"},
                "video_id": {"type": "string"},
            },
        },
    },
    {
        "name": "plan_frames",
        "description": (
            "Compute a frame plan: every Nth frame (anchors), and/or nearby fallbacks "
            "around a frame that failed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "interval": {"type": "integer", "description": "e.g. 1000 for every 1000th frame"},
                "start_frame": {"type": "integer"},
                "include_last": {"type": "boolean", "default": True},
                "around_frame": {"type": "integer", "description": "If set, also return nearby fallbacks"},
                "nearby_offset": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "create_object",
        "description": "Create a tracked object with a stable id, display name, and text description.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Display name, e.g. BackShave"},
                "description": {
                    "type": "string",
                    "description": "SAM3 text prompt / identity description",
                },
                "color": {"type": "string"},
                "video_id": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "update_object",
        "description": "Rename an object or update its text description.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "video_id": {"type": "string"},
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "text_segment",
        "description": (
            "Run SAM3 text-based segmentation on a frame and bind a mask to an "
            "object id (creates the object if needed). When several mice are detected, "
            "the leftmost or rightmost mask is chosen if the text or object name implies "
            "a side (NoShave/unshaved/left vs HeadShave/shaved/right). Call once per "
            "identity. Persists like a normal annotation. If confidence is low, "
            "try inspect_frame on a nearby frame (±20) and retry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Grounding phrase, e.g. 'mouse with shaved patch on back'"},
                "frame_idx": {"type": "integer"},
                "object_id": {"type": "string", "description": "Existing object id to bind"},
                "object_name": {"type": "string", "description": "Create/find object by this name if id omitted"},
                "video_id": {"type": "string"},
                "min_score": {"type": "number", "default": 0.0},
            },
            "required": ["text"],
        },
    },
    {
        "name": "add_point_prompt",
        "description": (
            "Add positive/negative point prompts for an object on a frame (same as a "
            "user click). Coordinates are normalized [0,1]. Use after inspect_frame "
            "when you can see where to click, or to refine a text mask."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string"},
                "frame_idx": {"type": "integer"},
                "points": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "description": "[[x, y], ...] normalized 0-1",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "1 = positive, 0 = negative; same length as points",
                },
                "video_id": {"type": "string"},
            },
            "required": ["object_id", "points", "labels"],
        },
    },
    {
        "name": "evaluate_segmentation",
        "description": (
            "Score saved masks on a frame (area, coverage, overlap, confidence). "
            "Compares against every object in the video: missing objects are failures "
            "(reason=missing_mask, incomplete=true). Do not finish while "
            "masked_object_count is below the expected animal count."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "frame_idx": {"type": "integer"},
                "video_id": {"type": "string"},
            },
        },
    },
    {
        "name": "commit_anchor",
        "description": "Mark a planned anchor frame as labeled (same as Done / next in the UI).",
        "parameters": {
            "type": "object",
            "properties": {
                "frame_idx": {"type": "integer"},
                "anchor_index": {"type": "integer"},
                "video_id": {"type": "string"},
            },
            "required": ["frame_idx"],
        },
    },
    {
        "name": "start_propagation",
        "description": (
            "Start whole-video (or range) label propagation from existing point prompts. "
            "The UI connects to the live tracker progress stream."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_frame": {"type": "integer"},
                "end_frame": {"type": "integer", "description": "Inclusive; omit for end of video"},
                "use_all_anchors": {"type": "boolean", "default": True},
                "video_id": {"type": "string"},
            },
        },
    },
    {
        "name": "get_propagation_status",
        "description": "Current tracking / propagation status for a video.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
            },
        },
    },
]


SYSTEM_PROMPT = """You are the SAM3 Web Tracker agent. You annotate and track objects in behavior videos by calling tools. The user watches a live chat of your reasoning and actions while the main canvas updates the same way a human annotation would.

## What you know
- The project may contain many videos. Always check get_project_overview if you are unsure which video you are on, how many videos exist, or what they are named.
- Each video has a frame count, fps, start_frame, objects (id, name, description), saved point prompts, and optional propagation state.
- Frames are 0-indexed. The UI shows JPEG frames when paused; you must goto_frame / inspect_frame so the backend extracts the frame before SAM can run.
- Object ids are strings ("1", "2", …). Names like BackShave are display names — create them with create_object and keep using the returned id.
- SAM3 tracker mode stores **point prompts**. Text segmentation finds a mask, then we convert it to points and persist them so tracking/propagation behaves normally.
- Propagation (Track Objects) needs at least one point prompt. After you have labeled the planned frames, call start_propagation if the user asked to track.

## How to work
1. Orient: get_project_overview (and get_video_details for the active video).
2. Plan: if the user says "every Nth frame", call plan_frames. Typical interval is 1000 (or the video's anchor_batch_size).
3. For each planned frame:
   a. goto_frame then inspect_frame so you can see animals, occlusion, blur, and identity cues.
   b. create_object once per identity (reuse existing objects with the same name).
              c. text_segment once per identity with a phrase that includes count/side/shave-state (e.g. "unshaved mouse on the left" vs "shaved mouse on the right near the water port"). Reuse existing object ids.
   d. evaluate_segmentation. If confidence is low, the frame is empty, identities are swapped, or animals overlap badly: think, then try nearby frames (±20, then ±40) via plan_frames(around_frame=..., nearby_offset=20). If the result is incomplete / missing_mask, stay on this frame and segment the remaining animals.
   e. If text segmentation is weak but you can see the animal, add_point_prompt at a chest/back point (positive=1). Use a negative point (0) on the other animal if they touch.
   f. commit_anchor when the frame is a planned grid/anchor frame and the masks look right.
4. Adapt. Do not blindly march the grid if a frame is unusable. Skip to a clearer neighbor, then continue the plan.
5. After the requested frames are labeled, start_propagation if the user wants tracking. Do not start it for a single-frame-only request.

## Text vs points
- Prefer text_segment when the description is visually distinctive (shaved patch, color, size).
- Prefer inspect_frame + add_point_prompt when animals look similar, are overlapping, or text confidence is low.
- Never invent coordinates without having inspected the frame (or a text_segment result that returned a bbox).
- If the user says there are N mice/animals, that count is ground truth. Call text_segment once per identity. Put side and shave-state in the phrase. The backend binds the leftmost or rightmost detection when the phrase or object name (NoShave / HeadShave) implies a side. Do not finish until evaluate_segmentation reports masked_object_count >= N and no missing_mask. One mask is a failure.
- If evaluate_segmentation returns incomplete or missing_mask, stay on this frame and segment the remaining object. A blob on empty bedding is not a mouse — re-inspect and click the animal that still has no mask. In top-down home-cage videos the two mice are often on opposite sides; a "second" mask around x=0.60–0.67 is frequently bedding, not the far-right animal. A click on the circular water-bottle port (far-right wall) makes SAM fill most of the cage — click the animal's torso instead, and add a negative point on the other mouse.
- When using add_point_prompt, place a new positive point on that animal's torso from the inspect JPEG. Do not copy another object's point list.

## Style
- Call think before multi-step plans and after failures.
- Keep user-facing messages short. Put detail in think / tool results.
- Do not reset or delete the user's existing objects unless they ask.
- Stay on the video the user specified; if they said "this video" use the current one.
- If the tracking mode is pose_tracking, say you only handle segmentation/tracking objects here.
"""


# ─── Run state ────────────────────────────────────────────────────────────────


@dataclass
class AgentSession:
    pid: str
    video_id: Optional[str]
    frame_idx: int
    cancel: threading.Event = field(default_factory=threading.Event)
    pending_images: list[dict] = field(default_factory=list)
    ui_events: list[dict] = field(default_factory=list)

    def emit_ui(self, action: str, **payload: Any) -> dict:
        ev = {"action": action, **payload}
        self.ui_events.append(ev)
        return ev


@dataclass
class AgentRunState:
    is_running: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    run_id: Optional[str] = None


_RUNS: dict[str, AgentRunState] = {}
_RUNS_LOCK = threading.Lock()


def get_agent_run_state(pid: str) -> AgentRunState:
    with _RUNS_LOCK:
        st = _RUNS.get(pid)
        if st is None:
            st = AgentRunState()
            _RUNS[pid] = st
        return st


def cancel_agent_run(pid: str) -> bool:
    st = get_agent_run_state(pid)
    if not st.is_running:
        return False
    st.cancel.set()
    return True


# ─── Tool execution ───────────────────────────────────────────────────────────


def _srv():
    import server as s
    return s


def _resolve_video(session: AgentSession, video_id: Optional[str]):
    s = _srv()
    vid = video_id or session.video_id
    if not vid:
        raise ValueError("No video selected. Call select_video or pass video_id.")
    video = s.pm.get_video(session.pid, vid)
    if video is None:
        raise ValueError(f"Video {vid} not found in project {session.pid}")
    return vid, video


def _ensure_frame(session: AgentSession, vid: str, frame_idx: int) -> dict:
    s = _srv()
    video = s.pm.get_video(session.pid, vid)
    if video is None:
        raise ValueError(f"Video {vid} not found")
    last = max(0, int(video.get("num_frames") or 1) - 1)
    fidx = max(0, min(last, int(frame_idx)))
    s.extract_annotated_frame(session.pid, vid, fidx)
    # Also populate frames/ so the viewer JPEG exists.
    frames_dir = s.pm.frames_dir(session.pid, vid)
    frame_path = frames_dir / f"{fidx:06d}.jpg"
    if not frame_path.exists():
        source_path = video.get("source_path")
        if source_path:
            from pathlib import Path
            from video_processor import extract_frame_range
            ds_max_dim, ds_scale_factor = s._video_ds_params(video)
            try:
                extract_frame_range(
                    source_path, str(frames_dir), fidx, fidx + 1,
                    max_dim=ds_max_dim, scale_factor=ds_scale_factor,
                )
            except Exception as e:
                logger.warning(f"frames/ extract {fidx}: {e}")
    session.video_id = vid
    session.frame_idx = fidx
    session.emit_ui("goto_frame", video_id=vid, frame_idx=fidx)
    return {"video_id": vid, "frame_idx": fidx, "num_frames": video.get("num_frames")}


def _encode_inspect_jpeg(path: str) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, INSPECT_MAX_EDGE / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def _find_or_create_object(session: AgentSession, vid: str, object_id: Optional[str], object_name: Optional[str], description: str = "") -> dict:
    s = _srv()
    video = s.pm.get_video(session.pid, vid)
    objects = (video or {}).get("objects") or {}
    if object_id and object_id in objects:
        return objects[object_id]
    if object_name:
        for obj in objects.values():
            if (obj.get("name") or "").lower() == object_name.lower():
                return obj
        obj = s.pm.add_object(session.pid, vid, object_name, description=description or "")
        session.emit_ui("refresh_project")
        session.emit_ui("select_object", object_id=obj["id"], video_id=vid)
        return obj
    if object_id:
        raise ValueError(f"Object {object_id} not found")
    raise ValueError("Provide object_id or object_name")


def _apply_points(session: AgentSession, vid: str, oid: str, frame_idx: int, points: list, labels: list) -> dict:
    s = _srv()
    from fastapi import HTTPException
    req = s.AddPointsRequest(frame_idx=frame_idx, points=points, labels=labels, text=None, anchor_mode=False)
    try:
        result = s.add_points(session.pid, vid, oid, req)
    except HTTPException as e:
        raise ValueError(e.detail if isinstance(e.detail, str) else str(e.detail)) from e
    masks = result.get("masks") or {}
    session.emit_ui("goto_frame", video_id=vid, frame_idx=frame_idx)
    session.emit_ui("select_object", object_id=str(oid), video_id=vid)
    session.emit_ui("set_masks", video_id=vid, frame_idx=frame_idx, masks=masks)
    session.emit_ui("set_points", video_id=vid, object_id=str(oid), frame_idx=frame_idx, points=points, labels=labels)
    session.emit_ui("refresh_project")
    return result


def _as_seq(val: Any) -> list:
    """Numpy arrays are truthy-ambiguous; never use `arr or []`."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return []
        return list(val)
    if isinstance(val, (list, tuple)):
        return list(val)
    try:
        return list(val)
    except TypeError:
        return []


def _detections_from_sam_outputs(frame_outputs: dict) -> list[dict]:
    if not isinstance(frame_outputs, dict):
        return []
    obj_ids = _as_seq(frame_outputs.get("out_obj_ids"))
    masks = _as_seq(frame_outputs.get("out_binary_masks"))
    boxes = _as_seq(frame_outputs.get("out_boxes_xywh"))
    probs = _as_seq(frame_outputs.get("out_probs"))
    dets: list[dict] = []
    for i, oid in enumerate(obj_ids):
        mask = masks[i] if i < len(masks) else None
        if mask is None:
            continue
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            continue
        binary = (mask > 0.5).astype(np.uint8) if mask.dtype != np.uint8 else (mask > 0).astype(np.uint8)
        score = None
        if i < len(probs):
            p = probs[i]
            if hasattr(p, "item"):
                p = p.item()
            try:
                score = float(p)
            except (TypeError, ValueError):
                score = None
        bbox = None
        if i < len(boxes):
            b = boxes[i]
            if hasattr(b, "tolist"):
                b = b.tolist()
            try:
                bbox = [float(x) for x in list(b)[:4]]
            except (TypeError, ValueError):
                bbox = None
        dets.append({"sam_obj_id": int(oid) if str(oid).lstrip("-").isdigit() else oid, "mask": binary, "score": score, "bbox": bbox})
    dets.sort(key=lambda d: (d["score"] is not None, d["score"] or 0.0), reverse=True)
    return dets


def _text_segment_impl(session: AgentSession, vid: str, frame_idx: int, text: str, object_id: Optional[str], object_name: Optional[str], min_score: float) -> dict:
    s = _srv()
    from sam_predictor import sample_points_from_mask
    from video_processor import encode_mask_as_png

    _ensure_frame(session, vid, frame_idx)
    obj = None
    if object_id or object_name:
        obj = _find_or_create_object(session, vid, object_id, object_name, description=text)

    video = s.pm.get_video(session.pid, vid)
    objects = (video or {}).get("objects") or {}
    ann_dir = s.pm.annotated_frames_dir(session.pid, vid)
    jpg = ann_dir / f"{frame_idx:06d}.jpg"

    detections: list[dict] = []
    method = None
    error = None

    # Image-level grounding does not touch the video tracker session.
    if jpg.exists():
        try:
            img_dets = s.sam.detect_text_on_image(str(jpg), text)
            detections = [
                {"sam_obj_id": None, "mask": d["mask"], "score": d.get("score"), "bbox": d.get("bbox_xywh_norm")}
                for d in img_dets
            ]
            if detections:
                method = "sam3_image_text"
        except Exception as e:
            error = str(e)
            logger.warning(f"SAM3 image text detect failed: {e}")

    # Prefer a throwaway single-frame session so detector object ids do not pollute tracking.
    if not detections:
        import shutil
        import tempfile
        from pathlib import Path
        tmp_ann = tempfile.mkdtemp(prefix="sam3wt_agent_txt_")
        try:
            if jpg.exists():
                shutil.copy2(str(jpg), str(Path(tmp_ann) / jpg.name))
                s.sam.init_session(session.pid, vid, tmp_ann)
                try:
                    out = s.sam.add_text_prompt(session.pid, vid, frame_idx, text)
                    frame_out = out.get(frame_idx) or out.get(str(frame_idx)) or {}
                    if not frame_out and out:
                        frame_out = next(iter(out.values()))
                    detections = _detections_from_sam_outputs(frame_out)
                    method = "sam3_video_text"
                except Exception as e:
                    error = error or str(e)
                    logger.warning(f"SAM3 video text prompt failed: {e}")
        except Exception as e:
            error = error or str(e)
            logger.warning(f"temp text session failed: {e}")
        finally:
            shutil.rmtree(tmp_ann, ignore_errors=True)
            try:
                s.sam.close_session(session.pid, vid)
            except Exception:
                pass

    detections = [d for d in detections if (d.get("score") is None or d["score"] >= min_score)]
    if not detections:
        return {
            "ok": False,
            "method": method,
            "error": error or "no_detections",
            "text": text,
            "frame_idx": frame_idx,
            "video_id": vid,
            "suggestion": (
                f"No confident mask for {text!r} on frame {frame_idx}. "
                f"inspect_frame, then try nearby_frames ±{DEFAULT_NEARBY_OFFSET} or add_point_prompt."
            ),
        }

    best = _pick_detection(detections, text, obj)
    picked_side = _prefer_side_from_text_and_object(text, obj)
    picked_xy = _bbox_center_xy(best)
    quality = evaluate_mask_quality(best["mask"], score=best.get("score"))
    far_right_miss = bool(
        picked_side == "right"
        and _text_implies_far_right(text, obj)
        and picked_xy[0] < 0.70
    )
    suggestion = None
    if far_right_miss:
        quality = {
            **quality,
            "ok": False,
            "reason": (
                "detection_not_on_far_right"
                if quality.get("reason") in (None, "", "ok")
                else f"{quality.get('reason')},detection_not_on_far_right"
            ),
            "retry_nearby": False,
        }
        suggestion = (
            "Text grounding's rightmost hit is still mid-cage (often empty bedding around x=0.60–0.67). "
            "add_point_prompt on the far-right mouse torso from the inspect JPEG (about 0.80, 0.42)."
        )

    bound = None
    if obj is not None and not far_right_miss:
        points = sample_points_from_mask(best["mask"], n_total=8)
        if not points:
            return {**quality, "ok": False, "error": "empty_best_mask", "frame_idx": frame_idx}
        labels = [1] * len(points)
        bound = _apply_points(session, vid, str(obj["id"]), frame_idx, points, labels)
        color = obj.get("color") or "#5B8DD9"
    else:
        color = (obj or {}).get("color") or "#5B8DD9"

    mask_b64 = encode_mask_as_png(best["mask"], color)
    extra = [_detection_summary(d) for d in detections]

    return {
        "ok": quality["ok"],
        "method": method,
        "text": text,
        "frame_idx": frame_idx,
        "video_id": vid,
        "object": obj,
        "quality": quality,
        "picked_side": picked_side,
        "picked_center_xy": [round(picked_xy[0], 4), round(picked_xy[1], 4)],
        "detection_count": len(detections),
        "all_detections": extra,
        "seed_points": bool(bound and bound.get("masks")),
        "masks": (bound or {}).get("masks") or ({str(obj["id"]): mask_b64} if obj and not far_right_miss else {"det": mask_b64}),
        "retry_nearby": quality.get("retry_nearby"),
        "suggestion": suggestion,
    }


def execute_tool(name: str, args: dict, session: AgentSession) -> tuple[dict, list[dict]]:
    """Run one tool. Returns (result_dict, ui_events_emitted_this_call)."""
    session.ui_events.clear()
    args = args or {}
    try:
        result = _execute_tool_inner(name, args, session)
    except Exception as e:
        logger.exception(f"agent tool {name} failed")
        result = {"ok": False, "error": str(e), "tool": name}
    ui = list(session.ui_events)
    session.ui_events.clear()
    return result, ui


def _execute_tool_inner(name: str, args: dict, session: AgentSession) -> dict:
    s = _srv()
    from anchor_helpers import compute_anchor_frames, video_anchor_batch_size
    from mask_store import VideoMaskStorage

    if name == "think":
        return {"ok": True, "thought": args.get("thought") or ""}

    if name == "get_project_overview":
        project = s.pm.get_project(session.pid)
        if project is None:
            raise ValueError("Project not found")
        return {"ok": True, **project_overview(project, session.video_id, session.frame_idx)}

    if name == "get_video_details":
        vid, video = _resolve_video(session, args.get("video_id"))
        start = int(video.get("start_frame") or 0)
        n = int(video.get("num_frames") or 0)
        batch = video_anchor_batch_size(video)
        anchors = compute_anchor_frames(start, n, batch) if n else []
        ms = VideoMaskStorage(s.pm.video_dir(session.pid, vid))
        saved = []
        try:
            saved = ms.list_frames()[:40] if hasattr(ms, "list_frames") else []
        except Exception:
            saved = []
        if not saved:
            try:
                from pathlib import Path
                mask_dir = s.pm.video_dir(session.pid, vid) / "masks"
                if mask_dir.exists():
                    saved = sorted(int(p.stem) for p in mask_dir.glob("*.npz"))[:40]
            except Exception:
                saved = []
        return {
            "ok": True,
            "video_id": vid,
            "name": video.get("name"),
            "num_frames": n,
            "fps": video.get("fps"),
            "width": video.get("width"),
            "height": video.get("height"),
            "start_frame": start,
            "anchor_batch_size": batch,
            "planned_anchors": anchors[:80],
            "anchor_count": len(anchors),
            "objects": video.get("objects") or {},
            "point_prompts": video.get("point_prompts") or {},
            "annotated_anchors": video.get("annotated_anchors") or [],
            "anchor_labeling_complete": bool(video.get("anchor_labeling_complete")),
            "propagation_complete": bool(video.get("propagation_complete")),
            "propagated_frame_count": len(video.get("propagated_frames") or []),
            "saved_mask_frames_sample": saved,
            "current_frame": session.frame_idx if session.video_id == vid else start,
        }

    if name == "select_video":
        vid, video = _resolve_video(session, args.get("video_id"))
        session.video_id = vid
        session.frame_idx = int(video.get("start_frame") or 0)
        session.emit_ui("select_video", video_id=vid)
        session.emit_ui("goto_frame", video_id=vid, frame_idx=session.frame_idx)
        return {"ok": True, "video_id": vid, "name": video.get("name"), "num_frames": video.get("num_frames"), "frame_idx": session.frame_idx}

    if name == "goto_frame":
        vid, video = _resolve_video(session, args.get("video_id"))
        info = _ensure_frame(session, vid, int(args["frame_idx"]))
        return {"ok": True, **info, "name": video.get("name")}

    if name == "inspect_frame":
        vid, video = _resolve_video(session, args.get("video_id"))
        fidx = int(args["frame_idx"]) if args.get("frame_idx") is not None else session.frame_idx
        _ensure_frame(session, vid, fidx)
        ann = s.pm.annotated_frames_dir(session.pid, vid) / f"{fidx:06d}.jpg"
        frames = s.pm.frames_dir(session.pid, vid) / f"{fidx:06d}.jpg"
        path = str(ann if ann.exists() else frames)
        if not (ann.exists() or frames.exists()):
            raise ValueError(f"Frame {fidx} could not be extracted")
        jpeg_b64 = _encode_inspect_jpeg(path)
        session.pending_images.append({"frame_idx": fidx, "video_id": vid, "jpeg_b64": jpeg_b64})
        objects = video.get("objects") or {}
        prompts = video.get("point_prompts") or {}
        on_frame = {
            oid: fmap.get(str(fidx))
            for oid, fmap in prompts.items()
            if isinstance(fmap, dict) and str(fidx) in fmap
        }
        evals = []
        try:
            ms = VideoMaskStorage(s.pm.video_dir(session.pid, vid))
            dense = ms.load_masks_dense(fidx) if hasattr(ms, "load_masks_dense") else {}
            for oid, mask in (dense or {}).items():
                q = evaluate_mask_quality(mask)
                q["object_id"] = str(oid)
                evals.append(q)
        except Exception:
            pass
        return {
            "ok": True,
            "video_id": vid,
            "video_name": video.get("name"),
            "frame_idx": fidx,
            "num_frames": video.get("num_frames"),
            "objects": [
                {"id": oid, "name": o.get("name"), "description": o.get("description") or ""}
                for oid, o in objects.items()
            ],
            "prompts_on_frame": on_frame,
            "existing_mask_quality": evals,
            "image_attached": True,
            "note": "A JPEG of this frame is attached for you to look at.",
        }

    if name == "plan_frames":
        vid, video = _resolve_video(session, args.get("video_id") if "video_id" in args else None)
        n = int(video.get("num_frames") or 0)
        start = int(args["start_frame"]) if args.get("start_frame") is not None else int(video.get("start_frame") or 0)
        interval = int(args["interval"]) if args.get("interval") is not None else int(video_anchor_batch_size(video))
        include_last = args.get("include_last", True)
        planned = plan_sample_frames(start, n, interval, include_last=bool(include_last))
        nearby = []
        if args.get("around_frame") is not None:
            nearby = nearby_frames(
                int(args["around_frame"]),
                n,
                offset=int(args.get("nearby_offset") or DEFAULT_NEARBY_OFFSET),
            )
        return {
            "ok": True,
            "video_id": vid,
            "interval": interval,
            "start_frame": start,
            "num_frames": n,
            "planned_frames": planned,
            "count": len(planned),
            "nearby_frames": nearby,
        }

    if name == "create_object":
        vid, _video = _resolve_video(session, args.get("video_id"))
        existing = (_srv().pm.get_video(session.pid, vid) or {}).get("objects") or {}
        for obj in existing.values():
            if (obj.get("name") or "").lower() == str(args["name"]).lower():
                return {"ok": True, "object": obj, "created": False}
        obj = s.pm.add_object(
            session.pid, vid, str(args["name"]),
            color=args.get("color"),
            description=args.get("description") or "",
        )
        session.emit_ui("refresh_project")
        session.emit_ui("select_object", object_id=obj["id"], video_id=vid)
        return {"ok": True, "object": obj, "created": True}

    if name == "update_object":
        vid, _video = _resolve_video(session, args.get("video_id"))
        oid = str(args["object_id"])
        updates = {}
        if args.get("name"):
            s.pm.rename_object(session.pid, vid, oid, args["name"])
        if args.get("description") is not None:
            updates["description"] = args["description"]
        if updates:
            s.pm.update_object(session.pid, vid, oid, **updates)
        session.emit_ui("refresh_project")
        video = s.pm.get_video(session.pid, vid)
        return {"ok": True, "object": (video or {}).get("objects", {}).get(oid)}

    if name == "text_segment":
        vid, _video = _resolve_video(session, args.get("video_id"))
        fidx = int(args["frame_idx"]) if args.get("frame_idx") is not None else session.frame_idx
        return _text_segment_impl(
            session, vid, fidx,
            text=str(args.get("text") or ""),
            object_id=args.get("object_id"),
            object_name=args.get("object_name"),
            min_score=float(args.get("min_score") or 0.0),
        )

    if name == "add_point_prompt":
        vid, _video = _resolve_video(session, args.get("video_id"))
        fidx = int(args["frame_idx"]) if args.get("frame_idx") is not None else session.frame_idx
        _ensure_frame(session, vid, fidx)
        points = args.get("points") or []
        labels = args.get("labels") or []
        if len(points) != len(labels) or not points:
            raise ValueError("points and labels must be non-empty and the same length")
        result = _apply_points(session, vid, str(args["object_id"]), fidx, points, labels)
        return {"ok": True, "frame_idx": fidx, "object_id": str(args["object_id"]), "mask_ids": list((result.get("masks") or {}).keys())}

    if name == "evaluate_segmentation":
        vid, video = _resolve_video(session, args.get("video_id"))
        fidx = int(args["frame_idx"]) if args.get("frame_idx") is not None else session.frame_idx
        ms = VideoMaskStorage(s.pm.video_dir(session.pid, vid))
        dense = {}
        try:
            dense = ms.load_masks_dense(fidx) if hasattr(ms, "load_masks_dense") else {}
        except Exception as e:
            return {"ok": False, "error": str(e), "frame_idx": fidx}
        objects = video.get("objects") or {}
        per_obj = []
        masks = list(dense.values())
        for oid, mask in (dense or {}).items():
            q = evaluate_mask_quality(mask)
            q["object_id"] = str(oid)
            q["name"] = (objects.get(str(oid)) or {}).get("name")
            # overlap with others
            overlap = 0
            b = np.squeeze(mask).astype(bool)
            for oid2, m2 in dense.items():
                if str(oid2) == str(oid):
                    continue
                overlap = max(overlap, int((b & np.squeeze(m2).astype(bool)).sum()))
            q["overlap_px"] = overlap
            if overlap > 0.35 * max(1, q["area_px"]):
                q["ok"] = False
                q["retry_nearby"] = True
                q["reason"] = (q.get("reason") or "") + ",high_overlap"
            per_obj.append(q)
        summary = finalize_segmentation_eval(objects, per_obj)
        return {
            "ok": summary["ok"],
            "frame_idx": fidx,
            "video_id": vid,
            "objects": summary["objects"],
            "object_count": len(summary["objects"]),
            "expected_object_count": summary["expected_object_count"],
            "masked_object_count": summary["masked_object_count"],
            "incomplete": summary["incomplete"],
            "note": summary["note"],
            "retry_nearby": summary["retry_nearby"],
            "nearby_suggestion": nearby_frames(fidx, int(video.get("num_frames") or 1)),
        }

    if name == "commit_anchor":
        vid, video = _resolve_video(session, args.get("video_id"))
        fidx = int(args["frame_idx"])
        start = int(video.get("start_frame") or 0)
        anchors = compute_anchor_frames(start, int(video.get("num_frames") or 0), video_anchor_batch_size(video))
        idx = int(args["anchor_index"]) if args.get("anchor_index") is not None else (
            anchors.index(fidx) if fidx in anchors else 0
        )
        from fastapi import HTTPException
        req = s.CommitAnchorRequest(anchor_index=idx, labeling_timing=None)
        try:
            result = s.commit_anchor_frame(session.pid, vid, fidx, req)
        except HTTPException as e:
            raise ValueError(e.detail if isinstance(e.detail, str) else str(e.detail)) from e
        session.emit_ui("refresh_project")
        session.emit_ui("set_anchor_phase", video_id=vid, frames=anchors, current_index=idx, committed_frame=fidx)
        return {"ok": True, **result, "planned_anchors": anchors}

    if name == "start_propagation":
        vid, video = _resolve_video(session, args.get("video_id"))
        prompts = video.get("point_prompts") or {}
        if not prompts:
            raise ValueError("No point prompts yet. Segment/annotate at least one frame first.")
        start_frame = int(args["start_frame"]) if args.get("start_frame") is not None else int(video.get("start_frame") or 0)
        end_frame = int(args["end_frame"]) if args.get("end_frame") is not None else -1
        use_all = bool(args.get("use_all_anchors", True))
        session.emit_ui(
            "start_propagation",
            video_id=vid,
            start_frame=start_frame,
            end_frame=end_frame,
            use_all_anchors=use_all,
        )
        return {
            "ok": True,
            "video_id": vid,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "use_all_anchors": use_all,
            "note": "The UI will attach to the live propagation stream.",
        }

    if name == "get_propagation_status":
        vid, _video = _resolve_video(session, args.get("video_id"))
        from fastapi import HTTPException
        try:
            return {"ok": True, **s.get_propagation_status(session.pid, vid)}
        except HTTPException as e:
            raise ValueError(e.detail if isinstance(e.detail, str) else str(e.detail)) from e

    raise ValueError(f"Unknown tool: {name}")


# ─── LLM I/O ──────────────────────────────────────────────────────────────────


def _openai_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TOOL_SCHEMAS
    ]


def _anthropic_tools() -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in TOOL_SCHEMAS
    ]


def _http_json(method: str, url: str, headers: dict, payload: dict, timeout: float = 120.0) -> dict:
    try:
        import httpx
        with httpx.Client(timeout=timeout) as client:
            r = client.request(method, url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()
    except ImportError:
        import urllib.request
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={**headers, "Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())


def call_llm(cfg: LLMConfig, messages: list[dict]) -> dict:
    """One chat turn. Returns {content, tool_calls: [{id, name, arguments}]}."""
    if cfg.provider == "anthropic":
        return _call_anthropic(cfg, messages)
    return _call_openai(cfg, messages)


def _message_image_parts_openai(images: list[dict]) -> list[dict]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{im['jpeg_b64']}"},
        }
        for im in images
        if im.get("jpeg_b64")
    ]


def _call_openai(cfg: LLMConfig, messages: list[dict]) -> dict:
    url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    oai_messages = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            oai_messages.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or m.get("id") or "tool",
                "content": m.get("content") or "",
            })
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {
                        "id": tc.get("id") or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"]),
                        },
                    }
                    for i, tc in enumerate(m["tool_calls"])
                ]
            oai_messages.append(msg)
        else:
            images = m.get("images") or []
            text = m.get("content") or ""
            if images:
                oai_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": text}, *_message_image_parts_openai(images)],
                })
            else:
                oai_messages.append({"role": role, "content": text})

    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": oai_messages,
        "tools": _openai_tools(),
        "tool_choice": "auto",
        "temperature": 0.2,
    }
    # SAM 3 Agent / vLLM use max_tokens; newer OpenAI prefers max_completion_tokens.
    if cfg.local or cfg.provider in LOCAL_PROVIDERS:
        payload["max_tokens"] = AGENT_MAX_TOKENS
    else:
        payload["max_completion_tokens"] = AGENT_MAX_TOKENS
    headers = {"Authorization": f"Bearer {cfg.api_key or LOCAL_DUMMY_KEY}"}
    timeout = 300.0 if cfg.local else 120.0
    data = _http_json("POST", url, headers, payload, timeout=timeout)
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    tool_calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            parsed = {}
        tool_calls.append({"id": tc.get("id") or str(uuid.uuid4()), "name": fn.get("name"), "arguments": parsed})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning and reasoning not in content:
        content = f"{reasoning}\n{content}".strip()
    return {"content": content, "tool_calls": tool_calls}


def _call_anthropic(cfg: LLMConfig, messages: list[dict]) -> dict:
    url = (cfg.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
    sys_text = ""
    anth_msgs: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            sys_text += (m.get("content") or "") + "\n"
            continue
        if m["role"] == "tool":
            anth_msgs.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id") or m.get("id") or "tool",
                "content": m.get("content") or "",
            }]})
            continue
        if m.get("images"):
            block = [{"type": "text", "text": m.get("content") or ""}]
            for im in m["images"]:
                if im.get("jpeg_b64"):
                    block.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": im["jpeg_b64"]},
                    })
            anth_msgs.append({"role": "user", "content": block})
            continue
        if m["role"] == "assistant":
            content: list[dict] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id") or str(uuid.uuid4()),
                    "name": tc["name"],
                    "input": tc["arguments"] if isinstance(tc["arguments"], dict) else {},
                })
            anth_msgs.append({"role": "assistant", "content": content or [{"type": "text", "text": ""}]})
            continue
        anth_msgs.append({"role": "user", "content": m.get("content") or ""})

    # Anthropic requires alternating roles; merge consecutive user tool results.
    merged: list[dict] = []
    for msg in anth_msgs:
        if merged and merged[-1]["role"] == msg["role"] and msg["role"] == "user":
            prev = merged[-1]["content"]
            extra = msg["content"]
            if isinstance(prev, str):
                prev = [{"type": "text", "text": prev}]
            if isinstance(extra, str):
                extra = [{"type": "text", "text": extra}]
            merged[-1]["content"] = list(prev) + list(extra)
        else:
            merged.append(msg)

    payload = {
        "model": cfg.model,
        "max_tokens": AGENT_MAX_TOKENS,
        "system": sys_text.strip() or SYSTEM_PROMPT,
        "messages": merged,
        "tools": _anthropic_tools(),
        "temperature": 0.2,
    }
    headers = {
        "x-api-key": cfg.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = _http_json("POST", url, headers, payload)
    content = ""
    tool_calls = []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            content += block.get("text") or ""
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") or str(uuid.uuid4()),
                "name": block.get("name"),
                "arguments": block.get("input") or {},
            })
    return {"content": content, "tool_calls": tool_calls}


# ─── Agent loop ───────────────────────────────────────────────────────────────


def build_initial_messages(
    user_text: str,
    overview: dict,
    history: list[dict],
) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    context = json.dumps(overview, indent=2)[:12000]
    msgs.append({
        "role": "user",
        "content": (
            f"Current project context (JSON):\n```json\n{context}\n```\n\n"
            f"User request:\n{user_text}"
        ),
    })
    return msgs


def prune_stale_inspect_images(messages: list[dict], keep: int = KEEP_INSPECT_IMAGES) -> int:
    """Drop all but the last `keep` inspect JPEGs from the in-flight transcript.

    The agent loop appends every inspect_frame image and never expires them.
    Qwen-VL charges ~400 tokens per 640×480 JPEG; a 36-step run with 4 inspects
    already overflows an 8k window, and a full every-1000th pass would not fit
    even 32k if every image were kept.
    """
    idxs = [i for i, m in enumerate(messages) if m.get("images")]
    if keep < 0:
        keep = 0
    drop = idxs if keep == 0 else idxs[:-keep]
    removed = 0
    for i in drop:
        n = len(messages[i].get("images") or [])
        messages[i].pop("images", None)
        removed += n
        prev = (messages[i].get("content") or "").strip()
        stub = f"[dropped {n} inspect JPEG(s) to save context; frame was already described in the tool result]"
        if stub not in prev:
            messages[i]["content"] = f"{prev}\n{stub}".strip() if prev else stub
    return removed


_LAST_AGENT_DUMP: dict[str, Any] = {}


def get_last_agent_dump() -> dict[str, Any]:
    return dict(_LAST_AGENT_DUMP)


def inspect_jpegs_in_messages(messages: list[dict]) -> list[dict]:
    """Metadata for inspect JPEGs currently attached to the in-flight transcript."""
    out: list[dict] = []
    for mi, m in enumerate(messages):
        for ii, im in enumerate(m.get("images") or []):
            out.append({
                "message_index": mi,
                "image_index": ii,
                "role": m.get("role"),
                "frame_idx": im.get("frame_idx"),
                "video_id": im.get("video_id"),
                "has_jpeg": bool(im.get("jpeg_b64")),
            })
    return out


def default_agent_dump_dir() -> Path:
    env = (os.environ.get("AGENT_LLM_DUMP_DIR") or "").strip()
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[1]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return root / "reports" / "agent-runs" / stamp


def _messages_without_jpegs(messages: list[dict]) -> list[dict]:
    slim: list[dict] = []
    for m in messages:
        item = {k: v for k, v in m.items() if k != "images"}
        images = m.get("images") or []
        if images:
            item["images"] = [
                {
                    "frame_idx": im.get("frame_idx"),
                    "video_id": im.get("video_id"),
                    "jpeg_chars": len(im.get("jpeg_b64") or ""),
                }
                for im in images
            ]
        slim.append(item)
    return slim


def write_agent_context_dump(
    dump_dir: Path,
    *,
    system_prompt: str,
    overview: dict,
    user_text: str,
    messages: list[dict],
    events: list[dict],
    llm_turns: list[dict],
    cfg: LLMConfig,
    inspect_images: list[dict],
) -> dict:
    """Write a readable agent-context report (markdown + JSON + inspect JPEGs)."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    jpeg_files: list[dict] = []
    seen: set[tuple] = set()
    for im in inspect_images:
        key = (str(im.get("video_id") or ""), im.get("frame_idx"))
        b64 = im.get("jpeg_b64") or ""
        if not b64 or key in seen:
            continue
        seen.add(key)
        name = f"inspect_vid-{key[0]}_frame-{im.get('frame_idx')}.jpg"
        try:
            (dump_dir / name).write_bytes(base64.b64decode(b64))
        except Exception:
            continue
        jpeg_files.append({
            "file": name,
            "frame_idx": im.get("frame_idx"),
            "video_id": im.get("video_id"),
        })

    final_in_context = inspect_jpegs_in_messages(messages)
    in_context_keys = {
        (str(x.get("video_id") or ""), x.get("frame_idx"))
        for x in final_in_context
    }
    for rec in jpeg_files:
        rec["in_final_llm_context"] = (
            str(rec.get("video_id") or ""), rec.get("frame_idx")
        ) in in_context_keys

    payload = {
        "llm": {
            "provider": cfg.provider,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "profile": normalize_llm_profile(),
            "local": cfg.local,
        },
        "user_text": user_text,
        "system_prompt": system_prompt,
        "retrieved_project_json": overview,
        "events": events,
        "llm_turns": llm_turns,
        "inspect_jpegs_seen": jpeg_files,
        "inspect_jpegs_in_final_context": final_in_context,
        "messages": _messages_without_jpegs(messages),
        "keep_inspect_images": KEEP_INSPECT_IMAGES,
        "max_tokens": AGENT_MAX_TOKENS,
        "dump_dir": str(dump_dir),
    }
    (dump_dir / "context.json").write_text(json.dumps(payload, indent=2, default=str))

    lines: list[str] = [
        "# Agent context dump",
        "",
        f"- Provider: `{cfg.provider}`  model: `{cfg.model}`  profile: `{normalize_llm_profile()}`",
        f"- Base URL: `{cfg.base_url}`",
        f"- User: {user_text}",
        f"- KEEP_INSPECT_IMAGES={KEEP_INSPECT_IMAGES}  AGENT_MAX_TOKENS={AGENT_MAX_TOKENS}",
        "",
        "## System prompt",
        "",
        "```",
        system_prompt.strip(),
        "```",
        "",
        "## Retrieved project JSON",
        "",
        "```json",
        json.dumps(overview, indent=2, default=str)[:12000],
        "```",
        "",
        "## Inspect JPEGs",
        "",
    ]
    if not jpeg_files:
        lines.append("_No inspect_frame JPEGs were attached._")
    else:
        lines.append("| File | video_id | frame | in final LLM context |")
        lines.append("|---|---|---|---|")
        for rec in jpeg_files:
            lines.append(
                f"| `{rec['file']}` | `{rec.get('video_id')}` | {rec.get('frame_idx')} | "
                f"{'yes' if rec.get('in_final_llm_context') else 'no (pruned)'} |"
            )
        lines.append("")
        for rec in jpeg_files:
            lines.append(f"### {rec['file']}")
            lines.append("")
            lines.append(f"![inspect frame {rec.get('frame_idx')}]({rec['file']})")
            lines.append("")

    lines += [
        "## LLM turns (which inspect JPEGs were in context)",
        "",
    ]
    for turn in llm_turns:
        kept = turn.get("inspect_jpegs_in_context") or []
        desc = ", ".join(
            f"video `{x.get('video_id')}` frame {x.get('frame_idx')}" for x in kept
        ) or "none"
        lines.append(f"- Step {turn.get('step')}: {desc}")
    lines += ["", "## Tool calls and results", ""]
    for ev in events:
        kind = ev.get("event")
        data = ev.get("data") or {}
        if kind == "tool_call":
            lines.append(f"### `{data.get('name')}` (step {data.get('step')})")
            lines.append("")
            lines.append("Arguments:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(data.get("arguments") or {}, indent=2, default=str))
            lines.append("```")
            lines.append("")
        elif kind == "tool_result":
            lines.append(f"Result (`ok={data.get('ok')}`, {data.get('elapsed_ms')} ms):")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(data.get("result") or {}, indent=2, default=str)[:8000])
            lines.append("```")
            lines.append("")
        elif kind in ("reasoning", "message", "error"):
            text = (data.get("text") or data.get("message") or "").strip()
            if text:
                lines.append(f"**{kind}:** {text}")
                lines.append("")
    (dump_dir / "context.md").write_text("\n".join(lines))

    latest = dump_dir.parent / "latest.json"
    latest.write_text(json.dumps({"dump_dir": str(dump_dir), "context_md": str(dump_dir / "context.md")}, indent=2))

    summary = {
        "dump_dir": str(dump_dir),
        "context_md": str(dump_dir / "context.md"),
        "context_json": str(dump_dir / "context.json"),
        "inspect_jpegs_seen": jpeg_files,
        "inspect_jpegs_in_final_context": final_in_context,
        "llm": payload["llm"],
        "tool_call_count": sum(1 for e in events if e.get("event") == "tool_call"),
    }
    global _LAST_AGENT_DUMP
    _LAST_AGENT_DUMP = summary
    return summary


def _tool_result_for_llm(result: dict) -> str:
    slim = dict(result)
    slim.pop("masks", None)
    if slim.get("point_prompts") and isinstance(slim["point_prompts"], dict):
        # Keep structure but drop huge point lists
        slim["point_prompts"] = {
            oid: {fk: {"n_points": len((pv or {}).get("points") or [])} for fk, pv in fmap.items()}
            if isinstance(fmap, dict) else fmap
            for oid, fmap in slim["point_prompts"].items()
        }
    text = json.dumps(slim, default=str)
    if len(text) > 8000:
        text = text[:8000] + "…"
    return text


def run_agent_sync(
    session: AgentSession,
    user_text: str,
    history: list[dict],
    emit: Callable[[str, dict], None],
) -> None:
    cfg = load_llm_config()
    if not cfg.configured:
        emit("error", {"message": cfg.missing_reason})
        return

    s = _srv()
    project = s.pm.get_project(session.pid)
    if project is None:
        emit("error", {"message": "Project not found"})
        return
    if (project.get("tracking_mode") or "segmentation_tracking") == "pose_tracking":
        emit("error", {"message": "The agent currently supports segmentation tracking, not pose tracking."})
        return

    overview = project_overview(project, session.video_id, session.frame_idx)
    emit("status", {
        "phase": "started",
        "provider": cfg.provider,
        "model": cfg.model,
        "video_id": session.video_id,
        "frame_idx": session.frame_idx,
        "project_name": overview.get("project_name"),
        "video_count": overview.get("video_count"),
    })

    messages = build_initial_messages(user_text, overview, history)
    pending_images: list[dict] = []
    recorded_events: list[dict] = [{"event": "status", "data": {
        "phase": "started",
        "provider": cfg.provider,
        "model": cfg.model,
        "video_id": session.video_id,
        "frame_idx": session.frame_idx,
        "project_name": overview.get("project_name"),
        "video_count": overview.get("video_count"),
    }}]
    llm_turns: list[dict] = []
    inspect_images: list[dict] = []

    def emit_and_record(event: str, payload: dict) -> None:
        recorded_events.append({"event": event, "data": payload})
        emit(event, payload)

    def flush_dump() -> None:
        try:
            dump_dir = default_agent_dump_dir()
            summary = write_agent_context_dump(
                dump_dir,
                system_prompt=SYSTEM_PROMPT,
                overview=overview,
                user_text=user_text,
                messages=messages,
                events=recorded_events,
                llm_turns=llm_turns,
                cfg=cfg,
                inspect_images=inspect_images,
            )
            emit("context_dump", summary)
        except Exception:
            logger.exception("Failed to write agent context dump")

    for step in range(MAX_AGENT_STEPS):
        if session.cancel.is_set():
            emit_and_record("error", {"message": "Cancelled"})
            flush_dump()
            return
        if pending_images:
            inspect_images.extend(pending_images)
            messages.append({
                "role": "user",
                "content": "Visual: JPEG(s) of the inspected frame(s) are attached.",
                "images": list(pending_images),
            })
            pending_images = []
        prune_stale_inspect_images(messages)
        llm_turns.append({
            "step": step + 1,
            "inspect_jpegs_in_context": inspect_jpegs_in_messages(messages),
        })
        emit_and_record("status", {"phase": "thinking", "step": step + 1, "max_steps": MAX_AGENT_STEPS})
        try:
            llm = call_llm(cfg, messages)
        except Exception as e:
            logger.exception("LLM call failed")
            emit_and_record("error", {"message": f"LLM request failed: {e}"})
            flush_dump()
            return
        content = (llm.get("content") or "").strip()
        tool_calls = [tc for tc in (llm.get("tool_calls") or []) if tc.get("name")]
        if content:
            kind = "reasoning" if tool_calls else "message"
            emit_and_record(kind, {"text": content, "step": step + 1})

        if not tool_calls:
            if content:
                emit_and_record("done", {"text": content, "steps": step + 1})
            else:
                emit_and_record("done", {"text": "Done.", "steps": step + 1})
            flush_dump()
            return

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            if session.cancel.is_set():
                emit_and_record("error", {"message": "Cancelled"})
                flush_dump()
                return
            tname = tc["name"]
            targs = tc.get("arguments") or {}
            if not isinstance(targs, dict):
                targs = {}
            emit_and_record("tool_call", {
                "id": tc.get("id"),
                "name": tname,
                "arguments": targs,
                "step": step + 1,
            })
            if tname == "think":
                emit_and_record("reasoning", {"text": targs.get("thought") or "", "step": step + 1})

            t0 = time.time()
            result, ui_events = execute_tool(tname, targs, session)
            dt = int((time.time() - t0) * 1000)
            for ui in ui_events:
                emit_and_record("ui", ui)
            emit_and_record("tool_result", {
                "id": tc.get("id"),
                "name": tname,
                "ok": bool(result.get("ok", True)),
                "result": {k: v for k, v in result.items() if k != "masks"},
                "elapsed_ms": dt,
                "step": step + 1,
            })
            if session.pending_images:
                pending_images.extend(session.pending_images)
                session.pending_images.clear()
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or "tool",
                "name": tname,
                "content": _tool_result_for_llm(result),
            })

    emit_and_record("message", {"text": "Stopped after the step limit. Ask me to continue from here if needed."})
    emit_and_record("done", {"text": "Step limit reached.", "steps": MAX_AGENT_STEPS})
    flush_dump()


async def run_agent_sse(
    pid: str,
    video_id: Optional[str],
    frame_idx: int,
    message: str,
    history: list[dict],
):
    """Async generator of SSE dicts {event, data}."""
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    session = AgentSession(pid=pid, video_id=video_id, frame_idx=int(frame_idx or 0))
    st = get_agent_run_state(pid)
    if st.is_running:
        yield {"event": "error", "data": json.dumps({"message": "An agent run is already in progress for this project."})}
        return
    st.is_running = True
    st.cancel = session.cancel
    st.run_id = str(uuid.uuid4())

    def emit(event: str, payload: dict) -> None:
        loop.call_soon_threadsafe(q.put_nowait, {"event": event, "data": json.dumps(payload, default=str)})

    def worker():
        try:
            run_agent_sync(session, message, history, emit)
        except Exception as e:
            logger.exception("agent worker crashed")
            emit("error", {"message": str(e)})
        finally:
            emit("_end", {})

    threading.Thread(target=worker, name=f"agent-{pid[:8]}", daemon=True).start()
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=30)
            except asyncio.TimeoutError:
                yield {"event": "heartbeat", "data": "{}"}
                continue
            if item.get("event") == "_end":
                break
            yield item
    finally:
        st.is_running = False
        st.run_id = None
