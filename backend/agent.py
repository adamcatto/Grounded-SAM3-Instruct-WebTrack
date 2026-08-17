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

# A long anchor-labeling request can require several tool turns per frame.  Keep
# this configurable, but do not silently stop a normal every-N-frames job after
# a handful of anchors.
MAX_AGENT_STEPS = max(36, int(os.environ.get("AGENT_MAX_STEPS", "144")))
MAX_HISTORY_TURNS = 16
INSPECT_MAX_EDGE = 640
DEFAULT_NEARBY_OFFSET = 20
CONFIDENCE_RETRY_THRESHOLD = 0.45
# Home-cage red-light layout on frame 160 (and similar): mice are the two
# darkest interior blobs (~cx 0.32 left, ~cx 0.66 right). The circular
# water-bottle port sits on the far-right wall (~cx 0.76+). Empty bedding
# is the gap between the animals, not the right-hand mouse.
WATER_PORT_MIN_CX = 0.74
BEDDING_GAP_CX = (0.42, 0.56)
RIGHT_ANIMAL_MIN_CX = 0.55
DARK_LUM_THR = 50
DARK_MIN_AREA = 2000
DARK_MATCH_IOU = 0.20
# Inspect JPEGs stay in the chat until pruned. Keep the last two so the model can
# compare the current frame with the previous one without blowing the context.
KEEP_INSPECT_IMAGES = 2
# Qwen3 Thinking uses the completion budget for its hidden reasoning *and* the
# tool call.  At 2048 it can exhaust the entire response before producing any
# visible content/tool call, which used to be reported as a successful "Done".
# The server defaults to a 64k context.  Give Thinking models enough room to
# reason through a long multi-anchor request before emitting their first tool
# call; deployment can still lower/raise this with AGENT_MAX_TOKENS.
AGENT_MAX_TOKENS = max(512, int(os.environ.get("AGENT_MAX_TOKENS", "32768")))

TEXT_SEGMENTATION_POLICY = """## Current agent execution policy (mandatory)
- Use text-based concept segmentation only. Do not call add_point_prompt or invent click coordinates.
- A text segmentation persists the concept-grounded mask directly; it does not create point prompts.
- For requests to segment every N frames, call segment_text_interval once. Do not loop over frames with separate LLM turns.
- For anchor labeling, call set_start_frame if needed, then call anchor_frame_labeling_loop once. Do not label anchor frames one at a time.
- Use text_segment for a one-frame request. Evaluate results when useful, but do not try point-based repairs.
- Direct text masks are annotation-only for now; do not start propagation unless a future workflow explicitly provides tracker seeds."""


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


def _count_large_mask_components(binary, min_frac: float = 0.12) -> int:
    """How many sizable connected components a mask has (body+tail split → 2)."""
    b = np.squeeze(np.asarray(binary)).astype(bool)
    if b.ndim != 2:
        return 0
    area = int(b.sum())
    if area == 0:
        return 0
    try:
        import cv2
        n, _, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    except Exception:
        return 1
    thresh = max(250, int(min_frac * area))
    return sum(1 for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) >= thresh)


def _mask_iou(a, b) -> float:
    aa = np.squeeze(np.asarray(a)).astype(bool)
    bb = np.squeeze(np.asarray(b)).astype(bool)
    if aa.ndim != 2 or bb.ndim != 2 or aa.shape != bb.shape:
        return 0.0
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    return float(inter / union) if union else 0.0


def dark_animal_blobs(frame_path, lum_thr: float = DARK_LUM_THR, min_area: int = DARK_MIN_AREA) -> list[dict]:
    """Two (or more) dark interior connected components — the mice, not the cage wall."""
    try:
        from PIL import Image
        import cv2
    except Exception:
        return []
    try:
        arr = np.asarray(Image.open(frame_path).convert("RGB"))
    except Exception:
        return []
    h, w = arr.shape[:2]
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    inset = (xx > 0.12 * w) & (xx < 0.92 * w) & (yy > 0.18 * h) & (yy < 0.88 * h)
    binary = ((lum < lum_thr) & inset).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs: list[dict] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        blobs.append({
            "mask": labels == i,
            "cx": float(centroids[i, 0]) / w,
            "cy": float(centroids[i, 1]) / h,
            "area": area,
        })
    blobs.sort(key=lambda b: -b["area"])
    blobs = blobs[:4]
    blobs.sort(key=lambda b: b["cx"])
    return blobs


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
    n_comp = _count_large_mask_components(binary)
    if n_comp >= 2:
        conf *= 0.4
        reasons.append("split_components")

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
        leftish = [d for d in ranked if _bbox_center_xy(d)[0] < RIGHT_ANIMAL_MIN_CX]
        return (leftish or ranked)[0]
    if side == "right":
        on_mouse = [
            d for d in ranked
            if RIGHT_ANIMAL_MIN_CX <= _bbox_center_xy(d)[0] < WATER_PORT_MIN_CX
        ]
        if on_mouse:
            return on_mouse[-1]
        not_port = [d for d in ranked if _bbox_center_xy(d)[0] < WATER_PORT_MIN_CX]
        return (not_port or ranked)[-1]
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


def _row_center_x(row: dict) -> Optional[float]:
    bbox = row.get("bbox_xywh_norm") or []
    if len(bbox) < 4:
        return None
    try:
        return float(bbox[0]) + float(bbox[2]) / 2.0
    except (TypeError, ValueError):
        return None


def _append_eval_reason(row: dict, extra: str) -> None:
    row["ok"] = False
    reason = str(row.get("reason") or "ok")
    if extra not in reason:
        row["reason"] = extra if reason in ("ok", "") else f"{reason},{extra}"


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
                    _append_eval_reason(p, extra)
    if expected >= 2:
        gap_lo, gap_hi = BEDDING_GAP_CX
        for p in with_bbox:
            cx = _row_center_x(p)
            if cx is None:
                continue
            if gap_lo <= cx <= gap_hi:
                _append_eval_reason(p, "likely_bedding")
            if cx >= WATER_PORT_MIN_CX:
                _append_eval_reason(p, "likely_water_port")
        named = []
        for p in with_bbox:
            side = _prefer_side_from_text_and_object("", {"name": p.get("name") or ""})
            cx = _row_center_x(p)
            if side and cx is not None:
                named.append((side, cx, p))
        lefts = [r for r in named if r[0] == "left"]
        rights = [r for r in named if r[0] == "right"]
        if lefts and rights and lefts[0][1] > rights[0][1]:
            _append_eval_reason(lefts[0][2], "identity_swap")
            _append_eval_reason(rights[0][2], "identity_swap")
    masked = sum(1 for p in out if (p.get("area_px") or 0) > 0)
    any_bad = any(not p.get("ok") for p in out)
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
            if expected > 0 and masked < expected
            else (
                "Masks failed quality checks (bedding blob, identity swap, or overlap). "
                "Re-inspect and click each animal's torso; HeadShave/shaved is the far-side "
                "mouse, NoShave is the other dark blob — not empty bedding."
                if any_bad
                else None
            )
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
        "name": "set_start_frame",
        "description": (
            "Set the video's start frame for anchor-frame labeling. This recomputes "
            "the anchor grid from that frame using the video's configured anchor interval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_frame": {"type": "integer", "description": "0-indexed first anchor frame"},
                "video_id": {"type": "string"},
            },
            "required": ["start_frame"],
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
        "name": "segment_text_interval",
        "description": (
            "Native batch operation for a regular frame grid. Use this whenever the user "
            "asks to segment one text-described object every N frames. It performs "
            "goto_frame → text_segment → evaluate_segmentation for every planned frame "
            "inside one tool call, including the final video frame if requested. It never "
            "uses point prompts and does not start propagation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text concept to segment, e.g. 'dark mouse'"},
                "start_frame": {"type": "integer"},
                "interval": {"type": "integer", "description": "Frame spacing; must be positive"},
                "object_id": {"type": "string"},
                "object_name": {"type": "string", "description": "Existing/new display name if object_id is omitted"},
                "video_id": {"type": "string"},
                "include_last": {"type": "boolean", "default": True},
                "evaluate": {"type": "boolean", "default": True},
                "min_score": {"type": "number", "default": 0.0},
            },
            "required": ["text", "start_frame", "interval"],
        },
    },
    {
        "name": "anchor_frame_labeling_loop",
        "description": (
            "Native anchor-labeling batch operation. It uses the video's configured "
            "start_frame and anchor interval, then for each anchor performs "
            "goto_frame → text_segment → evaluate_segmentation → commit_anchor. "
            "Use this after set_start_frame when the user asks to label anchor frames. "
            "It uses text segmentation only and never creates point prompts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text concept to segment at every anchor"},
                "object_id": {"type": "string"},
                "object_name": {"type": "string", "description": "Existing/new display name if object_id is omitted"},
                "video_id": {"type": "string"},
                "evaluate": {"type": "boolean", "default": True},
                "min_score": {"type": "number", "default": 0.0},
            },
            "required": ["text"],
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
   c. Prefer text_segment with a simple visual phrase like "dark mouse" once per identity (reuse HeadShave / NoShave ids). SAM3 often returns both dark animals; pick the left detection for NoShave and the right-of-center detection for HeadShave.
   d. evaluate_segmentation. If confidence is low, the frame is empty, identities are swapped, or animals overlap badly: think, then try nearby frames (±20, then ±40) via plan_frames(around_frame=..., nearby_offset=20). If the result is incomplete / missing_mask, stay on this frame and segment the remaining animals.
   e. If text segmentation is weak but you can see the animal, add_point_prompt at a chest/back point (positive=1) on the dark blob itself. Use a negative point (0) on the *other* animal if they touch — never on the same animal's tail.
   f. commit_anchor when the frame is a planned grid/anchor frame and the masks look right.
4. Adapt. Do not blindly march the grid if a frame is unusable. Skip to a clearer neighbor, then continue the plan.
5. After the requested frames are labeled, start_propagation if the user wants tracking. Do not start it for a single-frame-only request.

## Text vs points
- Prefer text_segment when the animals are the darkest blobs in the cage. A short prompt ("dark mouse") is enough — do not invent coordinates.
- Prefer inspect_frame + add_point_prompt when animals look similar, are overlapping, or text confidence is low.
- Inspect the frame before clicking. Choose points from the dark animal bodies you see in the JPEG — do not invent coordinates from the user's words, and do not write pixel locations into the user-facing reply.
- If the user says there are N mice/animals, that count is ground truth. Reuse existing objects with matching names (HeadShave / NoShave, etc.) instead of creating extras. Call text_segment or add_point_prompt once per identity. Do not finish until evaluate_segmentation reports ok=true, masked_object_count >= N, and no missing_mask / identity_swap / likely_bedding / likely_water_port / not_on_animal. One mask is a failure.
- If evaluate_segmentation returns incomplete, missing_mask, identity_swap, likely_bedding, likely_water_port, or not_on_animal, stay on this frame and re-click the dark blob. Do not jump to a nearby frame unless the current frame is empty or unusable.
- In top-down red-light home cages the two mice are dark blobs on the bedding: unshaved on the left, shaved to the right of center. The circular water-bottle port on the far-right WALL is not a mouse — never click it. Empty bedding is the gap between the two animals, not the right-hand mouse.
- When using add_point_prompt, place a new positive point on that animal's torso from the inspect JPEG. Do not copy another object's point list.
- One animal = one connected mask that includes head, body, and tail. Never put a negative click on the same mouse's tail (that splits body and tail into two blobs). If evaluate_segmentation reports split_components, add another positive point on the missing part (usually the tail), do not create a new object.

## Style
- Call think before multi-step plans and after failures.
- Keep user-facing messages short. Put detail in think / tool results.
- Do not reset or delete the user's existing objects unless they ask.
- Stay on the video the user specified; if they said "this video" use the current one.
- If the tracking mode is pose_tracking, say you only handle segmentation/tracking objects here.
"""


# ─── System-prompt templates ──────────────────────────────────────────────────
# The default (SYSTEM_PROMPT above) is tuned for red-light home-cage two-mouse
# videos. That specialization is wrong for other footage, so the user can pick a
# template and edit it. Templates are read-only starting points; the chosen /
# edited text is persisted per project in config as `agent_system_prompt`.

_CORE_WORKFLOW = """## What you know
- The project may contain many videos. Use get_project_overview if unsure which video you are on, how many exist, or their names.
- Each video has a frame count, fps, start_frame, objects (id, name, description), saved point prompts, and optional propagation state.
- Frames are 0-indexed. Call goto_frame / inspect_frame so the backend extracts the frame before SAM can run.
- Object ids are strings ("1", "2", …); names are display labels. Create them with create_object and keep using the returned id.
- SAM3 tracker mode stores point prompts. text_segment finds a mask, converts it to points, and persists them so tracking/propagation behaves normally.
- Propagation needs at least one point prompt. After labeling the planned frames, call start_propagation only if the user asked to track.

## How to work
1. Orient: get_project_overview (and get_video_details for the active video).
2. Plan: if the user says "every Nth frame", call plan_frames (typical interval 1000 or the video's anchor_batch_size).
3. For each planned frame:
   a. goto_frame then inspect_frame so you can see the subjects, occlusion, blur, and identity cues.
   b. create_object once per identity (reuse existing objects with the same name).
   c. Prefer text_segment with a short visual phrase per subject; otherwise inspect_frame + add_point_prompt on the subject's body.
   d. evaluate_segmentation. If confidence is low or the frame is unusable, think, then try nearby frames via plan_frames(around_frame=..., nearby_offset=20).
   e. commit_anchor when a planned grid/anchor frame looks right.
4. Adapt. Do not blindly march the grid if a frame is unusable — skip to a clearer neighbor, then continue.
5. After the requested frames are labeled, start_propagation only if the user wants tracking. Do not start it for a single-frame-only request.

## Text vs points
- Prefer text_segment when the subject is visually distinctive with a short phrase. Do not invent coordinates from words.
- Prefer inspect_frame + add_point_prompt when subjects look similar, overlap, or text confidence is low. Choose points from what you see in the JPEG.
- If the user states there are N subjects, that count is ground truth. Label once per identity; do not finish until evaluate_segmentation reports ok=true and masked_object_count >= N.
- One subject = one connected mask. If evaluate_segmentation reports split_components, add a positive point on the missing part; do not create a new object.

## Style
- Call think before multi-step plans and after failures.
- Keep user-facing messages short; put detail in think / tool results.
- Do not reset or delete the user's existing objects unless they ask.
- Stay on the video the user specified; if they said "this video" use the current one.
- If the tracking mode is pose_tracking, say you only handle segmentation/tracking objects here.
"""

_TWO_ANIMALS_PROMPT = (
    "You are the SAM3 Web Tracker agent. You annotate and track objects in videos by calling tools. "
    "The user watches a live chat of your reasoning while the canvas updates.\n\n"
    "This project contains two distinct animals — one object per animal identity. Reuse the same two "
    "object ids across every frame; never create extra objects. Use a negative point on the *other* "
    "animal only when the two touch, never on the same animal's own body or tail.\n\n"
    + _CORE_WORKFLOW
)

_SINGLE_ANIMAL_PROMPT = (
    "You are the SAM3 Web Tracker agent. You annotate and track objects in videos by calling tools. "
    "The user watches a live chat of your reasoning while the canvas updates.\n\n"
    "This project has a single subject/animal. Use one object id across all frames. Do not create a "
    "second object unless the user explicitly asks.\n\n"
    + _CORE_WORKFLOW
)

_DARK_SUBJECTS_PROMPT = (
    "You are the SAM3 Web Tracker agent. You annotate and track objects in videos by calling tools. "
    "The user watches a live chat of your reasoning while the canvas updates.\n\n"
    "The subjects are the darkest blobs against a lighter background. Prefer text_segment with a short "
    "phrase like 'dark <subject>'. Place positive points on the dark body; use negative points on other "
    "dark blobs only when they touch the target. Ignore dark fixtures on walls/edges that are not subjects.\n\n"
    + _CORE_WORKFLOW
)

_MRI_TISSUE_PROMPT = (
    "You are the SAM3 Web Tracker agent, segmenting medical imaging frames (e.g., MRI/CT slices) by "
    "calling tools. Make no animal or behavior assumptions.\n\n"
    "Segment the resected tissue / cavity region the user describes. Treat each distinct region as one "
    "object. Prefer inspect_frame + add_point_prompt on the region interior, with negative points on "
    "adjacent healthy tissue to tighten the boundary; use text_segment only when a clear phrase applies. "
    "Propagate across slices/frames only if the user asks.\n\n"
    + _CORE_WORKFLOW
)

_BLANK_PROMPT = (
    "You are the SAM3 Web Tracker agent. You annotate and track objects in videos by calling tools; the "
    "user watches your reasoning and the canvas updates live.\n\n"
    "Follow the user's instructions. Useful tools: get_project_overview, get_video_details, goto_frame, "
    "inspect_frame, create_object, text_segment, add_point_prompt, evaluate_segmentation, plan_frames, "
    "commit_anchor, start_propagation. Frames are 0-indexed; object ids are strings. Segment only what the "
    "user asks and start propagation only when they request tracking."
)

DEFAULT_TEMPLATE_ID = "home_cage_two_mice"

SYSTEM_PROMPT_TEMPLATES: list[dict] = [
    {
        "id": "home_cage_two_mice",
        "name": "Home-cage: two dark mice",
        "description": "Top-down red-light home cage with two dark mice (shaved vs unshaved). The current default.",
        "content": SYSTEM_PROMPT,
    },
    {
        "id": "two_animals",
        "name": "Two animals (generic)",
        "description": "Two distinct animals, one object per identity. No cage/lighting assumptions.",
        "content": _TWO_ANIMALS_PROMPT,
    },
    {
        "id": "single_animal",
        "name": "Single animal",
        "description": "One subject tracked across the whole video.",
        "content": _SINGLE_ANIMAL_PROMPT,
    },
    {
        "id": "dark_subjects",
        "name": "Dark subjects on light background",
        "description": "Subjects are the darkest blobs; favor text_segment 'dark <subject>'.",
        "content": _DARK_SUBJECTS_PROMPT,
    },
    {
        "id": "mri_resected_tissue",
        "name": "MRI / resected tissue",
        "description": "Medical imaging: segment resected tissue / cavity regions, no animal language.",
        "content": _MRI_TISSUE_PROMPT,
    },
    {
        "id": "blank",
        "name": "Blank (start from scratch)",
        "description": "Minimal generic prompt to write your own instructions.",
        "content": _BLANK_PROMPT,
    },
]


def resolve_system_prompt(project: Optional[dict]) -> str:
    """Resolve the agent system prompt for a project.

    - Never configured (key absent): fall back to the built-in default so legacy
      projects keep working.
    - Configured (even to an empty string via the "Custom (blank)" option): use
      the stored text verbatim. An empty result means no system message is sent,
      leaving only the model's own hidden default in effect.
    """
    if not project or "agent_system_prompt" not in project:
        return SYSTEM_PROMPT
    return (project.get("agent_system_prompt") or "").strip()


def build_video_metadata_block(project: Optional[dict]) -> str:
    """Concise, always-on project/video metadata for the system prompt.

    Hidden from the user-facing editor. Gives the agent video names and basic
    properties (frame count, fps, resolution, objects) so it does not need to
    inspect frame contents just to know what videos exist.
    """
    if not project:
        return ""
    videos = project.get("videos") or {}
    lines = [
        "## Project metadata (authoritative — do not re-derive by inspecting frames)",
        (
            f"Project: {project.get('name')!r} (id={project.get('id')}) · "
            f"tracking_mode={project.get('tracking_mode') or 'segmentation_tracking'} · "
            f"{len(videos)} video(s)"
        ),
    ]
    if not videos:
        lines.append("- (no videos yet)")
    for vid, v in videos.items():
        name = v.get("name") or vid
        num_frames = v.get("num_frames")
        fps = v.get("fps")
        width, height = v.get("width"), v.get("height")
        parts: list[str] = [f"{num_frames} frames" if num_frames is not None else "? frames"]
        if fps:
            parts.append(f"{fps} fps")
            try:
                if num_frames:
                    parts.append(f"{num_frames / float(fps):.1f}s")
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        if width and height:
            parts.append(f"{width}x{height}")
        parts.append(f"start_frame={v.get('start_frame', 0)}")
        parts.append(f"{len(v.get('objects') or {})} object(s)")
        if v.get("propagation_complete"):
            parts.append("propagated")
        lines.append(f"- {name} [id={vid}]: " + ", ".join(parts))
    return "\n".join(lines)


# ─── Run state ────────────────────────────────────────────────────────────────


@dataclass
class AgentSession:
    pid: str
    video_id: Optional[str]
    frame_idx: int
    cancel: threading.Event = field(default_factory=threading.Event)
    pending_images: list[dict] = field(default_factory=list)
    ui_events: list[dict] = field(default_factory=list)
    # Set only while a native batch tool is running.  It streams child tool
    # traces/UI events instead of buffering them until the batch completes.
    progress: Optional[Callable[[str, dict], None]] = None
    # The model-selected outer tool. Native child phases inherit this id so
    # traces can distinguish deterministic implementation from LLM choices.
    active_tool_id: Optional[str] = None

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


def _flush_native_ui(session: AgentSession) -> None:
    """Immediately forward UI mutations created inside a native batch tool."""
    if session.progress is None:
        return
    for ui in session.ui_events:
        session.progress("ui", ui)
    session.ui_events.clear()


def _run_native_phase(
    session: AgentSession,
    *,
    phase_id: str,
    name: str,
    arguments: dict,
    action: Callable[[], dict],
) -> dict:
    """Run and visibly trace one child operation of a native batch tool."""
    if session.cancel.is_set():
        raise InterruptedError("Cancelled")
    if session.progress is not None:
        session.progress("tool_call", {
            "id": phase_id, "name": name, "arguments": arguments,
            "parent_id": session.active_tool_id, "deterministic": True,
        })
    t0 = time.time()
    result = action()
    _flush_native_ui(session)
    if session.progress is not None:
        session.progress("tool_result", {
            "id": phase_id,
            "name": name,
            "parent_id": session.active_tool_id,
            "deterministic": True,
            "ok": bool(result.get("ok", True)),
            "result": {k: v for k, v in result.items() if k != "masks"},
            "elapsed_ms": int((time.time() - t0) * 1000),
        })
    return result


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


def _text_segment_impl(
    session: AgentSession, vid: str, frame_idx: int, text: str,
    object_id: Optional[str], object_name: Optional[str], min_score: float,
    *, ensure_frame: bool = True,
) -> dict:
    s = _srv()
    from video_processor import encode_mask_as_png

    if ensure_frame:
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
    suggestion = None
    cx = picked_xy[0]
    if picked_side == "right" and cx >= WATER_PORT_MIN_CX:
        quality = {
            **quality,
            "ok": False,
            "reason": (
                "likely_water_port"
                if quality.get("reason") in (None, "", "ok")
                else f"{quality.get('reason')},likely_water_port"
            ),
            "retry_nearby": False,
        }
        suggestion = (
            "That hit is the circular water-bottle port on the far-right wall, not a mouse. "
            "text_segment again with 'dark mouse' or add_point_prompt on the dark blob "
            "to the left of the port (right-of-center on the bedding)."
        )

    bind_ok = obj is not None and suggestion is None
    persisted_masks: dict[str, str] = {}
    if bind_ok:
        # Persist the concept-grounded mask directly.  Do not sample/replay point
        # prompts: synthetic points made the agent over-eager and can produce a
        # materially different mask than the text segmentation that selected it.
        try:
            s._persist_predicted_anchor_frame(
                session.pid,
                vid,
                frame_idx,
                {s._to_sam_obj_id_from_npz_key(str(obj["id"])): np.squeeze(best["mask"]).astype(np.uint8)},
                objects,
            )
            persisted_masks = s._get_encoded_masks(session.pid, vid, frame_idx, objects)
            session.emit_ui("goto_frame", video_id=vid, frame_idx=frame_idx)
            session.emit_ui("select_object", object_id=str(obj["id"]), video_id=vid)
            session.emit_ui("set_masks", video_id=vid, frame_idx=frame_idx, masks=persisted_masks)
            session.emit_ui("refresh_project")
        except Exception as e:
            logger.exception("Could not persist text-segmented mask")
            return {**quality, "ok": False, "error": f"persist_text_mask_failed: {e}", "frame_idx": frame_idx}
        color = obj.get("color") or "#5B8DD9"
    else:
        color = (obj or {}).get("color") or "#5B8DD9"

    mask_b64 = encode_mask_as_png(best["mask"], color)
    extra = [_detection_summary(d) for d in detections]

    return {
        "ok": quality["ok"] and suggestion is None,
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
        "seed_points": False,
        "masks": persisted_masks or ({str(obj["id"]): mask_b64} if bind_ok else {"det": mask_b64}),
        "retry_nearby": quality.get("retry_nearby"),
        "suggestion": suggestion,
    }


def _segment_text_interval_impl(
    session: AgentSession,
    vid: str,
    *,
    text: str,
    start_frame: int,
    interval: int,
    object_id: Optional[str],
    object_name: Optional[str],
    include_last: bool,
    evaluate: bool,
    min_score: float,
) -> dict:
    """Text-segment a regular grid without requiring an LLM turn per frame."""
    s = _srv()
    video = s.pm.get_video(session.pid, vid)
    if video is None:
        raise ValueError(f"Video {vid} not found")
    if interval <= 0:
        raise ValueError("interval must be positive")
    if not object_id and not object_name:
        raise ValueError("Provide object_id or object_name so the text masks can be saved")

    frames = plan_sample_frames(start_frame, int(video.get("num_frames") or 0), interval, include_last)
    results: list[dict] = []
    saved_object_id = object_id
    for frame_idx in frames:
        prefix = f"interval-{frame_idx}"
        _run_native_phase(
            session, phase_id=f"{prefix}-goto", name="goto_frame",
            arguments={"video_id": vid, "frame_idx": frame_idx},
            action=lambda f=frame_idx: {"ok": True, **_ensure_frame(session, vid, f)},
        )
        segment = _run_native_phase(
            session, phase_id=f"{prefix}-text", name="text_segment",
            arguments={"video_id": vid, "frame_idx": frame_idx, "text": text, "object_id": saved_object_id, "object_name": object_name},
            action=lambda f=frame_idx: _text_segment_impl(
                session, vid, f, text, saved_object_id, object_name, min_score, ensure_frame=False,
            ),
        )
        if segment.get("object"):
            saved_object_id = str(segment["object"].get("id") or saved_object_id or "")
        evaluation = None
        if evaluate and segment.get("ok"):
            evaluation = _run_native_phase(
                session, phase_id=f"{prefix}-evaluate", name="evaluate_segmentation",
                arguments={"video_id": vid, "frame_idx": frame_idx},
                action=lambda f=frame_idx: _execute_tool_inner(
                    "evaluate_segmentation", {"video_id": vid, "frame_idx": f}, session,
                ),
            )
        results.append({
            "frame_idx": frame_idx,
            "ok": bool(segment.get("ok")) and (evaluation is None or bool(evaluation.get("ok"))),
            "segment_ok": bool(segment.get("ok")),
            "evaluation_ok": evaluation.get("ok") if evaluation is not None else None,
            "reason": ((segment.get("quality") or {}).get("reason") or segment.get("error") or
                       ((evaluation or {}).get("note")) or "ok"),
            "method": segment.get("method"),
        })

    succeeded = sum(1 for item in results if item["ok"])
    return {
        "ok": succeeded == len(results),
        "video_id": vid,
        "object_id": saved_object_id,
        "text": text,
        "start_frame": frames[0] if frames else None,
        "interval": interval,
        "frames_processed": len(results),
        "frames_succeeded": succeeded,
        "frames_failed": len(results) - succeeded,
        "results": results,
        "note": "Text masks were persisted directly; no point prompts were created.",
    }


def _anchor_frame_labeling_loop_impl(
    session: AgentSession,
    vid: str,
    *,
    text: str,
    object_id: Optional[str],
    object_name: Optional[str],
    evaluate: bool,
    min_score: float,
) -> dict:
    """Text-label and commit every configured anchor without LLM turns between frames."""
    s = _srv()
    from anchor_helpers import compute_anchor_frames, video_anchor_batch_size

    video = s.pm.get_video(session.pid, vid)
    if video is None:
        raise ValueError(f"Video {vid} not found")
    if not object_id and not object_name:
        raise ValueError("Provide object_id or object_name so the text masks can be saved")
    start = int(video.get("start_frame") or 0)
    interval = video_anchor_batch_size(video)
    anchors = compute_anchor_frames(start, int(video.get("num_frames") or 0), interval)
    results: list[dict] = []
    saved_object_id = object_id

    for anchor_index, frame_idx in enumerate(anchors):
        prefix = f"anchor-{anchor_index}-{frame_idx}"
        _run_native_phase(
            session, phase_id=f"{prefix}-goto", name="goto_frame",
            arguments={"video_id": vid, "frame_idx": frame_idx},
            action=lambda f=frame_idx: {"ok": True, **_ensure_frame(session, vid, f)},
        )
        segment = _run_native_phase(
            session, phase_id=f"{prefix}-text", name="text_segment",
            arguments={"video_id": vid, "frame_idx": frame_idx, "text": text, "object_id": saved_object_id, "object_name": object_name},
            action=lambda f=frame_idx: _text_segment_impl(
                session, vid, f, text, saved_object_id, object_name, min_score, ensure_frame=False,
            ),
        )
        if segment.get("object"):
            saved_object_id = str(segment["object"].get("id") or saved_object_id or "")
        evaluation = None
        if evaluate and segment.get("ok"):
            evaluation = _run_native_phase(
                session, phase_id=f"{prefix}-evaluate", name="evaluate_segmentation",
                arguments={"video_id": vid, "frame_idx": frame_idx},
                action=lambda f=frame_idx: _execute_tool_inner(
                    "evaluate_segmentation", {"video_id": vid, "frame_idx": f}, session,
                ),
            )
        accepted = bool(segment.get("ok")) and (evaluation is None or bool(evaluation.get("ok")))
        if accepted:
            _run_native_phase(
                session, phase_id=f"{prefix}-commit", name="commit_anchor",
                arguments={"video_id": vid, "frame_idx": frame_idx, "anchor_index": anchor_index},
                action=lambda f=frame_idx, idx=anchor_index: _execute_tool_inner(
                    "commit_anchor", {"video_id": vid, "frame_idx": f, "anchor_index": idx}, session,
                ),
            )
        results.append({
            "frame_idx": frame_idx,
            "anchor_index": anchor_index,
            "ok": accepted,
            "committed": accepted,
            "segment_ok": bool(segment.get("ok")),
            "evaluation_ok": evaluation.get("ok") if evaluation is not None else None,
            "reason": ((segment.get("quality") or {}).get("reason") or segment.get("error") or
                       ((evaluation or {}).get("note")) or "ok"),
        })

    succeeded = sum(1 for item in results if item["ok"])
    return {
        "ok": succeeded == len(results),
        "video_id": vid,
        "object_id": saved_object_id,
        "text": text,
        "start_frame": start,
        "anchor_interval": interval,
        "anchor_count": len(results),
        "anchors_committed": succeeded,
        "anchors_failed": len(results) - succeeded,
        "results": results,
        "note": "Successful anchor masks were committed directly; no point prompts were created.",
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

    if name == "set_start_frame":
        vid, video = _resolve_video(session, args.get("video_id"))
        n = int(video.get("num_frames") or 0)
        if n <= 0:
            raise ValueError("Cannot set an anchor start frame on an empty video")
        start = int(args["start_frame"])
        if not 0 <= start < n:
            raise ValueError(f"start_frame must be between 0 and {n - 1}")
        updated = s.pm.update_video(session.pid, vid, {
            "start_frame": start,
            "anchor_labeling_complete": False,
        })
        batch = video_anchor_batch_size(updated)
        anchors = compute_anchor_frames(start, n, batch)
        session.video_id = vid
        session.frame_idx = start
        session.emit_ui("goto_frame", video_id=vid, frame_idx=start)
        session.emit_ui("set_anchor_phase", video_id=vid, frames=anchors, current_index=0)
        session.emit_ui("refresh_project")
        return {
            "ok": True,
            "video_id": vid,
            "start_frame": start,
            "anchor_interval": batch,
            "anchor_frames": anchors,
            "anchor_count": len(anchors),
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

    if name == "segment_text_interval":
        vid, _video = _resolve_video(session, args.get("video_id"))
        return _segment_text_interval_impl(
            session,
            vid,
            text=str(args.get("text") or ""),
            start_frame=int(args["start_frame"]),
            interval=int(args["interval"]),
            object_id=args.get("object_id"),
            object_name=args.get("object_name"),
            include_last=bool(args.get("include_last", True)),
            evaluate=bool(args.get("evaluate", True)),
            min_score=float(args.get("min_score") or 0.0),
        )

    if name == "anchor_frame_labeling_loop":
        vid, _video = _resolve_video(session, args.get("video_id"))
        return _anchor_frame_labeling_loop_impl(
            session,
            vid,
            text=str(args.get("text") or ""),
            object_id=args.get("object_id"),
            object_name=args.get("object_name"),
            evaluate=bool(args.get("evaluate", True)),
            min_score=float(args.get("min_score") or 0.0),
        )

    if name == "add_point_prompt":
        raise ValueError(
            "Point prompts are disabled for the agent. Use text_segment or "
            "segment_text_interval instead."
        )

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
        jpg = s.pm.annotated_frames_dir(session.pid, vid) / f"{fidx:06d}.jpg"
        if not jpg.exists():
            jpg = s.pm.frames_dir(session.pid, vid) / f"{fidx:06d}.jpg"
        blobs = dark_animal_blobs(jpg) if jpg.exists() else []
        if len(blobs) >= 2:
            by_area = sorted(blobs, key=lambda b: -b["area"])[:2]
            left_b, right_b = sorted(by_area, key=lambda b: b["cx"])
            dense_by_id = {str(k): v for k, v in (dense or {}).items()}
            for q in per_obj:
                mask = dense_by_id.get(str(q.get("object_id")))
                if mask is None:
                    continue
                i_l = _mask_iou(mask, left_b["mask"])
                i_r = _mask_iou(mask, right_b["mask"])
                q["dark_iou_left"] = round(i_l, 4)
                q["dark_iou_right"] = round(i_r, 4)
                best = max(i_l, i_r)
                if best < DARK_MATCH_IOU:
                    _append_eval_reason(q, "not_on_animal")
                    continue
                side = _prefer_side_from_text_and_object("", {"name": q.get("name") or ""})
                matched_right = i_r > i_l
                if side == "right" and not matched_right:
                    _append_eval_reason(q, "identity_swap")
                if side == "left" and matched_right:
                    _append_eval_reason(q, "identity_swap")
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


def call_llm(cfg: LLMConfig, messages: list[dict], on_delta: Optional[Callable[[str, str], None]] = None) -> dict:
    """One chat turn. Returns {content, tool_calls: [{id, name, arguments}], usage}.

    If on_delta is given (OpenAI-compatible / vLLM only), the response is streamed
    and on_delta(kind, text) is called for each token chunk, where kind is
    "reasoning" or "content".
    """
    if cfg.provider == "anthropic":
        return _call_anthropic(cfg, messages)
    return _call_openai(cfg, messages, on_delta=on_delta)


def probe_max_context(cfg: LLMConfig) -> Optional[int]:
    """Best-effort model context window (tokens) from an OpenAI-compatible /models."""
    if cfg.provider == "anthropic":
        return None
    base = _normalize_openai_base(cfg.base_url or "", cfg.provider)
    payload = _http_get_json(base + "/models", timeout=1.5)
    for item in (payload or {}).get("data") or []:
        if isinstance(item, dict):
            ml = item.get("max_model_len") or item.get("max_context_length")
            if ml:
                try:
                    return int(ml)
                except (TypeError, ValueError):
                    pass
    return None


def _message_image_parts_openai(images: list[dict]) -> list[dict]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{im['jpeg_b64']}"},
        }
        for im in images
        if im.get("jpeg_b64")
    ]


def _call_openai(cfg: LLMConfig, messages: list[dict], on_delta: Optional[Callable[[str, str], None]] = None) -> dict:
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

    if on_delta is not None:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        return _call_openai_stream(url, headers, payload, timeout, on_delta)

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
    return {
        "content": content,
        "tool_calls": tool_calls,
        "usage": data.get("usage"),
        "finish_reason": choice.get("finish_reason"),
    }


def _call_openai_stream(
    url: str, headers: dict, payload: dict, timeout: float,
    on_delta: Callable[[str, str], None],
) -> dict:
    """Stream an OpenAI-compatible chat completion, invoking on_delta per chunk."""
    import httpx

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_by_index: dict[int, dict] = {}
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None

    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw[6:] if raw.startswith("data: ") else raw
                line = line.strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                delta = choices[0].get("delta") or {}
                rc = delta.get("reasoning_content")
                if rc:
                    reasoning_parts.append(rc)
                    on_delta("reasoning", rc)
                c = delta.get("content")
                if c:
                    content_parts.append(c)
                    on_delta("content", c)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_by_index.setdefault(idx, {"id": None, "name": None, "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    if reasoning and reasoning not in content:
        content = f"{reasoning}\n{content}".strip()
    tool_calls = []
    for idx in sorted(tool_by_index):
        slot = tool_by_index[idx]
        if not slot["name"]:
            continue
        try:
            parsed = json.loads(slot["args"] or "{}")
        except json.JSONDecodeError:
            parsed = {}
        tool_calls.append({"id": slot["id"] or str(uuid.uuid4()), "name": slot["name"], "arguments": parsed})
    return {
        "content": content,
        "tool_calls": tool_calls,
        "usage": usage,
        "finish_reason": finish_reason,
    }


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
    system_prompt: str = SYSTEM_PROMPT,
    metadata_block: str = "",
) -> list[dict]:
    # Effective system message = user-editable prompt + always-on hidden metadata.
    system_content = (system_prompt or "").strip()
    system_parts = [system_content, TEXT_SEGMENTATION_POLICY]
    if metadata_block:
        system_parts.append(metadata_block)
    system_content = "\n\n".join(part for part in system_parts if part).strip()
    msgs: list[dict] = []
    if system_content:
        msgs.append({"role": "system", "content": system_content})
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
    system_prompt = resolve_system_prompt(project)
    metadata_block = build_video_metadata_block(project)
    max_context = probe_max_context(cfg)
    emit("status", {
        "phase": "started",
        "provider": cfg.provider,
        "model": cfg.model,
        "video_id": session.video_id,
        "frame_idx": session.frame_idx,
        "project_name": overview.get("project_name"),
        "video_count": overview.get("video_count"),
        "max_context": max_context,
    })

    messages = build_initial_messages(user_text, overview, history, system_prompt, metadata_block)
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
    empty_length_retries = 0

    def emit_and_record(event: str, payload: dict) -> None:
        recorded_events.append({"event": event, "data": payload})
        emit(event, payload)

    def flush_dump() -> None:
        try:
            dump_dir = default_agent_dump_dir()
            summary = write_agent_context_dump(
                dump_dir,
                system_prompt=system_prompt,
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

        _cur_step = step + 1

        def on_delta(kind: str, text: str) -> None:
            # Unrecorded live tokens (kept out of the context dump).
            emit("token", {"kind": kind, "text": text, "step": _cur_step})

        try:
            llm = call_llm(cfg, messages, on_delta=on_delta)
        except Exception as e:
            logger.exception("LLM call failed")
            emit_and_record("error", {"message": f"LLM request failed: {e}"})
            flush_dump()
            return
        usage = llm.get("usage") or {}
        if usage:
            emit_and_record("usage", {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "max_context": max_context,
                "step": _cur_step,
            })
        content = (llm.get("content") or "").strip()
        tool_calls = [tc for tc in (llm.get("tool_calls") or []) if tc.get("name")]
        if content:
            kind = "reasoning" if tool_calls else "message"
            emit_and_record(kind, {"text": content, "step": step + 1})

        if not tool_calls:
            # Thinking models may spend all completion tokens before emitting
            # the first tool call.  This is an interrupted response, not task
            # completion.  Give it one explicit continuation instead of
            # falsely showing "Done." with zero canvas changes.
            finish_reason = str(llm.get("finish_reason") or "").lower()
            completion_tokens = usage.get("completion_tokens")
            try:
                hit_token_limit = int(completion_tokens) >= AGENT_MAX_TOKENS
            except (TypeError, ValueError):
                hit_token_limit = False
            truncated = finish_reason in {"length", "max_tokens"} or hit_token_limit
            if truncated and empty_length_retries < 1:
                empty_length_retries += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response reached its token limit before making an action. "
                        "Continue the user's request now by calling the required tools. "
                        "Do not declare completion until the requested canvas changes are made."
                    ),
                })
                emit_and_record("status", {
                    "phase": "retrying_truncated_response",
                    "step": step + 1,
                    "finish_reason": finish_reason or "token_budget_exhausted",
                })
                continue
            if truncated:
                emit_and_record("error", {
                    "message": (
                        "The model reached its completion limit before making any tool call; "
                        "no segmentation was applied. Increase AGENT_MAX_TOKENS or use an Instruct model."
                    )
                })
                flush_dump()
                return
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
            # Native batch tools use this callback to stream their child
            # goto/text/evaluate/commit work as it happens.
            session.progress = lambda event, payload: emit_and_record(event, payload)
            session.active_tool_id = tc.get("id") or tname
            try:
                result, ui_events = execute_tool(tname, targs, session)
            finally:
                session.progress = None
                session.active_tool_id = None
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
