#!/usr/bin/env bash
# Serve a local vision LLM for the Web Tracker agent (OpenAI-compat /v1).
#
# Same pattern as SAM 3 Agent (vLLM :8001, dummy API key), but sized for
# workstation / cluster GPUs instead of the 8B notebook default.
#
# Usage:
#   bash scripts/serve_agent_llm.sh ollama [model]
#   bash scripts/serve_agent_llm.sh vllm [--profile a100|h100x4|a100-shared|demo] [--thinking]
#                                        [--context-size 65536] [--cpu-offload-gb 16] [--] [vllm flags…]
#
# Context window defaults to 2**16 (65536) tokens. Override with --context-size
# or VLLM_MAX_MODEL_LEN. If the KV cache does not fit VRAM, add --cpu-offload-gb
# (or VLLM_CPU_OFFLOAD_GB) to spill model weights to CPU RAM; KV swap space
# defaults to 16 GiB (VLLM_SWAP_SPACE).
#
# Profiles (also AGENT_LLM_PROFILE):
#   a100         1× 80GB A100 *dedicated to the LLM* → Qwen3-VL-32B-Instruct
#   h100x4       4× 80GB H100 NVL → Qwen2.5-VL-72B-Instruct, tensor-parallel 4
#   a100-shared  same 80GB GPU as SAM3 → Qwen3-VL-8B-Instruct (leave VRAM for tracking)
#   demo         SAM 3 Agent notebook → Qwen3-VL-8B-Thinking
#
# Do not co-locate 32B/72B with SAM3 on one 80GB card. Typical split:
#   workstation A100  → SAM3 backend
#   4× H100 NVL       → this script --profile h100x4
#   tracker env       → AGENT_LLM_BASE_URL=http://<h100-host>:8001/v1
set -euo pipefail

MODE="${1:-ollama}"
if [[ $# -gt 0 ]]; then
  shift
fi

PROFILE="${AGENT_LLM_PROFILE:-a100}"
THINKING="${AGENT_LLM_THINKING:-0}"
# Default context window: 2**16 = 65536 tokens. Override with --context-size,
# VLLM_MAX_MODEL_LEN, or per-profile. Large windows may not fit VRAM at the
# profile's gpu-memory-utilization — use --cpu-offload-gb to spill model weights
# to CPU RAM (freeing VRAM for the KV cache) when it does not.
CONTEXT_DEFAULT=65536
CONTEXT_SIZE="${VLLM_MAX_MODEL_LEN:-}"
CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      if [[ $# -lt 2 || -z "${2}" || "${2}" == --* ]]; then
        echo "error: --profile requires a value (a100 | h100x4 | a100-shared | demo)" >&2
        exit 1
      fi
      PROFILE="$2"; shift 2 ;;
    --profile=*)
      PROFILE="${1#*=}"
      if [[ -z "$PROFILE" ]]; then
        echo "error: --profile requires a value (a100 | h100x4 | a100-shared | demo)" >&2
        exit 1
      fi
      shift ;;
    --context-size)
      if [[ $# -lt 2 || -z "${2}" || "${2}" == --* ]]; then
        echo "error: --context-size requires a token count (e.g. 65536)" >&2
        exit 1
      fi
      CONTEXT_SIZE="$2"; shift 2 ;;
    --context-size=*)
      CONTEXT_SIZE="${1#*=}"
      if [[ -z "$CONTEXT_SIZE" ]]; then
        echo "error: --context-size requires a token count (e.g. 65536)" >&2
        exit 1
      fi
      shift ;;
    --cpu-offload-gb)
      if [[ $# -lt 2 || -z "${2}" || "${2}" == --* ]]; then
        echo "error: --cpu-offload-gb requires a GiB value (e.g. 16)" >&2
        exit 1
      fi
      CPU_OFFLOAD_GB="$2"; shift 2 ;;
    --cpu-offload-gb=*)
      CPU_OFFLOAD_GB="${1#*=}"; shift ;;
    --thinking)
      THINKING=1; shift ;;
    --)
      shift
      EXTRA+=("$@")
      break ;;
    *)
      EXTRA+=("$1"); shift ;;
  esac
done

case "${PROFILE,,}" in
  workstation|a100-80|80gb) PROFILE="a100" ;;
  cluster|h100|4xh100|h100nvl) PROFILE="h100x4" ;;
  shared) PROFILE="a100-shared" ;;
  sam3|8b) PROFILE="demo" ;;
esac

thinking_on() {
  case "${THINKING,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

_ensure_vllm_env() {
  if command -v vllm >/dev/null 2>&1; then
    # Conda-built libicu needs the env libstdc++, not the older system one.
    local bindir
    bindir="$(dirname "$(command -v vllm)")"
    local libdir
    libdir="$(cd "$bindir/../lib" 2>/dev/null && pwd || true)"
    if [[ -n "$libdir" && -d "$libdir" ]]; then
      export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    return 0
  fi
  local cand
  for cand in \
    "${VLLM_CONDA_PREFIX:-}" \
    "${HOME}/miniconda3/envs/vllm" \
    "${HOME}/mambaforge/envs/vllm" \
    "/opt/conda/envs/vllm"
  do
    [[ -n "$cand" && -x "$cand/bin/vllm" ]] || continue
    export PATH="$cand/bin:$PATH"
    export LD_LIBRARY_PATH="$cand/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Using vLLM from $cand"
    return 0
  done
  echo "vLLM not found on PATH. Activate a vLLM env or set VLLM_CONDA_PREFIX." >&2
  exit 1
}

case "$PROFILE" in
  a100)
    VLLM_MODEL="Qwen/Qwen3-VL-32B-Instruct"
    VLLM_THINKING_MODEL="Qwen/Qwen3-VL-32B-Thinking"
    OLLAMA_MODEL="qwen2.5vl:32b"
    TP_DEFAULT=1
    MAX_LEN_DEFAULT=24576
    GPU_UTIL=0.95
    REASONING=0
    TOOLS=1
    ENFORCE_EAGER=0
    MAX_NUM_SEQS=2
    ;;
  h100x4)
    VLLM_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
    VLLM_THINKING_MODEL="Qwen/Qwen3-VL-32B-Thinking"
    OLLAMA_MODEL="qwen2.5vl:72b"
    TP_DEFAULT=4
    MAX_LEN_DEFAULT=32768
    GPU_UTIL=0.90
    REASONING=0
    TOOLS=1
    ENFORCE_EAGER=0
    MAX_NUM_SEQS=4
    ;;
  a100-shared)
    VLLM_MODEL="Qwen/Qwen3-VL-8B-Instruct"
    VLLM_THINKING_MODEL="Qwen/Qwen3-VL-8B-Thinking"
    OLLAMA_MODEL="qwen2.5vl"
    TP_DEFAULT=1
    MAX_LEN_DEFAULT=16384
    GPU_UTIL=0.50
    REASONING=0
    TOOLS=1
    ENFORCE_EAGER=0
    MAX_NUM_SEQS=1
    ;;
  demo)
    VLLM_MODEL="Qwen/Qwen3-VL-8B-Thinking"
    VLLM_THINKING_MODEL="Qwen/Qwen3-VL-8B-Thinking"
    OLLAMA_MODEL="qwen2.5vl"
    TP_DEFAULT=1
    MAX_LEN_DEFAULT=16384
    GPU_UTIL=0.90
    REASONING=1
    TOOLS=1
    ENFORCE_EAGER=1
    MAX_NUM_SEQS=2
    ;;
  *)
    echo "Unknown profile: $PROFILE"
    echo "Use: a100 | h100x4 | a100-shared | demo"
    exit 1
    ;;
esac

# Precedence: --context-size / VLLM_MAX_MODEL_LEN, then the 64k default.
# (Per-profile MAX_LEN_DEFAULT is retained above as the previously tuned value.)
MAX_LEN="${CONTEXT_SIZE:-$CONTEXT_DEFAULT}"

if thinking_on; then
  VLLM_MODEL="$VLLM_THINKING_MODEL"
  REASONING=1
  # 72B has no Thinking checkpoint; h100x4 --thinking switches to 32B Thinking.
  if [[ "$PROFILE" == "h100x4" ]]; then
    TP_DEFAULT="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
  fi
fi

case "$MODE" in
  ollama)
    if [[ ${#EXTRA[@]} -gt 0 && "${EXTRA[0]}" != -* ]]; then
      MODEL="${EXTRA[0]}"
    else
      MODEL="${AGENT_LLM_MODEL:-$OLLAMA_MODEL}"
    fi
    echo "Ollama OpenAI-compat endpoint: http://127.0.0.1:11434/v1"
    echo "Profile: $PROFILE"
    echo "Model: $MODEL"
    echo "Pulling if needed, then serving…"
    command -v ollama >/dev/null || { echo "Install Ollama from https://ollama.com"; exit 1; }
    ollama pull "$MODEL"
    exec ollama serve
    ;;
  vllm)
    _ensure_vllm_env
    MODEL="${AGENT_LLM_MODEL:-$VLLM_MODEL}"
    TP="${VLLM_TENSOR_PARALLEL_SIZE:-$TP_DEFAULT}"
    PORT="${VLLM_PORT:-8001}"
    SWAP_SPACE="${VLLM_SWAP_SPACE:-16}"
    echo "vLLM OpenAI-compat endpoint: http://127.0.0.1:${PORT}/v1"
    echo "Profile: $PROFILE"
    echo "Model: $MODEL  tensor_parallel=${TP}  max_model_len=${MAX_LEN}  gpu_mem=${GPU_UTIL}"
    echo "KV swap space: ${SWAP_SPACE} GiB (CPU)  cpu_offload=${CPU_OFFLOAD_GB} GiB"
    echo "API key: DUMMY_API_KEY (not used)"
    if [[ "${CPU_OFFLOAD_GB}" == "0" ]]; then
      echo "Tip: if 64k context OOMs the KV cache, add --cpu-offload-gb 16 to spill weights to CPU RAM."
    fi
    if [[ "$PROFILE" == "a100" ]]; then
      echo "Note: 32B bf16 needs a dedicated 80GB GPU. Do not share it with SAM3."
    fi
    if [[ "$PROFILE" == "a100-shared" ]]; then
      echo "Note: 8B at gpu_mem=${GPU_UTIL} / max_model_len=${MAX_LEN} so SAM3 can stay on the same 80GB card."
    fi
    ARGS=(
      serve "$MODEL"
      --tensor-parallel-size "$TP"
      --port "$PORT"
      --dtype bfloat16
      --max-model-len "$MAX_LEN"
      --gpu-memory-utilization "$GPU_UTIL"
      --max-num-seqs "$MAX_NUM_SEQS"
      --swap-space "$SWAP_SPACE"
      --allowed-local-media-path /
    )
    if [[ "${CPU_OFFLOAD_GB}" != "0" && -n "${CPU_OFFLOAD_GB}" ]]; then
      ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
    fi
    if [[ "$TOOLS" == "1" ]]; then
      ARGS+=(--enable-auto-tool-choice --tool-call-parser hermes)
    fi
    if [[ "$REASONING" == "1" ]]; then
      ARGS+=(--reasoning-parser qwen3)
    fi
    if [[ "$ENFORCE_EAGER" == "1" ]]; then
      ARGS+=(--enforce-eager)
    fi
    exec vllm "${ARGS[@]}" "${EXTRA[@]}"
    ;;
  *)
    echo "Usage: $0 ollama|vllm [--profile a100|h100x4|a100-shared|demo] [--thinking]"
    echo
    echo "  a100         Qwen3-VL-32B-Instruct on 1× 80GB A100 (dedicated GPU)"
    echo "  h100x4       Qwen2.5-VL-72B-Instruct TP=4 on 4× 80GB H100 NVL"
    echo "  a100-shared  Qwen3-VL-8B-Instruct, ~50% of an 80GB card, 16k ctx (rest for SAM3)"
    echo "  demo         Qwen3-VL-8B-Thinking (SAM 3 Agent notebook)"
    echo
    echo "  --thinking          Use the Qwen3-VL *Thinking* checkpoint (32B on a100; 32B not 72B on h100x4)"
    echo "  --context-size N    Max context window in tokens (default 65536 = 2**16)"
    echo "  --cpu-offload-gb N  Spill N GiB of model weights to CPU RAM to fit a larger KV cache"
    exit 1
    ;;
esac
