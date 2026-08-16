#!/usr/bin/env bash
# Serve a local vision LLM for the Web Tracker agent (OpenAI-compat /v1).
#
# Same pattern as SAM 3 Agent (vLLM :8001, dummy API key), but sized for
# workstation / cluster GPUs instead of the 8B notebook default.
#
# Usage:
#   bash scripts/serve_agent_llm.sh ollama [model]
#   bash scripts/serve_agent_llm.sh vllm [--profile a100|h100x4|a100-shared|demo] [--thinking] [--] [vllm flags…]
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
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"; shift 2 ;;
    --profile=*)
      PROFILE="${1#*=}"; shift ;;
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

case "$PROFILE" in
  a100)
    VLLM_MODEL="Qwen/Qwen3-VL-32B-Instruct"
    VLLM_THINKING_MODEL="Qwen/Qwen3-VL-32B-Thinking"
    OLLAMA_MODEL="qwen2.5vl:32b"
    TP_DEFAULT=1
    MAX_LEN=8192
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
    MAX_LEN=16384
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
    MAX_LEN=8192
    GPU_UTIL=0.40
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
    MAX_LEN=16384
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
    MODEL="${AGENT_LLM_MODEL:-$VLLM_MODEL}"
    TP="${VLLM_TENSOR_PARALLEL_SIZE:-$TP_DEFAULT}"
    PORT="${VLLM_PORT:-8001}"
    echo "vLLM OpenAI-compat endpoint: http://127.0.0.1:${PORT}/v1"
    echo "Profile: $PROFILE"
    echo "Model: $MODEL  tensor_parallel=${TP}  max_model_len=${MAX_LEN}  gpu_mem=${GPU_UTIL}"
    echo "API key: DUMMY_API_KEY (not used)"
    if [[ "$PROFILE" == "a100" ]]; then
      echo "Note: 32B bf16 needs a dedicated 80GB GPU. Do not share it with SAM3."
    fi
    ARGS=(
      serve "$MODEL"
      --tensor-parallel-size "$TP"
      --port "$PORT"
      --dtype bfloat16
      --max-model-len "$MAX_LEN"
      --gpu-memory-utilization "$GPU_UTIL"
      --max-num-seqs "$MAX_NUM_SEQS"
      --allowed-local-media-path /
    )
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
    echo "  a100-shared  Qwen3-VL-8B-Instruct, ~40% of an 80GB card (rest for SAM3)"
    echo "  demo         Qwen3-VL-8B-Thinking (SAM 3 Agent notebook)"
    echo
    echo "  --thinking   Use the Qwen3-VL *Thinking* checkpoint (32B on a100; 32B not 72B on h100x4)"
    exit 1
    ;;
esac
