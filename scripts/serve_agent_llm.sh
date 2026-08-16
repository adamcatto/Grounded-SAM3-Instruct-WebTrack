#!/usr/bin/env bash
# Serve a local vision LLM for the Web Tracker agent.
# Matches SAM 3 Agent: OpenAI-compatible /v1 on vLLM :8001 (dummy API key),
# plus Ollama on :11434 as the easier single-GPU option.
set -euo pipefail

MODE="${1:-ollama}"  # ollama | vllm
shift || true

case "$MODE" in
  ollama)
    MODEL="${AGENT_LLM_MODEL:-qwen2.5vl}"
    echo "Ollama OpenAI-compat endpoint: http://127.0.0.1:11434/v1"
    echo "Model: $MODEL"
    echo "Pulling if needed, then serving…"
    command -v ollama >/dev/null || { echo "Install Ollama from https://ollama.com"; exit 1; }
    ollama pull "$MODEL"
    exec ollama serve
    ;;
  vllm)
    # Same command SAM 3 Agent documents in examples/sam3_agent.ipynb
    MODEL="${AGENT_LLM_MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
    TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    PORT="${VLLM_PORT:-8001}"
    echo "vLLM OpenAI-compat endpoint: http://127.0.0.1:${PORT}/v1"
    echo "Model: $MODEL  tensor_parallel=${TP}"
    echo "API key: DUMMY_API_KEY (not used)"
    exec vllm serve "$MODEL" \
      --tensor-parallel-size "$TP" \
      --allowed-local-media-path / \
      --enforce-eager \
      --port "$PORT" \
      "$@"
    ;;
  *)
    echo "Usage: $0 ollama|vllm"
    echo "  ollama  — ollama serve (default model qwen2.5vl)"
    echo "  vllm    — vllm serve Qwen/Qwen3-VL-8B-Thinking --port 8001"
    exit 1
    ;;
esac
