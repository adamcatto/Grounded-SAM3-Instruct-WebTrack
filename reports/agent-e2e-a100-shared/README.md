# Agent E2E — A100-80GB shared with SAM3

Workstation: My Machines worker `a100-80gb`. SAM3 was already resident (~4 GB), so the vision LLM used **`a100-shared`** (not 32B):

```bash
bash scripts/serve_agent_llm.sh vllm --profile a100-shared
```

| | |
|---|---|
| GPU | 1× NVIDIA A100 80GB PCIe |
| SAM3 | `sam3` conda, uvicorn `:8000` (~4034 MiB) |
| Agent LLM | vLLM `Qwen/Qwen3-VL-8B-Instruct` `:8001` (~39 GB, `max_model_len=16384`, `gpu_mem=0.50`) |
| Combined | ~43 GB / 80 GB |
| Project | `sandbox_Home-Cage-Interactions-0126-test-day` (`9f8a7b6c`) |
| Video | `1A_video_test1a_20260130_090255.mp4` (`f09434ac`), frame **160** |
| Prompt | Inspect the current frame. If two mice are present, segment each of them. Reuse existing objects when the names already match; otherwise create objects. Do not start propagation. |

Playwright: `e2e/agent-e2e.spec.ts` (passed).

Full LLM transcript (system prompt, retrieved project JSON, every tool call/result, which inspect JPEGs were in context):

[`../agent-runs/20260816-155409/context.md`](../agent-runs/20260816-155409/context.md)

## What the agent did

1. **`inspect_frame`** frame 160 of `f09434ac` — JPEG stayed in the LLM context for steps 2–6.
2. **`text_segment`** ×2 (`HeadShave` / `NoShave`) — failed on a numpy truthiness bug (`array or []`); fixed in this branch after the run.
3. **`add_point_prompt`** ×2 on objects `1` and `2` — succeeded; reused existing HeadShave / NoShave.
4. Final message: inspected frame 160, annotated both mice, did not start propagation.

## Screenshots

### 01 — App loaded

![01-app-loaded](screenshots/01-app-loaded.png)

### 02 — Project / video open (frame 160)

![02-project-open](screenshots/02-project-open.png)

### 03 — Agent pane

![03-agent-pane](screenshots/03-agent-pane.png)

### 04 — After inspect (run completed in one burst)

![04-after-inspect](screenshots/04-after-inspect.png)

### 05 — After segment

![05-after-segment](screenshots/05-after-segment.png)

### 06 — Final

![06-final](screenshots/06-final.png)

## Inspect JPEG in LLM context

![inspect frame 160](../agent-runs/20260816-155409/inspect_vid-f09434ac_frame-160.jpg)

Kept in the final transcript (`KEEP_INSPECT_IMAGES=2`; only one inspect this run).
