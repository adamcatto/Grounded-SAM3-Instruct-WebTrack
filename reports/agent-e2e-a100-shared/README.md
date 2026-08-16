# Agent E2E (A100-80GB shared profile)

Project: `sandbox_Home-Cage-Interactions-0126-test-day`  video: `1A_video_test1a_20260130_090255.mp4`  frame: 160

LLM: `{"provider":"vllm","model":"Qwen/Qwen3-VL-8B-Instruct","base_url":"http://127.0.0.1:8001/v1","profile":"a100","local":true}`

Tool calls: 4

## Outcome

Sandbox video was **reset** before the run (no leftover prompts/masks). Agent inspected, then one positive click per object.

| Object | Name | Mask bbox xywh (norm) | Center x | Area px | reason |
|---|---|---|---|---|---|
| 2 | NoShave (left) | `[0.175, 0.428, 0.202, 0.289]` | 0.276 | 22836 | ok |
| 1 | HeadShave (far right) | `[0.737, 0.398, 0.092, 0.214]` | 0.783 | 26728 | ok |

No `split_components`. Screenshot `02-project-open` is the blank canvas after reset. After the test the video is reset again (source + objects only).

Context dump: `reports/agent-runs/20260816-165553/`


## Screenshots

### 01-app-loaded

![01-app-loaded](screenshots/01-app-loaded.png)

### 02-project-open

![02-project-open](screenshots/02-project-open.png)

### 03-agent-pane

![03-agent-pane](screenshots/03-agent-pane.png)

### 04-after-inspect

![04-after-inspect](screenshots/04-after-inspect.png)

### 05-after-segment

![05-after-segment](screenshots/05-after-segment.png)

### 06-final

![06-final](screenshots/06-final.png)

## Context trace

Full dump: `/opt/software/Grounded-SAM3-Instruct-WebTrack/reports/agent-runs/20260816-165553`  ([context.md](../../reports/agent-runs/20260816-165553/context.md))

Inspect JPEGs in final LLM context:

```json
[
  {
    "message_index": 4,
    "image_index": 0,
    "role": "user",
    "frame_idx": 160,
    "video_id": "f09434ac",
    "has_jpeg": true
  }
]
```

Inspect JPEGs seen:

```json
[
  {
    "file": "inspect_vid-f09434ac_frame-160.jpg",
    "frame_idx": 160,
    "video_id": "f09434ac",
    "in_final_llm_context": true
  }
]
```
