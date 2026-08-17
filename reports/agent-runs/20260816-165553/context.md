# Agent context dump

- Provider: `vllm`  model: `Qwen/Qwen3-VL-8B-Instruct`  profile: `a100`
- Base URL: `http://127.0.0.1:8001/v1`
- User: This home-cage frame has exactly two mice. Each mouse is ONE object: head, body, AND tail in a single connected mask. Do not split body and tail. Do not create extra objects. Shaved HeadShave is on the FAR RIGHT (torso about 0.78, 0.50). Unshaved NoShave is on the LEFT (torso about 0.32, 0.50). Inspect first. The canvas should be empty of old points. Then add_point_prompt once per existing object: 1) NoShave (object 2): one positive click on the left mouse torso. Include the tail in that same mask — never put a negative click on its tail. 2) HeadShave (object 1): one positive click on the right mouse torso. Do not click the circular water-bottle port (~0.85, 0.42) or empty bedding (x=0.50–0.70). evaluate_segmentation. If split_components, add another positive point on the missing tail/body, still on the same object id. Do not finish until both mice have compact connected masks on opposite sides. Do not start propagation.
- KEEP_INSPECT_IMAGES=2  AGENT_MAX_TOKENS=2048

## System prompt

```
You are the SAM3 Web Tracker agent. You annotate and track objects in behavior videos by calling tools. The user watches a live chat of your reasoning and actions while the main canvas updates the same way a human annotation would.

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
   e. If text segmentation is weak but you can see the animal, add_point_prompt at a chest/back point (positive=1). Use a negative point (0) on the *other* animal if they touch — never on the same animal's tail.
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
- One animal = one connected mask that includes head, body, and tail. Never put a negative click on the same mouse's tail (that splits body and tail into two blobs). If evaluate_segmentation reports split_components, add another positive point on the missing part (usually the tail), do not create a new object.

## Style
- Call think before multi-step plans and after failures.
- Keep user-facing messages short. Put detail in think / tool results.
- Do not reset or delete the user's existing objects unless they ask.
- Stay on the video the user specified; if they said "this video" use the current one.
- If the tracking mode is pose_tracking, say you only handle segmentation/tracking objects here.
```

## Retrieved project JSON

```json
{
  "project_id": "9f8a7b6c",
  "project_name": "sandbox_Home-Cage-Interactions-0126-test-day",
  "tracking_mode": "segmentation_tracking",
  "video_count": 47,
  "current_video_id": "f09434ac",
  "current_frame": 160,
  "videos": [
    {
      "id": "f09434ac",
      "name": "1A_video_test1a_20260130_090255.mp4",
      "num_frames": 59443,
      "fps": 30.0,
      "width": 1920,
      "height": 1080,
      "start_frame": 160,
      "anchor_batch_size": null,
      "object_count": 2,
      "objects": [
        {
          "id": "1",
          "name": "HeadShave",
          "description": "",
          "color": "#5B8DD9"
        },
        {
          "id": "2",
          "name": "NoShave",
          "description": "",
          "color": "#E8A445"
        }
      ],
      "prompted_frame_count": 0,
      "prompted_frames_sample": [],
      "annotated_anchors": [],
      "anchor_labeling_complete": false,
      "propagation_complete": false,
      "propagated_frame_count": 0,
      "is_current": true
    },
    {
      "id": "6bcb3f3a",
      "name": "1A_video_test1b_20260130_100517.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "3b15bce1",
      "name": "1A_video_test1c_20260130_111209.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "f0ad21e9",
      "name": "1A_video_test1d_20260130_121202.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "f04b915c",
      "name": "1A_video_test1e_20260130_131549.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "99d4a9cf",
      "name": "1A_video_test1f_20260130_140736.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "963b9700",
      "name": "1B_video_test1a_20260130_090255.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "a34f6ae6",
      "name": "1B_video_test1b_20260130_100517.mp4",
      "num_frames": 59443,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "9bb72f59",
      "name": "1B_video_test1c_20260130_111209.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "570b06d1",
      "name": "1B_video_test1d_20260130_121202.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "b5791e07",
      "name": "1B_video_test1e_20260130_131549.mp4",
      "num_frames": 59443,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "4c48a42f",
      "name": "1B_video_test1f_20260130_140736.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "946d42eb",
      "name": "2A_video_test1a_20260130_090255.mp4",
      "num_frames": 59442,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "9cda62c1",
      "name": "2A_video_test1b_20260130_100517.mp4",
      "num_frames": 59443,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "e100b21f",
      "name": "2A_video_test1c_20260130_111209.mp4",
      "num_frames": 59443,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "3bdd1043",
      "name": "2A_video_test1d_20260130_121202.mp4",
      "num_frames": 59443,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "8486b200",
      "name": "2A_video_test1e_20260130_131549.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "2a349693",
      "name": "2A_video_test1f_20260130_140736.mp4",
      "num_frames": 59438,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "daa91132",
      "name": "2B_video_test1a_20260130_090255.mp4",
      "num_frames": 59432,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "93f96275",
      "name": "2B_video_test1b_20260130_100517.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "7bf3ec98",
      "name": "2B_video_test1c_20260130_111209.mp4",
      "num_frames": 59433,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "e5099689",
      "name": "2B_video_test1d_20260130_121202.mp4",
      "num_frames": 59435,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "596eacb4",
      "name": "2B_video_test1e_20260130_131549.mp4",
      "num_frames": 59435,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "1641a5f6",
      "name": "2B_video_test1f_20260130_140736.mp4",
      "num_frames": 59432,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "9465c0b2",
      "name": "3A_video_test1a_20260130_090255.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "9bd13c70",
      "name": "3A_video_test1b_20260130_100517.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "757e9f04",
      "name": "3A_video_test1c_20260130_111209.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "00d48b80",
      "name": "3A_video_test1d_20260130_121202.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "4e107c2e",
      "name": "3A_video_test1e_20260130_131549.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "202b7239",
      "name": "3A_video_test1f_20260130_140736.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "baac729e",
      "name": "3B_video_test1a_20260130_090255.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "c0e2cab7",
      "name": "3B_video_test1b_20260130_100517.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "03462ba1",
      "name": "3B_video_test1c_20260130_111209.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "d2016605",
      "name": "3B_video_test1d_20260130_121202.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "6e181cda",
      "name": "3B_video_test1e_20260130_131549.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "03405fe3",
      "name": "4A_video_test1a_20260130_090255.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "96d4abf2",
      "name": "4A_video_test1b_20260130_100517.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "79ebb088",
      "name": "4A_video_test1c_20260130_111209.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "bcddef38",
      "name": "4A_video_test1d_20260130_121202.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "c7b1f910",
      "name": "4A_video_test1e_20260130_131549.mp4",
      "num_frames": 59442,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "ab6ee81b",
      "name": "4A_video_test1f_20260130_140736.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "affc225e",
      "name": "4B_video_test1a_20260130_090255.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "e27bba61",
      "name": "4B_video_test1b_20260130_100517.mp4",
      "num_frames": 59443,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "65285bbb",
      "name": "4B_video_test1c_20260130_111209.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "646538c1",
      "name": "4B_video_test1d_20260130_121202.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "e35f08ef",
      "name": "4B_video_test1e_20260130_131549.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    },
    {
      "id": "03c0e06a",
      "name": "4B_video_test1f_20260130_140736.mp4",
      "num_frames": 59444,
      "object_count": 2,
      "is_current": false
    }
  ]
}
```

## Inspect JPEGs

| File | video_id | frame | in final LLM context |
|---|---|---|---|
| `inspect_vid-f09434ac_frame-160.jpg` | `f09434ac` | 160 | yes |

### inspect_vid-f09434ac_frame-160.jpg

![inspect frame 160](inspect_vid-f09434ac_frame-160.jpg)

## LLM turns (which inspect JPEGs were in context)

- Step 1: none
- Step 2: video `f09434ac` frame 160
- Step 3: video `f09434ac` frame 160
- Step 4: video `f09434ac` frame 160
- Step 5: video `f09434ac` frame 160

## Tool calls and results

### `inspect_frame` (step 1)

Arguments:

```json
{
  "frame_idx": 160,
  "video_id": "f09434ac"
}
```

Result (`ok=True`, 372 ms):

```json
{
  "ok": true,
  "video_id": "f09434ac",
  "video_name": "1A_video_test1a_20260130_090255.mp4",
  "frame_idx": 160,
  "num_frames": 59443,
  "objects": [
    {
      "id": "1",
      "name": "HeadShave",
      "description": ""
    },
    {
      "id": "2",
      "name": "NoShave",
      "description": ""
    }
  ],
  "prompts_on_frame": {},
  "existing_mask_quality": [],
  "image_attached": true,
  "note": "A JPEG of this frame is attached for you to look at."
}
```

### `add_point_prompt` (step 2)

Arguments:

```json
{
  "object_id": "2",
  "frame_idx": 160,
  "points": [
    [
      0.32,
      0.5
    ]
  ],
  "labels": [
    1
  ],
  "video_id": "f09434ac"
}
```

Result (`ok=True`, 3018 ms):

```json
{
  "ok": true,
  "frame_idx": 160,
  "object_id": "2",
  "mask_ids": [
    "2"
  ]
}
```

### `add_point_prompt` (step 3)

Arguments:

```json
{
  "object_id": "1",
  "frame_idx": 160,
  "points": [
    [
      0.78,
      0.5
    ]
  ],
  "labels": [
    1
  ],
  "video_id": "f09434ac"
}
```

Result (`ok=True`, 679 ms):

```json
{
  "ok": true,
  "frame_idx": 160,
  "object_id": "1",
  "mask_ids": [
    "1",
    "2"
  ]
}
```

### `evaluate_segmentation` (step 4)

Arguments:

```json
{
  "frame_idx": 160,
  "video_id": "f09434ac"
}
```

Result (`ok=True`, 49 ms):

```json
{
  "ok": true,
  "frame_idx": 160,
  "video_id": "f09434ac",
  "objects": [
    {
      "ok": true,
      "confidence": 0.72,
      "reason": "ok",
      "area_px": 22836,
      "coverage": 0.011013,
      "bbox_xywh_norm": [
        0.175,
        0.42778,
        0.20208,
        0.28889
      ],
      "bbox_fill": 0.1886,
      "model_score": null,
      "retry_nearby": false,
      "object_id": "2",
      "name": "NoShave",
      "overlap_px": 0
    },
    {
      "ok": true,
      "confidence": 0.72,
      "reason": "ok",
      "area_px": 26728,
      "coverage": 0.01289,
      "bbox_xywh_norm": [
        0.73698,
        0.39815,
        0.09167,
        0.21389
      ],
      "bbox_fill": 0.6574,
      "model_score": null,
      "retry_nearby": false,
      "object_id": "1",
      "name": "HeadShave",
      "overlap_px": 0
    }
  ],
  "object_count": 2,
  "expected_object_count": 2,
  "masked_object_count": 2,
  "incomplete": false,
  "note": null,
  "retry_nearby": false,
  "nearby_suggestion": [
    180,
    140,
    200,
    120
  ]
}
```

**message:** Both mice have compact, connected masks on opposite sides of the cage. The NoShave mouse (object 2) is on the left with a torso click at (0.32, 0.50), and the HeadShave mouse (object 1) is on the right with a torso click at (0.78, 0.50). The masks are complete and do not require additional points. No propagation has been started as requested.
