---
name: playwright
description: Run browser E2E tests with Playwright for SAM3 Web Tracker (frontend + backend). Use when the user asks for Playwright tests, browser automation, UI verification, or end-to-end checks of mask swap/clear, annotation, or propagation flows.
---

# Playwright E2E (SAM3 Web Tracker)

## Layout

| Path | Purpose |
|------|---------|
| `e2e/playwright.config.ts` | Config — base URL `http://localhost:5173`, backend at `:8000` |
| `e2e/mask-ops.spec.ts` | Reversible mask swap/clear UI tests |
| `e2e/helpers/maskOps.ts` | API backup/restore so tests leave no lasting project changes |
| `scripts/test_mask_ops_reversible.py` | API + disk verification (no browser) |

## Prerequisites

Both servers must be running:

```bash
bash start.sh
# or: bash start_backend.sh && bash start_frontend.sh
```

Install Playwright once:

```bash
cd e2e && npm install && npx playwright install chromium
```

## Sandbox project (default test fixture)

| Field | Value |
|-------|-------|
| Project id | `9f8a7b6c` |
| Name | `sandbox_Home-Cage-Interactions-0126-test-day` |
| Video id | `f09434ac` |
| Anchor frame | `160` |
| Objects | `1` HeadShave, `2` NoShave |

Path: `/opt/projects/segmentation_tracking_projects/9f8a7b6c-sandbox_Home-Cage-Interactions-0126-test-day`

Agent E2E (`agent-e2e.spec.ts`) **resets the sandbox video** before and after the run: masks, point prompts, SAM session, and propagation state are wiped; the source video and HeadShave/NoShave objects are kept. Other videos in the project are left alone. Mask-swap tests still snapshot/restore and do not reset.

## Run tests

```bash
cd e2e
/home/adam/.nvm/versions/node/v22.18.0/bin/npx playwright test mask-clear.spec.ts
/home/adam/.nvm/versions/node/v22.18.0/bin/npx playwright test mask-ops.spec.ts  # swap UI ~2 min/swap
```

Or: `bash e2e/run-tests.sh mask-clear.spec.ts` (uses Node 22 npx directly).

Agent UI (needs vLLM on :8001 and SAM loaded):

```
bash e2e/run-tests.sh agent-e2e.spec.ts
```

API-only verification (disk + config + session):

API-only verification (disk + config + session):

```bash
conda run -n sam3 python scripts/test_mask_ops_reversible.py
```

## Reversibility rule

**Never leave the sandbox mutated.** Before any write:

1. Snapshot masks via `GET /api/projects/{pid}/videos/{vid}/masks/{frame}`.
2. Snapshot point prompts from `GET /api/projects/{pid}`.
3. After assertions, restore with `POST .../masks/restore_frames` and `PUT .../objects/{oid}/frames/{frame}/prompts`.

Use helpers in `e2e/helpers/maskOps.ts` or the Python script pattern.

## Opening the sandbox in the UI

1. Wait for loading screen to disappear (SAM model loaded).
2. Click header hamburger → **Projects** drawer.
3. Click project name containing `sandbox_Home-Cage`.
4. Click video `1A_video_test1a_20260130_090255.mp4`.
5. Jump frame via timeline number input (title `Jump to frame`).

## Key UI selectors

| Action | Locator hint |
|--------|----------------|
| Swap masks | Button title contains `Swap two objects' masks` |
| Swap modal confirm | Modal heading `Swap object masks` → button `Swap` |
| Clear frame masks | Button title `Clear saved masks for frame` |
| Clear modal (this frame) | Modal `Clear masks` → `This frame only` |
| Object clear | Object card → `Clear selection` or `Clear` |
| Frame jump | `input[title="Jump to frame"]` |

## What to verify at each layer

| Layer | How |
|-------|-----|
| Disk | Python `VideoMaskStorage.load_masks_dense` — SQLite + NPZ agree |
| API | `GET /masks/{frame}` — per-object PNG base64; compare mask area fingerprints |
| Config | `point_prompts` in project JSON — anchor clears remove prompts; bulk mask clear does not |
| Inference | `GET /session/state` — swap does not change prompts; object clear on anchor may leave stale SAM cache until rebuild |
| UI | Playwright toast text, canvas presence, saved mask cache after frame jump |

## Additional resources

- App architecture: `CLAUDE.md`
- Mask storage: `docs/ai/masks-sqlite-storage.md`
