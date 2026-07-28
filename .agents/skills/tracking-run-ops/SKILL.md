---
name: tracking-run-ops
description: >-
  Monitor, diagnose, and fix full-scale N-way parallel video-tracking runs of
  the Grounded-SAM3-Instruct-WebTrack (SAM3 WebTracker) on the Minerva LSF
  cluster. Use when submitting/babysitting a batch tracking job for a project
  under /sc/arion/projects/KennyComputational/Behavior/projects/, when a
  tracking job appears stuck/hung, when videos are eligible but never picked up,
  when migrating a project's config.json off /opt paths, or when verifying a run
  finished. Covers submitting jobs, migrated-project path fixes, bjobs/bpeek
  monitoring, orphaned-video and hung-worker diagnosis, and cleanup.
---

# Tracking Run Ops (SAM3 WebTracker, LSF parallel tracking)

Operational runbook for full-scale parallel video tracking in the
`Grounded-SAM3-Instruct-WebTrack` repo (the user sometimes calls it
"Grounded-SAM3-WebTracker" — same thing). Everything here is grounded in the
actual code; do not invent behavior beyond what is documented.

Key absolute paths:

- Repo: `/sc/arion/projects/KennyComputational/Behavior/Grounded-SAM3-Instruct-WebTrack`
- Scripts: `<repo>/scripts`
- Projects root: `/sc/arion/projects/KennyComputational/Behavior/projects`
- LSF logs: `/sc/arion/projects/KennyComputational/Behavior/logs/%J.out` (and `%J.err`)
- Conda env: `/sc/arion/work/cattoa01/miniconda3/envs/sam3`
- Example live project used throughout: `/sc/arion/projects/KennyComputational/Behavior/projects/2165bda9-SCORCH-2025`

## 1. Overview — how the tracking system works

A **project directory** holds:

- `config.json` — project + per-video metadata. Per video (`videos.<vid>`):
  `source_path`, `point_prompts`, `anchor_labeling_complete`,
  `propagation_complete`, `whole_video_inference.status` (`none` / `running` /
  `complete` / `failed`), `updated_at`, `num_frames`, `start_frame`.
- `videos/<vid>_<name>/` — per-video tree: `source.mp4`, `masks/` (`NNNNNN.npz`),
  `bboxes/` (`NNNNNN.json`), `frames/`, `annotated_frames/`,
  `propagation_progress.txt`.

**Full-scale tracking = submit `scripts/bsub_parallel_project_tracking.bsub` N
times** (N-way parallelism via N independent jobs — this is NOT an LSF array).
Each job:

- requests 1 H100 GPU (`-gpu num=1`, `h100nvl`) + 32 slots (`-n 32`) + a
  single-host span,
- activates the `sam3` conda env,
- starts **its own** uvicorn backend on a free OS-assigned port (see §3),
- runs:
  `python scripts/parallel_tracking_worker.py --local-gpu-workers 0 --backend http://127.0.0.1:<port> --use-all-anchors --clear-stuck <PROJECT_DIR>`.

Workers coordinate through a shared **claims file**
`tracked_videos_in_progress.txt` in the project dir, guarded by `fcntl` `flock`.
Each worker (see `_run_sequential_claim_loop` / `_claim_next_video_update` in
`scripts/parallel_tracking_worker.py`):

1. locks the claims file, prunes ids whose video is now `propagation_complete`,
   scans videos in sorted order, and claims **one** eligible unclaimed video,
2. propagates it via the backend SSE endpoint (`propagation_sse`), which writes
   masks/bboxes and marks `propagation_complete`,
3. releases the claim, then loops to claim the next one.

When no video is claimable, a worker does an **idle sweep**: sleeps
`idle_sleep_min`–`idle_sleep_max` seconds (15–1800 by default) and exits after
`idle_loops_before_exit` (288) consecutive empty sweeps. If there is genuinely
no eligible work left, it prints `No eligible propagation work left — exiting.`
and exits immediately.

Eligibility is decided by `video_eligibility(vid, vm)` in
`scripts/run_pending_inference.py` (~line 347). A video is eligible when it is
**anchored** (`anchor_labeling_complete` true, or inferred from point prompts /
mask files) AND **not** `propagation_complete` AND has **point_prompts** AND is
**not actively inferring**. A `whole_video_inference.status == "running"` blocks
eligibility only if it is *fresh*: if no `propagation_progress.txt` or
`masks/*.npz` file has been touched within ~300s (`stale_timeout_s`), the
video is treated as stale-running and becomes eligible again. `status ==
"failed"` is eligible (retry).

## 2. Submitting a run

```bash
cd /sc/arion/projects/KennyComputational/Behavior/Grounded-SAM3-Instruct-WebTrack/scripts
export PROJECT_DIR=/sc/arion/projects/KennyComputational/Behavior/projects/2165bda9-SCORCH-2025
# N = number of parallel jobs / H100s to use (one video propagates per job at a time)
N=8
for i in $(seq 1 $N); do bsub < ./bsub_parallel_project_tracking.bsub; done
```

Rules of thumb:

- Roughly submit as many jobs as videos you want propagating concurrently (each
  job holds one H100 and propagates one video at a time in this config).
- Over-submitting is safe: extra workers find no eligible work and exit cleanly.
- Under-submitting is safe: workers loop through remaining videos serially.
- To add capacity to a run that's already going, just `bsub` more copies of the
  same `.bsub` with the same `PROJECT_DIR`.

## 3. Co-located-jobs free-port fix (already in the .bsub — know why)

LSF `span[hosts=1]` + `-gpu num=1` does **not** reserve a whole node, so multiple
tracking jobs frequently land on the **same host**. The original `.bsub`
hardcoded `BACKEND_PORT=8810` and ran `fuser -k 8810/tcp`, so a co-located job
would kill its neighbor's backend — surfacing as
`uvicorn exited before becoming healthy`.

The current `.bsub` fixes this by picking a free OS-assigned port at runtime
(after `conda activate`):

```bash
BACKEND_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
```

GPU isolation is fine even when co-located: LSF cgroups renumber the allocated
GPU to index 0, so `CUDA_VISIBLE_DEVICES=0` stays correct per job. You should not
need to touch this — it's documented so you recognize the symptom and don't
re-introduce a fixed port.

## 4. Migrated-project path fix (REQUIRED for projects created on another machine)

Projects created on a different machine have `config.json` `source_path` fields
pointing at `/opt/projects/segmentation_tracking_projects/...`. The backend reads
`source_path` directly and calls `.exists()` on it during propagation, so on this
machine those paths **must be rewritten** or propagation fails.

Path mapping (only the prefix differs; video subdir names match):

```
/opt/projects/segmentation_tracking_projects/  ->  /sc/arion/projects/KennyComputational/Behavior/projects/
```

Procedure — **always back up first**, rewrite only `source_path`, then verify:

```bash
PROJECT_DIR=/sc/arion/projects/KennyComputational/Behavior/projects/2165bda9-SCORCH-2025

# 1. Back up config.json
cp "$PROJECT_DIR/config.json" "$PROJECT_DIR/config.json.bak_optpaths_$(date +%Y%m%d_%H%M%S)"

# 2. Rewrite only source_path fields
python - "$PROJECT_DIR/config.json" <<'PY'
import json, sys
OLD = "/opt/projects/segmentation_tracking_projects/"
NEW = "/sc/arion/projects/KennyComputational/Behavior/projects/"
p = sys.argv[1]
cfg = json.load(open(p))
n = 0
for vid, vm in (cfg.get("videos") or {}).items():
    sp = vm.get("source_path")
    if isinstance(sp, str) and sp.startswith(OLD):
        vm["source_path"] = NEW + sp[len(OLD):]
        n += 1
json.dump(cfg, open(p, "w"), indent=2)
print(f"rewrote {n} source_path fields")
PY

# 3. Verify 0 /opt refs remain and every source_path exists
python - "$PROJECT_DIR/config.json" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
opt = 0; missing = []
for vid, vm in (cfg.get("videos") or {}).items():
    sp = vm.get("source_path") or ""
    if "/opt/projects/segmentation_tracking_projects" in sp:
        opt += 1
    if sp and not os.path.isfile(sp):
        missing.append((vid, sp))
print(f"remaining /opt refs: {opt}")
print(f"missing files: {len(missing)}")
for vid, sp in missing[:20]:
    print("  MISSING", vid, sp)
PY
```

Do not proceed with a run until remaining `/opt` refs are 0 and missing files
is 0.

## 5. Monitoring

**Jobs:**

```bash
bjobs -J tracking -o "jobid stat exec_host run_time"
```

**Completion + status counts** (reads `config.json` directly):

```bash
PROJECT_DIR=/sc/arion/projects/KennyComputational/Behavior/projects/2165bda9-SCORCH-2025
python - "$PROJECT_DIR/config.json" <<'PY'
import json, sys
from collections import Counter
cfg = json.load(open(sys.argv[1]))
vids = cfg.get("videos") or {}
prop = Counter()
inf = Counter()
for vid, vm in vids.items():
    prop["complete" if vm.get("propagation_complete") else "incomplete"] += 1
    inf[((vm.get("whole_video_inference") or {}).get("status") or "none")] += 1
print(f"total videos: {len(vids)}")
print("propagation_complete:", dict(prop))
print("whole_video_inference.status:", dict(inf))
PY
```

**In-flight (claimed) videos:**

```bash
grep -v '^#' /sc/arion/projects/KennyComputational/Behavior/projects/2165bda9-SCORCH-2025/tracked_videos_in_progress.txt
```

**Live job stdout — use `bpeek`, NOT the log file.** The `.bsub` uses `-oo`
(overwrite) and LSF flushes those `logs/%J.out` files only at **job
completion**. For a running job, always use `bpeek`:

```bash
bpeek <jobid> | grep -E 'Claimed|Done|Idle sweep|No eligible|ERROR|Traceback|Batch'
```

Useful log lines to look for (from the worker):

- `Claimed <vid>; starting SSE propagation …` — worker picked up a video
- `Done <vid> ...` — video fully propagated + marked complete
- `Idle sweep #<n>: ...` — no claimable work this pass
- `No eligible propagation work left — exiting.` — clean shutdown
- `Batch ...` — SSE mini-batch progress
- `Failed <vid>: ...` / `ERROR` / `Traceback` — problems

## 6. Diagnosing

### 6a. Orphaned / eligible-but-unclaimed video

Symptom: a video with `propagation_complete=false`,
`whole_video_inference.status="none"`, `updated_at=None`, **not** in the claims
file, while tracking jobs are running — yet nobody picks it up.

Confirm it really is eligible by calling `video_eligibility` directly (set
`SAM3_PROJECTS_DIR` to the projects root first):

```bash
REPO=/sc/arion/projects/KennyComputational/Behavior/Grounded-SAM3-Instruct-WebTrack
PROJECT_DIR=/sc/arion/projects/KennyComputational/Behavior/projects/2165bda9-SCORCH-2025
VID=<video-id>
SAM3_PROJECTS_DIR="$(dirname "$PROJECT_DIR")" \
/sc/arion/work/cattoa01/miniconda3/envs/sam3/bin/python - "$REPO" "$PROJECT_DIR" "$VID" <<'PY'
import json, sys
from pathlib import Path
repo, project_dir, vid = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, str(Path(repo) / "scripts"))
sys.path.insert(0, str(Path(repo) / "backend"))
from run_pending_inference import video_eligibility
cfg = json.load(open(Path(project_dir) / "config.json"))
vm = (cfg.get("videos") or {}).get(vid) or {}
ok, why = video_eligibility(vid, vm, project_dir=Path(project_dir))
print(f"{vid}: eligible={ok}  reason={why}")
PY
```

If it reports `eligible=True` but no worker is claiming it, the running workers
are wedged — see §6b — or the video id is stuck in the claims file (see §7,
stale claims).

### 6b. Hung workers (the key failure mode)

Observed: workers wedge **after** finishing a video. They completed propagation
fine (`Done <vid>` appears in `bpeek`) but then got stuck in the post-completion
re-claim step. They keep the LSF job in `RUN` state but do **zero work**, never
pick up remaining eligible videos, and waste H100s. This was **not** a flock
problem — the claims lock was grabbable instantly.

**Detect a hung job definitively via flat CPU time.** An actively-propagating
job's CPU time grows by hundreds/thousands of seconds between snapshots; a hung
job's `CPU time used` is completely flat (0 growth). Snapshot each job twice,
~60–90s apart, and flag the flat ones:

```bash
# CPU-delta hang check for all running 'tracking' jobs.
cpu_snapshot() {
  for j in $(bjobs -J tracking -o 'jobid' -noheader 2>/dev/null); do
    cpu=$(bjobs -l "$j" 2>/dev/null | tr '\n' ' ' \
          | grep -oE 'CPU time used is [0-9.]+ seconds' | grep -oE '[0-9.]+' | head -1)
    echo "$j ${cpu:-NA}"
  done
}
echo "== t0 =="; cpu_snapshot > /tmp/cpu_t0.txt; cat /tmp/cpu_t0.txt
echo "waiting 90s..."; sleep 90
echo "== t1 =="; cpu_snapshot > /tmp/cpu_t1.txt; cat /tmp/cpu_t1.txt
echo "== delta (FLAT = suspected hung) =="
join /tmp/cpu_t0.txt /tmp/cpu_t1.txt 2>/dev/null | awk '{
  d = $3 - $2;
  printf "job %s  cpu %s -> %s  delta %.1f%s\n", $1, $2, $3, d, (d==0 ? "   <-- FLAT/HUNG" : "")
}'
```

(`sleep` may be blocked in some harnesses; if so, take the two snapshots by hand
~60–90s apart and compare.)

Before killing anything, cross-check the claims file: a hung job must hold **no
active claim**. Because the hang happens only *after* a video is fully
propagated and marked complete, in-progress propagations always finish
correctly — leftover hung jobs are pure resource cleanup, **not** data loss.

## 7. Fixing

**Orphaned video (§6a):** submit **one** fresh job with the same `.bsub`. A fresh
worker's very first claim always succeeds, so it grabs the orphan within ~30s.
Verify by watching the claims file / the video's status flip to `running`:

```bash
cd /sc/arion/projects/KennyComputational/Behavior/Grounded-SAM3-Instruct-WebTrack/scripts
export PROJECT_DIR=/sc/arion/projects/KennyComputational/Behavior/projects/2165bda9-SCORCH-2025
bsub < ./bsub_parallel_project_tracking.bsub
```

**Hung jobs (§6b):** after confirming FLAT CPU **and** that the job holds no
active claim, `bkill` it to free the H100. Be careful to KEEP genuinely-active
jobs — re-confirm each keeper's CPU is still **growing** before killing anything.

```bash
bkill <hung-jobid> [<hung-jobid> ...]
```

**Stale `status="running"` left by killed jobs:** the worker's `--clear-stuck`
flag (baked into the `.bsub`) resets these back to `none` at startup. Independent
of that, `video_eligibility` already treats a stale-running video (no
`propagation_progress.txt` / `masks/*.npz` update within ~300s) as eligible
again, so a fresh job will reclaim it without manual intervention.

**Stale claims:** a `bkill -9` can leave a video id in
`tracked_videos_in_progress.txt`. The claim-prune logic only removes ids whose
video is `propagation_complete`, so a stale claim on an **incomplete** video will
block re-claiming forever. Remove that line manually to free it:

```bash
CLAIMS=/sc/arion/projects/KennyComputational/Behavior/projects/2165bda9-SCORCH-2025/tracked_videos_in_progress.txt
cp "$CLAIMS" "$CLAIMS.bak_$(date +%Y%m%d_%H%M%S)"
# inspect first:
grep -v '^#' "$CLAIMS"
# then delete the stale video id line (do this only if no live job is propagating it):
grep -v '^<stale-video-id>$' "$CLAIMS.bak_"* > "$CLAIMS"
```

## 8. Verifying final completion

The run is done when **all** videos have `propagation_complete=true` (use the
§5 status-count snippet; `propagation_complete` should be all `complete`).

Then `bkill` any leftover `RUN` tracking jobs — because the hang happens at the
final empty re-claim, some workers may not exit cleanly on their own:

```bash
bjobs -J tracking -o "jobid stat exec_host run_time"
# for any that are still RUN after all videos are complete:
bkill <jobid> ...
```

Optionally re-run the migrated-project verifier (§4 step 3) to confirm every
`source_path` still resolves, and spot-check that each video dir has populated
`masks/` and `bboxes/`.
