import axios from 'axios'
import type { Project, VideoMeta, ObjectDef, MaskData } from '../types'
import { parseSseChunk, SSE_EVENT_SPLIT } from './sseParse'

const api = axios.create({ baseURL: '/api' })

// ─── Health ──────────────────────────────────────────────────────────────────

export interface HealthStatus {
  status: string
  sam_ready: boolean
  sam_loaded: boolean
  sam_loading: boolean
  sam_model: string
  sam_load_error: string | null
}

export const checkHealth = () =>
  api.get<HealthStatus>('/health').then(r => r.data)

// ─── Projects ────────────────────────────────────────────────────────────────

export const listProjects = () =>
  api.get<Project[]>('/projects').then(r => r.data)

export interface ProjectsRootInfo {
  active_root: string
  env_default_root: string
  env_var: string | null
  app_root: string
  home: string
}

export const getProjectsRoot = () =>
  api.get<ProjectsRootInfo>('/projects/root').then(r => r.data)

export const setProjectsRoot = (path: string) =>
  api.post<ProjectsRootInfo>('/projects/root', { path }).then(r => r.data)

export interface FsListEntry {
  name: string
  path: string
  is_dir: boolean
  is_project: boolean
  size: number
  is_video: boolean
}

export interface FsListDirResponse {
  path: string
  parent: string | null
  entries: FsListEntry[]
}

export const listDir = (path?: string) =>
  api
    .get<FsListDirResponse>('/fs/list_dir', { params: { path: path ?? '' } })
    .then(r => r.data)

export const createProject = (
  name: string,
  trackingMode: 'segmentation_tracking' | 'pose_tracking' = 'segmentation_tracking',
) =>
  api.post<Project>('/projects', { name, tracking_mode: trackingMode }).then(r => r.data)

export const addPoseObject = (pid: string, vid: string, name: string, color: string) =>
  api.post(`/projects/${pid}/videos/${vid}/pose/objects`, { name, color }).then(r => r.data)

export const addPosePart = (pid: string, vid: string, oid: string, name: string, color: string) =>
  api.post(`/projects/${pid}/videos/${vid}/pose/objects/${oid}/parts`, { name, color }).then(r => r.data)

export const setPoseAnnotation = (
  pid: string, vid: string, oid: string, partId: string, frameIdx: number, x: number, y: number,
) => api.put(
  `/projects/${pid}/videos/${vid}/pose/objects/${oid}/parts/${partId}/frames/${frameIdx}`,
  { x, y, visible: true },
).then(r => r.data)

export const setPoseOccluded = (
  pid: string, vid: string, oid: string, partId: string, frameIdx: number,
) => api.put(
  `/projects/${pid}/videos/${vid}/pose/objects/${oid}/parts/${partId}/frames/${frameIdx}`,
  { visible: false },
).then(r => r.data)

export const deletePoseAnnotation = (
  pid: string, vid: string, oid: string, partId: string, frameIdx: number,
) => api.delete(
  `/projects/${pid}/videos/${vid}/pose/objects/${oid}/parts/${partId}/frames/${frameIdx}`,
).then(r => r.data)

export const clearPoseAnnotationsForFrame = (pid: string, vid: string, frameIdx: number) =>
  api.delete(`/projects/${pid}/videos/${vid}/pose/frames/${frameIdx}/annotations`).then(r => r.data)

export const skipPoseAnchor = (pid: string, vid: string, frameIdx: number) =>
  api.post(`/projects/${pid}/videos/${vid}/pose/anchors/${frameIdx}/skip`).then(r => r.data)

export const unskipPoseAnchor = (pid: string, vid: string, frameIdx: number) =>
  api.delete(`/projects/${pid}/videos/${vid}/pose/anchors/${frameIdx}/skip`).then(r => r.data)

export const predictPose = (pid: string, vid: string, startFrame: number, numFrames: number) =>
  api.post(`/projects/${pid}/videos/${vid}/pose/predict`, {
    start_frame: startFrame,
    num_frames: numFrames,
  }).then(r => r.data)

export const getPoseTracks = (pid: string, vid: string, frameIdx: number) =>
  api.get(`/projects/${pid}/videos/${vid}/pose/tracks/${frameIdx}`).then(r => r.data)

export const getPoseModelStatus = () => api.get('/pose/model').then(r => r.data)

export const cachePoseMemory = (pid: string, vid: string, frameIdx: number) =>
  api.post(`/projects/${pid}/videos/${vid}/pose/memory/${frameIdx}`).then(r => r.data)

export const mergeProjects = (
  name: string,
  leftProjectId: string,
  rightProjectId: string,
  outputParent?: string,
) =>
  api
    .post<Project>('/projects/merge', {
      name,
      left_project_id: leftProjectId,
      right_project_id: rightProjectId,
      output_parent: outputParent ?? null,
    })
    .then(r => r.data)

export const getProject = (pid: string) =>
  api.get<Project>(`/projects/${pid}`).then(r => r.data)

export const updateProject = (pid: string, name: string) =>
  api.patch<Project>(`/projects/${pid}`, { name }).then(r => r.data)

export const deleteProject = (pid: string) =>
  api.delete(`/projects/${pid}`)

// ─── Videos ──────────────────────────────────────────────────────────────────

export interface DownsampleOptions {
  maxDim?: number
  scaleFactor?: number
}

export const addVideo = (pid: string, file: File, onProgress?: (pct: number) => void, ds?: DownsampleOptions) => {
  const form = new FormData()
  form.append('file', file)
  if (ds?.maxDim) form.append('max_dim', String(ds.maxDim))
  if (ds?.scaleFactor) form.append('scale_factor', String(ds.scaleFactor))
  return api.post<VideoMeta>(`/projects/${pid}/videos`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  }).then(r => r.data)
}

export const importVideo = (pid: string, serverPath: string, ds?: DownsampleOptions, symlink?: boolean) =>
  api.post<VideoMeta>(`/projects/${pid}/videos/import`, {
    path: serverPath,
    max_dim: ds?.maxDim ?? null,
    scale_factor: ds?.scaleFactor ?? null,
    symlink: symlink ?? false,
  }).then(r => r.data)

export const downsampleVideo = (pid: string, vid: string, ds: DownsampleOptions) =>
  api.post<{ status: string; width: number; height: number; num_frames?: number; message?: string }>(
    `/projects/${pid}/videos/${vid}/downsample`,
    { max_dim: ds.maxDim ?? null, scale_factor: ds.scaleFactor ?? null },
    { timeout: 1800_000 },
  ).then(r => r.data)

export const initializeRegistration = (pid: string, targetSize = 1000) =>
  api.post(`/projects/${pid}/registration`, { target_size: targetSize }).then(r => r.data)

export const addRegistrationPoints = (
  pid: string,
  vid: string,
  points: [number, number][],
  labels: number[],
) =>
  api.post<{
    frame_idx: number
    mask: string
    registration: {
      points: [number, number][]
      labels: number[]
      labeled: boolean
      registered: boolean
    }
  }>(`/projects/${pid}/registration/videos/${vid}/points`, { points, labels }).then(r => r.data)

export const computeRegistration = (
  pid: string,
  warpMode: 'bounded_full_frame_mesh' | 'affine_full_frame',
) =>
  api.post(`/projects/${pid}/registration/compute`, { warp_mode: warpMode }).then(r => r.data)

export const getRegistrationMask = (pid: string, vid: string) =>
  api.get<{ mask: string | null; registration: import('../types').VideoRegistration }>(
    `/projects/${pid}/registration/videos/${vid}/mask`,
  ).then(r => r.data)

export const clearRegistrationVideo = (pid: string, vid: string) =>
  api.delete<{ mask: null; registration: import('../types').VideoRegistration }>(
    `/projects/${pid}/registration/videos/${vid}`,
  ).then(r => r.data)

export const setRegistrationPolygon = (
  pid: string,
  vid: string,
  vertices: [number, number][],
) =>
  api.post<{ mask: string; registration: import('../types').VideoRegistration }>(
    `/projects/${pid}/registration/videos/${vid}/polygon`,
    { vertices },
  ).then(r => r.data)

export const fitRegistrationEdges = (
  pid: string,
  vid: string,
  edges: Record<'top' | 'right' | 'bottom' | 'left', [number, number][]>,
) =>
  api.post<{ registration: import('../types').VideoRegistration; straightness_rms_pixels: number }>(
    `/projects/${pid}/registration/videos/${vid}/edges`,
    { edges },
  ).then(r => r.data)

export const registrationPreviewUrl = (pid: string, vid: string) =>
  `/api/projects/${pid}/registration/videos/${vid}/preview`

export const registrationFramePreviewUrl = (
  pid: string,
  vid: string,
  frameIdx: number,
  view: 'original' | 'registered',
  revision = '',
) => {
  const params = new URLSearchParams({ view })
  if (revision) params.set('rev', revision)
  return `/api/projects/${pid}/registration/videos/${vid}/preview/${frameIdx}?${params}`
}

export const applyRegistrationMorphology = (
  pid: string,
  vid: string,
  operation: 'opening' | 'closing',
  kernelSize: number,
) =>
  api.post<{ mask: string; registration: import('../types').VideoRegistration }>(
    `/projects/${pid}/registration/videos/${vid}/morphology`,
    { operation, kernel_size: kernelSize },
  ).then(r => r.data)

export const undoRegistrationMorphology = (pid: string, vid: string) =>
  api.post<{ mask: string; registration: import('../types').VideoRegistration }>(
    `/projects/${pid}/registration/videos/${vid}/morphology/undo`,
  ).then(r => r.data)

export const redoRegistrationMorphology = (pid: string, vid: string) =>
  api.post<{ mask: string; registration: import('../types').VideoRegistration }>(
    `/projects/${pid}/registration/videos/${vid}/morphology/redo`,
  ).then(r => r.data)

export interface BrowseEntry {
  name: string
  path: string
  size: number
  is_video: boolean
}

export const browseDirectory = (path: string, depth = 1) =>
  api.get<{ directory: string; files: BrowseEntry[] }>('/browse', { params: { path, depth } }).then(r => r.data)

export const getVideo = (pid: string, vid: string) =>
  api.get<VideoMeta>(`/projects/${pid}/videos/${vid}`).then(r => r.data)

export const getVideoInfo = (pid: string, vid: string) =>
  api.get<{ num_frames: number; actual_frames: number; frames_extracted: boolean; fps: number; width: number; height: number }>(
    `/projects/${pid}/videos/${vid}/info`
  ).then(r => r.data)

export const updateVideoMeta = (
  pid: string,
  vid: string,
  updates: { start_frame?: number; anchor_batch_size?: number },
) => api.patch(`/projects/${pid}/videos/${vid}`, updates).then(r => r.data)

export const removeVideo = (pid: string, vid: string) =>
  api.delete(`/projects/${pid}/videos/${vid}`)

export const resetVideo = (pid: string, vid: string, opts?: { keepObjects?: boolean }) =>
  api.post(`/projects/${pid}/videos/${vid}/reset`, null, {
    params: opts?.keepObjects ? { keep_objects: true } : {},
  }).then(r => r.data)

export const clearFramePrompts = (pid: string, vid: string, frameIdx: number) =>
  api.delete(`/projects/${pid}/videos/${vid}/frames/${frameIdx}/prompts`).then(r => r.data)

export const frameUrl = (pid: string, vid: string, fidx: number) =>
  `/api/projects/${pid}/videos/${vid}/frames/${fidx}`

export const displayMaskUrl = (pid: string, vid: string, fidx: number) =>
  `/api/projects/${pid}/videos/${vid}/display/${fidx}`

export interface CacheManifest {
  cache_revision: number
  display: { dir: string; ext: string; width: number; height: number; count: number; bytes: number }
  frames: { dir: string; ext: string; count: number; bytes: number }
  propagated_frames: number[]
  bytes_total: number
}

export const getCacheManifest = (pid: string, vid: string) =>
  api.get<CacheManifest>(`/projects/${pid}/videos/${vid}/cache/manifest`).then(r => r.data)

export const thumbUrl = frameUrl

// All backend traffic goes through the Vite dev proxy (relative `/api`) so the
// UI works over SSH tunnels / remote IPs where the browser can't resolve the
// backend's own localhost. The proxy preserves Accept-Ranges (video seeking)
// and un-buffers text/event-stream (SSE), so a direct connection isn't needed.
// Set VITE_BACKEND_URL only if you deliberately want to bypass the proxy.
const BACKEND = import.meta.env.VITE_BACKEND_URL ?? ''

export const videoSourceUrl = (pid: string, vid: string) =>
  `${BACKEND}/api/projects/${pid}/videos/${vid}/source`

// ─── Extract single frame (on-demand for annotation) ─────────────────────────

export const extractFrame = (pid: string, vid: string, frameIdx: number) =>
  api.post<{ status: string; frame_idx: number; path: string }>(
    `/projects/${pid}/videos/${vid}/extract_frame/${frameIdx}`
  ).then(r => r.data)

// ─── SAM3 Session ─────────────────────────────────────────────────────────────

export const initSession = (pid: string, vid: string) =>
  api.post<{ session_id: string; status: string }>(`/projects/${pid}/videos/${vid}/session`).then(r => r.data)

export const closeSession = (pid: string, vid: string) =>
  api.delete(`/projects/${pid}/videos/${vid}/session`)

export interface SessionState {
  session_active: boolean
  session_id: string | null
  model: string
  frame_map: number[]
  cached_sam_indices: number[]
  action_history_len: number
  obj_ids_tracked: number[]
  point_prompts: Record<string, Record<string, { points: [number, number][]; labels: number[] }>>
  saved_mask_frames: number[]
  saved_mask_obj_counts: Record<number, number>
  annotated_frame_files: number[]
  num_frames: number
  objects: Record<string, { id: string; name: string; color: string }>
}

export const getSessionState = (pid: string, vid: string) =>
  api.get<SessionState>(`/projects/${pid}/videos/${vid}/session/state`).then(r => r.data)

// ─── Objects ──────────────────────────────────────────────────────────────────

export const addObject = (
  pid: string,
  vid: string,
  name: string,
  color?: string,
  description?: string,
) =>
  api.post<ObjectDef>(`/projects/${pid}/videos/${vid}/objects`, {
    name,
    color,
    description: description ?? '',
  }).then(r => r.data)

export const updateObject = (
  pid: string,
  vid: string,
  oid: string,
  updates: { name?: string; color?: string; description?: string }
) =>
  api.patch(`/projects/${pid}/videos/${vid}/objects/${oid}`, updates).then(r => r.data)

export const renameObject = (pid: string, vid: string, oid: string, name: string) =>
  api.patch(`/projects/${pid}/videos/${vid}/objects/${oid}`, { name })

export const removeObject = (pid: string, vid: string, oid: string) =>
  api.delete(`/projects/${pid}/videos/${vid}/objects/${oid}`)

// ─── Points / Masks ──────────────────────────────────────────────────────────

export interface AddPointsResponse {
  frame_idx: number
  masks: MaskData
  new_objects?: Array<{
    id: string
    name: string
    color: string
    description?: string
  }>
}

export const addPoints = (
  pid: string,
  vid: string,
  oid: string,
  frameIdx: number,
  points: [number, number][],
  labels: number[],
  anchorMode = false,
) =>
  api.post<AddPointsResponse>(
    `/projects/${pid}/videos/${vid}/objects/${oid}/points`,
    { frame_idx: frameIdx, points, labels, anchor_mode: anchorMode }
  ).then(r => r.data)

export const clearObjectPoints = (pid: string, vid: string, oid: string) =>
  api.post(`/projects/${pid}/videos/${vid}/objects/${oid}/clear`)

export const clearObjectFramePoints = (pid: string, vid: string, oid: string, frameIdx: number) =>
  api.delete(`/projects/${pid}/videos/${vid}/objects/${oid}/frames/${frameIdx}/points`)

// ─── Saved Masks ─────────────────────────────────────────────────────────────

export const clearFrameMasks = (pid: string, vid: string, fidx: number) =>
  api.delete<{ status: string; frame_idx: number; deleted: string[] }>(`/projects/${pid}/videos/${vid}/masks/${fidx}`).then(r => r.data)

export type ClearMasksMode = 'all' | 'from_frame' | 'to_frame' | 'range'
export const clearMasksBulk = (
  pid: string, vid: string,
  mode: ClearMasksMode,
  from_frame?: number,
  to_frame?: number,
) =>
  api.delete<{ status: string; mode: string; deleted_frames: number }>(
    `/projects/${pid}/videos/${vid}/masks`,
    { params: { mode, ...(from_frame != null ? { from_frame } : {}), ...(to_frame != null ? { to_frame } : {}) } },
  ).then(r => r.data)

export const swapObjectMasks = (
  pid: string, vid: string,
  objA: string, objB: string,
  fromFrame?: number, toFrame?: number,
) =>
  api.post<{ status: string; frames_swapped: number }>(
    `/projects/${pid}/videos/${vid}/masks/swap`,
    { obj_a: objA, obj_b: objB, ...(fromFrame != null ? { from_frame: fromFrame } : {}), ...(toFrame != null ? { to_frame: toFrame } : {}) }
  ).then(r => r.data)

export const getSavedMask = (pid: string, vid: string, fidx: number) =>
  api.get<{ frame_idx: number; masks: MaskData }>(`/projects/${pid}/videos/${vid}/masks/${fidx}`)
    .then(r => r.data)

/**
 * Predict masks for all objects on a single frame using the annotated inference
 * state (all labeled anchor frames), WITHOUT persisting them or adding to the
 * inference state (preview only).
 */
export const predictFrame = (pid: string, vid: string, frameIdx: number) =>
  api.post<{ frame_idx: number; masks: MaskData }>(
    `/projects/${pid}/videos/${vid}/predict_frame/${frameIdx}`,
  ).then(r => r.data)

export interface RebuildSessionResponse {
  status: string
  masks_by_frame: Record<string, MaskData>
}

export const rebuildFromConfig = (
  pid: string,
  vid: string,
  returnMasksForFrames: number[],
  anchorMode = false,
  anchorFrame: number | null = null,
) =>
  api
    .post<RebuildSessionResponse>(`/projects/${pid}/videos/${vid}/session/rebuild_from_config`, {
      return_masks_for_frames: returnMasksForFrames,
      anchor_mode: anchorMode,
      ...(anchorMode && anchorFrame != null ? { anchor_frame: anchorFrame } : {}),
    })
    .then(r => r.data)

export const replaceFramePromptsData = (
  pid: string,
  vid: string,
  oid: string,
  frameIdx: number,
  points: [number, number][],
  labels: number[],
) =>
  api
    .put<{ status: string }>(
      `/projects/${pid}/videos/${vid}/objects/${oid}/frames/${frameIdx}/prompts`,
      { points, labels },
    )
    .then(r => r.data)

export const restoreMaskFrames = (pid: string, vid: string, frames: Record<string, MaskData>) =>
  api
    .post<{ status: string; restored: number }>(
      `/projects/${pid}/videos/${vid}/masks/restore_frames`,
      { frames },
    )
    .then(r => r.data)

export const restoreObjectSnapshot = (
  pid: string,
  vid: string,
  object: { id: string; name: string; color: string; description?: string; min_instances?: number; max_instances?: number },
  point_prompts: Record<string, { points: [number, number][]; labels: number[] }>,
  instance_group?: number[] | null,
) =>
  api
    .post<{ status: string }>(`/projects/${pid}/videos/${vid}/objects/restore`, {
      object,
      point_prompts,
      instance_group: instance_group ?? undefined,
    })
    .then(r => r.data)

// ─── Propagation SSE ─────────────────────────────────────────────────────────

export interface PropagationStatusResponse {
  is_running: boolean
  is_paused: boolean
  paused_at_frame: number
  frames_done: number
  total_frames: number
  propagation_complete: boolean
  last_frame: number
  start_frame: number
}

export const getPropagationStatus = (pid: string, vid: string) =>
  api.get<PropagationStatusResponse>(`/projects/${pid}/videos/${vid}/propagate/status`).then(r => r.data)

export const pausePropagation = (pid: string, vid: string) =>
  api.post<{ status: string; paused_at_frame: number }>(
    `/projects/${pid}/videos/${vid}/propagate/pause`
  ).then(r => r.data)

export const resumeFromFrame = (pid: string, vid: string, resumeFrame: number, clearFromFrame = true) =>
  api.post<{ status: string; resume_frame: number; deleted_files: number; kept_propagated_frames: number }>(
    `/projects/${pid}/videos/${vid}/propagate/resume`,
    { resume_frame: resumeFrame, clear_from_frame: clearFromFrame }
  ).then(r => r.data)

export const startPropagationSSE = (pid: string, vid: string, startFrame = 0, resumeFrom = -1, endFrame = -1, useAllAnchors = true) => {
  const params = new URLSearchParams({ start_frame: String(startFrame) })
  if (resumeFrom >= 0) params.set('resume_from', String(resumeFrom))
  if (endFrame >= 0) params.set('end_frame', String(endFrame))
  if (useAllAnchors) params.set('use_all_anchors', 'true')
  return new EventSource(`/api/projects/${pid}/videos/${vid}/propagate?${params}`)
}

// ─── Anchor Frames ────────────────────────────────────────────────────────────

export interface AnchorFramesResponse {
  anchor_frames: number[]
  count: number
  anchor_batch_size?: number
  manual_anchor_prefix_before_infer?: number
}

export const getAnchorFrames = (pid: string, vid: string) =>
  api.get<AnchorFramesResponse>(`/projects/${pid}/videos/${vid}/anchor_frames`).then(r => r.data)

export interface AnchorFrameLabelingTiming {
  entered_ms: number
  finished_ms: number
}

export const commitAnchorFrame = (
  pid: string,
  vid: string,
  frameIdx: number,
  anchorIndex: number,
  labelingTiming?: AnchorFrameLabelingTiming,
) =>
  api.post<{ status: string; committed_frame: number; anchor_index: number }>(
    `/projects/${pid}/videos/${vid}/anchors/${frameIdx}/commit`,
    {
      anchor_index: anchorIndex,
      labeling_timing: labelingTiming ?? undefined,
    },
  ).then(r => r.data)

export const startAnchorRemainderPredictionSSE = (pid: string, vid: string, interactive = true) => {
  const qs = interactive ? '?interactive=true' : '?interactive=false'
  return new EventSource(`${BACKEND}/api/projects/${pid}/videos/${vid}/anchors/predict_remainder_sse${qs}`)
}

export const continueAnchorRemainderReview = (pid: string, vid: string) =>
  api
    .post<{ status: string }>(`/projects/${pid}/videos/${vid}/anchors/predict_remainder_continue`)
    .then(r => r.data)

// ─── Export SSE ───────────────────────────────────────────────────────────────

export const startExportSSE = (pid: string, vid: string) =>
  new EventSource(`/api/projects/${pid}/videos/${vid}/export`)

// ─── Agentic prompting ───────────────────────────────────────────────────────

export interface AgentLlmStatus {
  configured: boolean
  provider: string
  model: string
  base_url: string | null
  missing_reason: string
  reachable?: boolean
  local?: boolean
  available_models?: string[]
  running?: boolean
  run_id?: string | null
  profile?: string
  recommended_model?: string
  profile_blurb?: string
}

export const getAgentStatus = (pid?: string) =>
  api.get<AgentLlmStatus>(pid ? `/projects/${pid}/agent/status` : '/agent/status').then(r => r.data)

export const cancelAgentRun = (pid: string) =>
  api.post<{ status: string; cancelled: boolean }>(`/projects/${pid}/agent/cancel`).then(r => r.data)

export interface AgentRunBody {
  message: string
  video_id?: string | null
  frame_idx?: number
  history?: { role: string; content: string }[]
}

export const textSegment = (
  pid: string,
  vid: string,
  frameIdx: number,
  text: string,
  opts?: { objectId?: string; objectName?: string; minScore?: number },
) =>
  api.post(`/projects/${pid}/videos/${vid}/text_segment`, {
    frame_idx: frameIdx,
    text,
    object_id: opts?.objectId,
    object_name: opts?.objectName,
    min_score: opts?.minScore ?? 0,
  }).then(r => r.data)

function emitParsedSse(
  chunk: string,
  onEvent: (event: string, data: Record<string, unknown>) => void,
) {
  const parsed = parseSseChunk(chunk)
  if (!parsed) return
  let data: Record<string, unknown> = {}
  try { data = JSON.parse(parsed.data) as Record<string, unknown> } catch { data = { raw: parsed.data } }
  onEvent(parsed.event, data)
}

export async function startAgentRun(
  pid: string,
  body: AgentRunBody,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`/api/projects/${pid}/agent/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok) {
    let detail = `Agent request failed (${res.status})`
    try {
      const j = await res.json()
      if (j?.detail) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch { /* ignore */ }
    throw new Error(detail)
  }
  if (!res.body) throw new Error('No response body from agent')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const parts = buf.split(SSE_EVENT_SPLIT)
    buf = parts.pop() ?? ''
    for (const chunk of parts) emitParsedSse(chunk, onEvent)
  }
  if (buf.trim()) emitParsedSse(buf, onEvent)
}
