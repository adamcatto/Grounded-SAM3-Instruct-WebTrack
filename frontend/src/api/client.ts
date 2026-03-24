import axios from 'axios'
import type { Project, VideoMeta, ObjectDef, MaskData, UncertaintyData, CorrectionRecord, ClassifierFramesResponse, ClassifierResults } from '../types'

// Direct backend URL for SSE/streaming (bypasses Vite dev proxy)
const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? 'http://localhost:8000'

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

export const createProject = (name: string) =>
  api.post<Project>('/projects', { name }).then(r => r.data)

export const getProject = (pid: string) =>
  api.get<Project>(`/projects/${pid}`).then(r => r.data)

export const updateProject = (pid: string, name: string) =>
  api.patch<Project>(`/projects/${pid}`, { name }).then(r => r.data)

export const deleteProject = (pid: string) =>
  api.delete(`/projects/${pid}`)

// ─── Videos ──────────────────────────────────────────────────────────────────

export const addVideo = (pid: string, file: File, onProgress?: (pct: number) => void) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<VideoMeta>(`/projects/${pid}/videos`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  }).then(r => r.data)
}

export const importVideo = (pid: string, serverPath: string) =>
  api.post<VideoMeta>(`/projects/${pid}/videos/import`, { path: serverPath }).then(r => r.data)

export const getVideo = (pid: string, vid: string) =>
  api.get<VideoMeta>(`/projects/${pid}/videos/${vid}`).then(r => r.data)

export const getVideoInfo = (pid: string, vid: string) =>
  api.get<{ num_frames: number; actual_frames: number; frames_extracted: boolean; fps: number; width: number; height: number }>(
    `/projects/${pid}/videos/${vid}/info`
  ).then(r => r.data)

export const updateVideoMeta = (pid: string, vid: string, updates: { start_frame?: number }) =>
  api.patch(`/projects/${pid}/videos/${vid}`, updates).then(r => r.data)

export const removeVideo = (pid: string, vid: string) =>
  api.delete(`/projects/${pid}/videos/${vid}`)

export const resetVideo = (pid: string, vid: string) =>
  api.post(`/projects/${pid}/videos/${vid}/reset`).then(r => r.data)

export const clearFramePrompts = (pid: string, vid: string, frameIdx: number) =>
  api.delete(`/projects/${pid}/videos/${vid}/frames/${frameIdx}/prompts`).then(r => r.data)

export const frameUrl = (pid: string, vid: string, fidx: number) =>
  `/api/projects/${pid}/videos/${vid}/frames/${fidx}`


// Alias for timeline thumbnails (same endpoint as frameUrl)
export const thumbUrl = frameUrl

// Video needs HTTP Range request support for seeking.  Vite's dev proxy
// re-chunks streaming responses and can break 206 Partial Content replies,
// so in dev we hit the backend directly (CORS is allow_origins=["*"]).
const BACKEND = import.meta.env.DEV
  ? (import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000')
  : ''

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
  frame_map: number[]             // real frame indices, indexed by SAM sequential idx
  cached_sam_indices: number[]    // SAM sequential indices with cached outputs
  action_history_len: number
  obj_ids_tracked: number[]
  point_prompts: Record<string, Record<string, { points: [number, number][]; labels: number[] }>>
  saved_mask_frames: number[]     // real frame indices with .npz on disk
  saved_mask_obj_counts: Record<number, number>
  annotated_frame_files: number[] // real frame indices in annotated_frames/
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
  minInstances?: number,
  maxInstances?: number,
) =>
  api.post<ObjectDef>(`/projects/${pid}/videos/${vid}/objects`, {
    name,
    color,
    description: description ?? '',
    min_instances: minInstances ?? 1,
    max_instances: maxInstances ?? 1,
  }).then(r => r.data)

export const updateObject = (
  pid: string,
  vid: string,
  oid: string,
  updates: { name?: string; color?: string; description?: string; min_instances?: number; max_instances?: number }
) =>
  api.patch(`/projects/${pid}/videos/${vid}/objects/${oid}`, updates).then(r => r.data)

export const renameObject = (pid: string, vid: string, oid: string, name: string) =>
  api.patch(`/projects/${pid}/videos/${vid}/objects/${oid}`, { name })

export const removeObject = (pid: string, vid: string, oid: string) =>
  api.delete(`/projects/${pid}/videos/${vid}/objects/${oid}`)

export const addInstance = (pid: string, vid: string, oid: string) =>
  api.post<{ sam_obj_id: number; ui_obj_id: string }>(
    `/projects/${pid}/videos/${vid}/objects/${oid}/instances`
  ).then(r => r.data)

export const getInstanceGroups = (pid: string, vid: string) =>
  api.get<Record<string, number[]>>(`/projects/${pid}/videos/${vid}/instance_groups`).then(r => r.data)

// ─── Points / Masks ──────────────────────────────────────────────────────────

export interface AddPointsResponse {
  frame_idx: number
  masks: MaskData
  new_objects?: Array<{
    id: string
    name: string
    color: string
    description?: string
    min_instances?: number
    max_instances?: number
    _parent_obj?: string
  }>
  uncertainty_update?: UncertaintyData  // Updated uncertainty data from annotation
}

export const addPoints = (
  pid: string,
  vid: string,
  oid: string,
  frameIdx: number,
  points: [number, number][],
  labels: number[],
  text?: string,
) =>
  api.post<AddPointsResponse>(
    `/projects/${pid}/videos/${vid}/objects/${oid}/points`,
    { frame_idx: frameIdx, points, labels, text }
  ).then(r => r.data)

export const clearObjectPoints = (pid: string, vid: string, oid: string) =>
  api.post(`/projects/${pid}/videos/${vid}/objects/${oid}/clear`)

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

export const getSavedMask = (pid: string, vid: string, fidx: number) =>
  api.get<{ frame_idx: number; masks: MaskData }>(`/projects/${pid}/videos/${vid}/masks/${fidx}`, {
    headers: { 'Cache-Control': 'no-cache' }
  }).then(r => r.data)

// ─── Single-frame prediction ──────────────────────────────────────────────────

export interface PredictFrameResult {
  frame_idx: number
  masks: MaskData
  used_prev_frame_mask: boolean
  prev_frame_idx: number | null
}

export const predictFrame = (pid: string, vid: string, frameIdx: number, usePrevFrameMask: boolean) =>
  api.post<PredictFrameResult>(
    `/projects/${pid}/videos/${vid}/predict_frame/${frameIdx}`,
    { use_prev_frame_mask: usePrevFrameMask }
  ).then(r => r.data)

export const saveFrameInference = (pid: string, vid: string, frameIdx: number) =>
  api.post<{ status: string; frame_idx: number; objects_saved: number }>(
    `/projects/${pid}/videos/${vid}/frames/${frameIdx}/save_inference`
  ).then(r => r.data)

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

// ─── Tracking Params ──────────────────────────────────────────────────────────

export interface TrackingParams {
  min_iou_threshold: number
  max_area_ratio: number
  max_centroid_jump: number
  consecutive_reject_limit: number
  propagation_mode: 'temporal_tracking' | 'per_frame' | 'dual_candidate'
}

export const getTrackingParams = (pid: string, vid: string) =>
  api.get<TrackingParams>(`/projects/${pid}/videos/${vid}/tracking_params`).then(r => r.data)

export const updateTrackingParams = (pid: string, vid: string, params: Partial<TrackingParams>) =>
  api.patch<TrackingParams>(`/projects/${pid}/videos/${vid}/tracking_params`, params).then(r => r.data)

// ─── Resume from Frame ────────────────────────────────────────────────────────

export interface ResumeFromFrameRequest {
  resume_frame: number
  clear_from_frame?: boolean
}

export interface ResumeFromFrameResponse {
  status: string
  resume_frame: number
  deleted_files: number
  kept_propagated_frames: number
}

export const resumeFromFrame = (pid: string, vid: string, resumeFrame: number, clearFromFrame = true) =>
  api.post<ResumeFromFrameResponse>(
    `/projects/${pid}/videos/${vid}/propagate/resume`,
    { resume_frame: resumeFrame, clear_from_frame: clearFromFrame }
  ).then(r => r.data)

export type SwapMode = 'this_frame' | 'all_future'

export interface SwapMasksResponse {
  frame_idx: number
  masks: MaskData
  swapped_frames: number
  swap_mode: SwapMode
}

export const swapMasks = (
  pid: string,
  vid: string,
  fidx: number,
  objIdA: string,
  objIdB: string,
  swapMode: SwapMode = 'this_frame'
) =>
  api.post<SwapMasksResponse>(
    `/projects/${pid}/videos/${vid}/masks/${fidx}/swap`,
    { obj_id_a: objIdA, obj_id_b: objIdB, swap_mode: swapMode }
  ).then(r => r.data)

// SSE stream for swapping all future frames (bypasses Vite proxy to avoid timeouts)
// ─── SSE helpers (use proxy-compatible relative URLs) ────────────────────────
// All SSE connections go through the Vite proxy (/api → localhost:8000) so
// they work regardless of whether port 8000 is directly reachable from the
// browser (e.g. VS Code port forwarding only exposes the dev-server port).
// Only videoSourceUrl uses BACKEND directly because video streaming requires
// HTTP Range requests which Vite's proxy may buffer incorrectly.

export const startSwapAllSSE = (
  pid: string,
  vid: string,
  fidx: number,
  objIdA: string,
  objIdB: string
) => {
  const params = new URLSearchParams({ obj_id_a: objIdA, obj_id_b: objIdB })
  return new EventSource(`/api/projects/${pid}/videos/${vid}/masks/${fidx}/swap_stream?${params}`)
}

export const startPropagationSSE = (pid: string, vid: string, startFrame = 0, resumeFrom = -1) => {
  const params = new URLSearchParams({ start_frame: String(startFrame) })
  if (resumeFrom >= 0) params.set('resume_from', String(resumeFrom))
  return new EventSource(`/api/projects/${pid}/videos/${vid}/propagate?${params}`)
}

// ─── Export SSE ───────────────────────────────────────────────────────────────

export const startExportSSE = (pid: string, vid: string) =>
  new EventSource(`/api/projects/${pid}/videos/${vid}/export`)

// ─── Uncertainty ──────────────────────────────────────────────────────────────

export const getUncertainty = (pid: string, vid: string) =>
  api.get<UncertaintyData>(`/projects/${pid}/videos/${vid}/uncertainty`).then(r => r.data)

export interface UncertaintyFrameDetail {
  frame_idx: number
  frame_image: string | null  // base64 JPEG
  masks: MaskData
  confusion_score: number
  per_object: Record<string, { anomaly_score?: number; overlap_ratio?: number; confidence?: number }>
}

export const getUncertaintyFrameDetail = (pid: string, vid: string, fidx: number) =>
  api.get<UncertaintyFrameDetail>(`/projects/${pid}/videos/${vid}/uncertainty/frames/${fidx}`).then(r => r.data)

// ─── Overlaps ──────────────────────────────────────────────────────────────────

export interface OcclusionWindow {
  pair: [number, number]
  onset_frame: number
  end_frame: number           // -1 if still active at propagation end
  frames: number[]
  swap_detected: boolean
  identity_map: Record<string, string> | null
  corrected: boolean
}

export interface OverlapsData {
  windows: OcclusionWindow[]
}

export const getOverlaps = (pid: string, vid: string) =>
  api.get<OverlapsData>(`/projects/${pid}/videos/${vid}/overlaps`).then(r => r.data)

// ─── Raw masks (pre-correction) ───────────────────────────────────────────────

export const getRawMasks = (pid: string, vid: string, fidx: number) =>
  api.get<{ frame_idx: number; masks: MaskData }>(`/projects/${pid}/videos/${vid}/masks/${fidx}/raw`).then(r => r.data)

// ─── Identity Corrections ─────────────────────────────────────────────────────

export const getCorrections = (pid: string, vid: string) =>
  api.get<CorrectionRecord[]>(`/projects/${pid}/videos/${vid}/corrections`).then(r => r.data)

export const applyCorrection = (
  pid: string,
  vid: string,
  correctionId: string,
  method: 'swap' | 'repropagate' = 'swap'
) =>
  api.post<{ status: string; method: string; affected_frames: number }>(
    `/projects/${pid}/videos/${vid}/corrections/${correctionId}/apply`,
    { method }
  ).then(r => r.data)

export const rejectCorrection = (pid: string, vid: string, correctionId: string) =>
  api.post<{ status: string }>(
    `/projects/${pid}/videos/${vid}/corrections/${correctionId}/reject`
  ).then(r => r.data)

// ─── Classifier ───────────────────────────────────────────────────────────────

// Legacy frame-fetch endpoint (kept for potential debugging use)
export const getClassifierFrames = (
  pid: string, vid: string,
  start: number, end: number,
  step: number, maxSize = 96,
) =>
  api.get<ClassifierFramesResponse>(
    `/projects/${pid}/videos/${vid}/classifier-frames`,
    { params: { start, end, step, max_size: maxSize } }
  ).then(r => r.data)

export interface ClassifierTrainParams {
  startFrame: number
  endFrame:   number
  step:       number
  epochs:     number
  lr:         number
  trainRatio: number
  evalRatio:  number
}

export type ClassifierSSEEvent =
  | { type: 'split';               train_frames: number[]; eval_frames: number[] }
  | { type: 'status';              message: string }
  | { type: 'epoch';               epoch: number; epochs: number; loss: number; train_acc: number }
  | { type: 'inference_progress';  done: number; total: number }
  | { type: 'done';                object_ids: string[]; frame_assignments: Record<string, Record<string, { predictedClass: string; confidence: number; scores: Record<string, number> }>>; train_frames: number[]; eval_frames: number[]; train_accuracy: number; eval_agreement: number }
  | { type: 'cancelled' }
  | { type: 'error';               message: string }

/**
 * Open an SSE connection to the backend classifier training endpoint.
 * Returns a cleanup function that closes the connection.
 */
export function openClassifierStream(
  pid: string,
  vid: string,
  params: ClassifierTrainParams,
  onEvent: (ev: ClassifierSSEEvent) => void,
): () => void {
  const qs = new URLSearchParams({
    start_frame:  String(params.startFrame),
    end_frame:    String(params.endFrame),
    step:         String(params.step),
    epochs:       String(params.epochs),
    lr:           String(params.lr),
    train_ratio:  String(params.trainRatio),
    eval_ratio:   String(params.evalRatio),
  })
  const url = `${BACKEND_URL}/api/projects/${pid}/videos/${vid}/classifier/train?${qs}`
  const es  = new EventSource(url)

  const handle = (type: string, raw: string) => {
    try {
      onEvent({ type, ...JSON.parse(raw) } as ClassifierSSEEvent)
    } catch {
      onEvent({ type: 'error', message: `Parse error for event "${type}"` })
    }
  }

  for (const t of ['split', 'status', 'epoch', 'inference_progress', 'done', 'cancelled', 'error']) {
    es.addEventListener(t, (e: MessageEvent) => {
      handle(t, e.data)
      if (t === 'done' || t === 'cancelled' || t === 'error') es.close()
    })
  }
  es.onerror = () => {
    onEvent({ type: 'error', message: 'SSE connection error' })
    es.close()
  }

  return () => es.close()
}

export const cancelClassifier = (pid: string, vid: string) =>
  api.post(`/projects/${pid}/videos/${vid}/classifier/cancel`)

export const getClassifierResults = (pid: string, vid: string) =>
  api.get<ClassifierResults>(`/projects/${pid}/videos/${vid}/classifier/results`).then(r => r.data)
