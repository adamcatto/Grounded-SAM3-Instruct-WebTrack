import axios from 'axios'
import type { Project, VideoMeta, ObjectDef, MaskData } from '../types'

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

export const removeVideo = (pid: string, vid: string) =>
  api.delete(`/projects/${pid}/videos/${vid}`)

export const resetVideo = (pid: string, vid: string) =>
  api.post(`/projects/${pid}/videos/${vid}/reset`).then(r => r.data)

export const frameUrl = (pid: string, vid: string, fidx: number) =>
  `/api/projects/${pid}/videos/${vid}/frames/${fidx}`

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

// ─── Objects ──────────────────────────────────────────────────────────────────

export const addObject = (pid: string, vid: string, name: string, color?: string) =>
  api.post<ObjectDef>(`/projects/${pid}/videos/${vid}/objects`, { name, color }).then(r => r.data)

export const renameObject = (pid: string, vid: string, oid: string, name: string) =>
  api.patch(`/projects/${pid}/videos/${vid}/objects/${oid}`, { name })

export const removeObject = (pid: string, vid: string, oid: string) =>
  api.delete(`/projects/${pid}/videos/${vid}/objects/${oid}`)

// ─── Points / Masks ──────────────────────────────────────────────────────────

export const addPoints = (
  pid: string,
  vid: string,
  oid: string,
  frameIdx: number,
  points: [number, number][],
  labels: number[]
) =>
  api.post<{ frame_idx: number; masks: MaskData }>(
    `/projects/${pid}/videos/${vid}/objects/${oid}/points`,
    { frame_idx: frameIdx, points, labels }
  ).then(r => r.data)

export const clearObjectPoints = (pid: string, vid: string, oid: string) =>
  api.post(`/projects/${pid}/videos/${vid}/objects/${oid}/clear`)

// ─── Saved Masks ─────────────────────────────────────────────────────────────

export const getSavedMask = (pid: string, vid: string, fidx: number) =>
  api.get<{ frame_idx: number; masks: MaskData }>(`/projects/${pid}/videos/${vid}/masks/${fidx}`).then(r => r.data)

// ─── Propagation SSE ─────────────────────────────────────────────────────────

// SSE also benefits from bypassing Vite's proxy to avoid buffering/re-chunking
export const startPropagationSSE = (pid: string, vid: string, startFrame = 0) =>
  new EventSource(`${BACKEND}/api/projects/${pid}/videos/${vid}/propagate?start_frame=${startFrame}`)

// ─── Export SSE ───────────────────────────────────────────────────────────────

export const startExportSSE = (pid: string, vid: string) =>
  new EventSource(`${BACKEND}/api/projects/${pid}/videos/${vid}/export`)
