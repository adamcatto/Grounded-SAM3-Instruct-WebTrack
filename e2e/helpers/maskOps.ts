/** Reversible API helpers for sandbox mask-op tests. */

export const SANDBOX = {
  pid: '9f8a7b6c',
  vid: 'f09434ac',
  anchorFrame: 160,
  nonAnchorFrame: 500,
  objA: '1',
  objB: '2',
  videoName: '1A_video_test1a_20260130_090255.mp4',
  projectName: 'sandbox_Home-Cage-Interactions-0126-test-day',
} as const

const API = 'http://localhost:8000/api'

export type MaskData = Record<string, string>

export interface FrameSnapshot {
  masks: MaskData
  hasMasks: boolean
}

export interface PromptSnapshot {
  points: number[][]
  labels: number[]
}

export interface ProjectSnapshot {
  frames: Record<number, FrameSnapshot>
  pointPrompts: Record<string, Record<string, PromptSnapshot>>
}

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${init?.method ?? 'GET'} ${path} → ${res.status}: ${body}`)
  }
  return res.json() as Promise<T>
}

export async function getSavedMasks(pid: string, vid: string, frame: number): Promise<FrameSnapshot> {
  const data = await apiJson<{ masks?: MaskData }>(`/projects/${pid}/videos/${vid}/masks/${frame}`)
  const masks = data.masks ?? {}
  return { masks, hasMasks: Object.keys(masks).length > 0 }
}

export async function getPointPrompts(pid: string = SANDBOX.pid): Promise<Record<string, Record<string, PromptSnapshot>>> {
  const proj = await apiJson<{ videos: Record<string, { point_prompts?: Record<string, Record<string, PromptSnapshot>> }> }>(
    `/projects/${pid}`,
  )
  return proj.videos[SANDBOX.vid]?.point_prompts ?? {}
}

export async function snapshotFrames(frames: number[]): Promise<ProjectSnapshot> {
  const pointPrompts = await getPointPrompts(SANDBOX.pid)
  const out: Record<number, FrameSnapshot> = {}
  for (const f of frames) {
    out[f] = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, f)
  }
  return { frames: out, pointPrompts: JSON.parse(JSON.stringify(pointPrompts)) as ProjectSnapshot['pointPrompts'] }
}

export async function restoreSnapshot(snap: ProjectSnapshot, onlyFrames?: number[]): Promise<void> {
  const framesPayload: Record<string, MaskData> = {}
  for (const [f, data] of Object.entries(snap.frames)) {
    framesPayload[f] = data.masks
  }
  await apiJson(`/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/masks/restore_frames`, {
    method: 'POST',
    body: JSON.stringify({ frames: framesPayload }),
  })

  const frameKeys = new Set(
    (onlyFrames ?? Object.keys(snap.frames).map(Number)).map(String),
  )
  for (const [objId, frameMap] of Object.entries(snap.pointPrompts)) {
    for (const [frameStr, prompt] of Object.entries(frameMap)) {
      if (!frameKeys.has(frameStr)) continue
      await apiJson(`/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/objects/${objId}/frames/${frameStr}/prompts`, {
        method: 'PUT',
        body: JSON.stringify({ points: prompt.points, labels: prompt.labels }),
      })
    }
  }

  const current = await getPointPrompts(SANDBOX.pid)
  for (const [objId, frameMap] of Object.entries(current)) {
    for (const frameStr of Object.keys(frameMap)) {
      if (!frameKeys.has(frameStr)) continue
      if (!snap.pointPrompts[objId]?.[frameStr]) {
        await apiJson(
          `/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/objects/${objId}/frames/${frameStr}/prompts`,
          { method: 'PUT', body: JSON.stringify({ points: [], labels: [] }) },
        )
      }
    }
  }
}

export async function swapMasks(frame: number, objA = SANDBOX.objA, objB = SANDBOX.objB) {
  return apiJson<{ frames_swapped: number }>(`/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/masks/swap`, {
    method: 'POST',
    body: JSON.stringify({ obj_a: objA, obj_b: objB, from_frame: frame, to_frame: frame }),
  })
}

export async function clearObjectFrame(objId: string, frame: number) {
  await apiJson(`/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/objects/${objId}/frames/${frame}/points`, {
    method: 'DELETE',
  })
}

export async function clearFrameMasks(frame: number) {
  await apiJson(`/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/masks/${frame}`, { method: 'DELETE' })
}

/** After swap, object A's PNG should match what B had before (and vice versa). */
export function masksSwapped(
  before: MaskData,
  after: MaskData,
  objA: string,
  objB: string,
): boolean {
  if (!before[objA] || !before[objB]) return false
  return after[objA] === before[objB] && after[objB] === before[objA]
}

export function promptsUnchanged(
  before: Record<string, Record<string, PromptSnapshot>>,
  after: Record<string, Record<string, PromptSnapshot>>,
): boolean {
  return JSON.stringify(before) === JSON.stringify(after)
}

export async function waitForBackend(): Promise<void> {
  for (let i = 0; i < 60; i++) {
    try {
      const h = await apiJson<{ sam_loaded?: boolean }>('/health')
      if (h.sam_loaded) return
    } catch { /* retry */ }
    await new Promise(r => setTimeout(r, 2000))
  }
  throw new Error('Backend/SAM not ready after 120s')
}

export async function getSandboxProject(): Promise<{
  videos: Record<string, { objects?: Record<string, { id?: string; name?: string }> }>
}> {
  return apiJson(`/projects/${SANDBOX.pid}`)
}

export async function listSandboxObjectIds(): Promise<string[]> {
  const p = await getSandboxProject()
  return Object.keys(p.videos?.[SANDBOX.vid]?.objects ?? {})
}

export async function deleteSandboxObject(oid: string): Promise<void> {
  await apiJson(`/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/objects/${oid}`, { method: 'DELETE' })
}

export async function getAgentLlmStatus(): Promise<{
  configured?: boolean
  provider?: string
  model?: string
  reachable?: boolean
}> {
  return apiJson('/agent/status')
}

export async function getLastAgentDump(): Promise<Record<string, unknown>> {
  return apiJson('/agent/last_dump')
}

export async function resetSandboxVideo(keepObjects = true): Promise<{ status?: string; kept_object_ids?: string[] }> {
  const qs = keepObjects ? '?keep_objects=true' : ''
  return apiJson(`/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/reset${qs}`, { method: 'POST' })
}

export async function ensureSandboxObjects(): Promise<void> {
  const p = await getSandboxProject()
  const objs = p.videos?.[SANDBOX.vid]?.objects ?? {}
  const names = new Set(
    Object.values(objs).map(o => (o.name || '').toLowerCase()),
  )
  if (!names.has('headshave')) {
    await apiJson(`/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/objects`, {
      method: 'POST',
      body: JSON.stringify({ name: 'HeadShave', color: '#5B8DD9', description: 'shaved mouse' }),
    })
  }
  const p2 = await getSandboxProject()
  const names2 = new Set(
    Object.values(p2.videos?.[SANDBOX.vid]?.objects ?? {}).map(o => (o.name || '').toLowerCase()),
  )
  if (!names2.has('noshave')) {
    await apiJson(`/projects/${SANDBOX.pid}/videos/${SANDBOX.vid}/objects`, {
      method: 'POST',
      body: JSON.stringify({ name: 'NoShave', color: '#E8A445', description: 'unshaved mouse' }),
    })
  }
}

export async function sandboxObjectIdsByName(): Promise<{ headshave?: string; noshave?: string }> {
  const p = await getSandboxProject()
  const objs = p.videos?.[SANDBOX.vid]?.objects ?? {}
  const out: { headshave?: string; noshave?: string } = {}
  for (const [oid, obj] of Object.entries(objs)) {
    const n = (obj.name || '').toLowerCase()
    if (n === 'headshave') out.headshave = oid
    if (n === 'noshave') out.noshave = oid
  }
  return out
}
