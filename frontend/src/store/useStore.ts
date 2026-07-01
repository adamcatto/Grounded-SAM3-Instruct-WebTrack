import { create } from 'zustand'
import type { Project, VideoMeta, MaskData } from '../types'
import { evictMaskImages } from '../utils/maskUtils'
import { invalidateMaskLoaderFrame } from '../utils/maskLoader'
import { clearCompositeCache } from '../utils/compositeMaskCache'
import type { HistoryCommand } from '../history/undoHistory'
import { MAX_UNDO_STACK } from '../history/undoHistory'

// Max number of frames to keep in the in-memory mask cache.
const MAX_SAVED_MASK_FRAMES = 40

// Insertion-order tracking for FIFO eviction of savedMaskCache.
let _savedMaskCacheOrder: number[] = []

export type PointMode = 'add' | 'remove' | null
export type PropagationStatus = 'idle' | 'running' | 'paused' | 'done' | 'error'
export type AppStep = 'upload' | 'annotate' | 'review'
export type ViewerTab = 'annotate' | 'player' | 'inference' | 'config'

// ─── App config ───────────────────────────────────────────────────────────────

export interface AppConfig {
  usePrevFrameMask: boolean
  showMasks: boolean
  maskOpacity: number   // 0–1
  pointSize: number     // scale factor relative to default (1.0)
  useAllAnchors: boolean  // default tracking method when anchor frames are labeled
  /** After the manual anchor prefix, auto-run SAM to fill remaining anchors (SSE job). Default off. */
  autoInferAnchorRemainder: boolean
  /** Default anchor / propagation batch interval (frames) for new videos. */
  anchorBatchSize: number
}

const CONFIG_KEY = 'sam3wt_config'

const CONFIG_DEFAULTS: AppConfig = {
  usePrevFrameMask: true,
  showMasks: true,
  maskOpacity: 0.85,
  pointSize: 1.0,
  useAllAnchors: true,
  autoInferAnchorRemainder: false,
  anchorBatchSize: 1000,
}

function loadConfig(): AppConfig {
  try {
    const raw = localStorage.getItem(CONFIG_KEY)
    if (raw) return { ...CONFIG_DEFAULTS, ...JSON.parse(raw) }
  } catch { /* ignore */ }
  return { ...CONFIG_DEFAULTS }
}

function writeConfig(c: AppConfig) {
  try { localStorage.setItem(CONFIG_KEY, JSON.stringify(c)) } catch { /* ignore */ }
}

// ─── Toasts ───────────────────────────────────────────────────────────────────

export interface Toast {
  id: string
  message: string
  type: 'info' | 'success' | 'error'
}

interface LocalAnnotation {
  points: { x: number; y: number; label: 0 | 1 }[]
}

interface AppState {
  // Project / video state
  project: Project | null
  currentVideoId: string | null

  // Annotation state
  currentFrame: number
  currentObjectId: string | null
  pointMode: PointMode
  localAnnotations: Record<string, Record<string, LocalAnnotation>>  // objId → frameIdx → pts

  // Mask cache: frameIdx → objId → base64 PNG
  currentFrameMasks: MaskData
  currentFrameMasksFrame: number | null   // which frame currentFrameMasks belongs to
  savedMaskCache: Record<number, MaskData>
  pendingInferenceFrame: number | null

  // Playback
  isPlaying: boolean

  // Propagation
  propagationStatus: PropagationStatus
  propagationProgress: number
  propagationCurrentFrame: number
  propagationStartFrame: number
  propagationPausedAtFrame: number

  // Anchor annotation phase
  anchorPhase: boolean                                   // true during anchor annotation
  anchorFrames: number[]                                 // [start, start+1000, ..., last]
  currentAnchorIndex: number                             // which anchor user is on (0-based)
  annotatedAnchorIndices: number[]                       // which anchor indices have been committed
  anchorRemainderInferencing: boolean                   // SAM auto-filling remaining anchors
  /** User edited a predicted anchor and must commit with Done, next */
  anchorRemainderAwaitingCommit: boolean

  // UI
  viewerTab: ViewerTab
  drawerOpen: boolean
  uploadModalOpen: boolean
  sessionInitialized: boolean
  frameJump: number
  setFrameJump: (n: number) => void

  // Config
  config: AppConfig
  configDirty: boolean
  setConfig: (updates: Partial<AppConfig>) => void
  persistConfig: () => void
  revertConfig: () => void

  // Toasts
  toasts: Toast[]
  addToast: (message: string, type?: Toast['type']) => void
  removeToast: (id: string) => void

  // Actions
  setProject: (p: Project | null) => void
  setCurrentVideo: (vid: string | null) => void
  setCurrentFrame: (f: number) => void
  setCurrentObject: (oid: string | null) => void
  setPointMode: (m: PointMode) => void
  addLocalPoint: (objId: string, frameIdx: number, x: number, y: number, label: 0 | 1) => void
  clearLocalPoints: (objId: string) => void
  clearLocalPointsForFrame: (objId: string, frameIdx: number) => void
  setCurrentFrameMasks: (masks: MaskData, frame?: number | null) => void
  setSavedMask: (fidx: number, masks: MaskData) => void
  clearSavedMaskCache: () => void
  setPendingInferenceFrame: (f: number | null) => void
  setPlaying: (v: boolean) => void
  setPropagationStatus: (s: PropagationStatus) => void
  setPropagationProgress: (p: number, frame: number) => void
  setPropagationPausedAtFrame: (f: number) => void
  setDrawerOpen: (v: boolean) => void
  setUploadModalOpen: (v: boolean) => void
  setViewerTab: (tab: ViewerTab) => void
  setSessionInitialized: (v: boolean) => void
  setPropagationStartFrame: (f: number) => void
  updateVideo: (updates: Partial<VideoMeta>) => void
  resetVideoState: () => void

  // Anchor phase actions
  setAnchorPhase: (v: boolean) => void
  setAnchorFrames: (frames: number[]) => void
  setCurrentAnchorIndex: (i: number) => void
  addAnnotatedAnchor: (index: number) => void
  setAnnotatedAnchorIndices: (indices: number[]) => void
  setAnchorRemainderInferencing: (v: boolean) => void
  setAnchorRemainderAwaitingCommit: (v: boolean) => void
  invalidateSavedMaskFrame: (fidx: number) => void
  resetAnchorState: () => void

  // Undo / redo (annotation actions)
  undoStack: HistoryCommand[]
  redoStack: HistoryCommand[]
  historyBusy: boolean
  pushHistory: (cmd: HistoryCommand) => void
  undoLast: () => Promise<void>
  redoLast: () => Promise<void>
  clearHistory: () => void
}

const _initialConfig = loadConfig()

export const useStore = create<AppState>((set, get) => ({
  project: null,
  currentVideoId: null,
  currentFrame: 0,
  currentObjectId: null,
  pointMode: null,
  localAnnotations: {},
  currentFrameMasks: {},
  currentFrameMasksFrame: null,
  savedMaskCache: {},
  pendingInferenceFrame: null,
  isPlaying: false,
  propagationStatus: 'idle',
  propagationProgress: 0,
  propagationCurrentFrame: 0,
  propagationStartFrame: 0,
  propagationPausedAtFrame: -1,
  drawerOpen: false,
  uploadModalOpen: false,
  viewerTab: 'annotate' as ViewerTab,
  sessionInitialized: false,
  frameJump: 1,
  setFrameJump: (n: number) => set({ frameJump: Math.max(1, Math.round(n)) }),

  // Anchor phase state
  anchorPhase: false,
  anchorFrames: [],
  currentAnchorIndex: 0,
  annotatedAnchorIndices: [],
  anchorRemainderInferencing: false,
  anchorRemainderAwaitingCommit: false,

  config: _initialConfig,
  configDirty: false,
  toasts: [],
  undoStack: [],
  redoStack: [],
  historyBusy: false,

  clearHistory: () => set({ undoStack: [], redoStack: [] }),

  pushHistory: cmd => {
    const { undoStack } = get()
    const next = [...undoStack, cmd].slice(-MAX_UNDO_STACK)
    set({ undoStack: next, redoStack: [] })
  },

  undoLast: async () => {
    const { undoStack, historyBusy } = get()
    if (historyBusy || undoStack.length === 0) return
    const cmd = undoStack[undoStack.length - 1]
    set({ historyBusy: true })
    try {
      await cmd.undo()
      set(s => ({
        undoStack: s.undoStack.slice(0, -1),
        redoStack: [...s.redoStack, cmd],
      }))
      get().addToast(`Undid: ${cmd.labelUndo}`, 'info')
    } catch (e) {
      console.error('undo failed', e)
      get().addToast('Undo failed', 'error')
    } finally {
      set({ historyBusy: false })
    }
  },

  redoLast: async () => {
    const { redoStack, historyBusy } = get()
    if (historyBusy || redoStack.length === 0) return
    const cmd = redoStack[redoStack.length - 1]
    set({ historyBusy: true })
    try {
      await cmd.redo()
      set(s => ({
        redoStack: s.redoStack.slice(0, -1),
        undoStack: [...s.undoStack, cmd],
      }))
      get().addToast(`Redid: ${cmd.labelRedo}`, 'info')
    } catch (e) {
      console.error('redo failed', e)
      get().addToast('Redo failed', 'error')
    } finally {
      set({ historyBusy: false })
    }
  },

  setProject: p => {
    const prevId = get().project?.id
    const nextId = p?.id
    if (prevId !== nextId) {
      set({ project: p, undoStack: [], redoStack: [] })
    } else {
      set({ project: p })
    }
  },

  setCurrentVideo: vid => {
    const prev = get().currentVideoId
    if (prev !== vid) {
      const vidData = vid ? get().project?.videos[vid] : undefined
      const alreadyPropagated = vidData?.propagation_complete ?? false

      // Restore saved point prompts into localAnnotations
      const localAnnotations: Record<string, Record<string, LocalAnnotation>> = {}
      if (vidData?.point_prompts) {
        for (const [objId, framePts] of Object.entries(vidData.point_prompts)) {
          localAnnotations[objId] = {}
          for (const [frameIdx, { points, labels }] of Object.entries(framePts)) {
            localAnnotations[objId][frameIdx] = {
              points: points.map(([x, y], i) => ({ x, y, label: labels[i] as 0 | 1 })),
            }
          }
        }
      }

      _savedMaskCacheOrder = []
      const startFrame = vidData?.start_frame ?? 0
      set({
        currentVideoId: vid,
        currentFrame: startFrame,
        propagationStartFrame: startFrame,
        currentObjectId: null,
        pointMode: null,
        localAnnotations,
        currentFrameMasks: {},
        currentFrameMasksFrame: null,
        savedMaskCache: {},
        isPlaying: false,
        propagationStatus: alreadyPropagated ? 'done' : 'idle',
        propagationProgress: alreadyPropagated ? 1 : 0,
        sessionInitialized: false,
        // Reset anchor state on video switch
        anchorPhase: false,
        anchorFrames: [],
        currentAnchorIndex: 0,
        annotatedAnchorIndices: [],
        anchorRemainderInferencing: false,
        anchorRemainderAwaitingCommit: false,
        undoStack: [],
        redoStack: [],
      })
    }
  },

  // Never allow currentFrame to go below propagationStartFrame
  setCurrentFrame: f => {
    const { propagationStartFrame } = get()
    const clamped = Math.max(f, propagationStartFrame)
    set({ currentFrame: clamped })
  },
  setCurrentObject: oid => set({ currentObjectId: oid, pointMode: oid ? 'add' : null }),
  setPointMode: m => set({ pointMode: m }),

  addLocalPoint: (objId, frameIdx, x, y, label) => {
    const { localAnnotations } = get()
    const key = String(frameIdx)
    const existing = localAnnotations[objId]?.[key] ?? { points: [] }
    set({
      localAnnotations: {
        ...localAnnotations,
        [objId]: {
          ...(localAnnotations[objId] ?? {}),
          [key]: { points: [...existing.points, { x, y, label }] },
        },
      },
    })
  },

  clearLocalPoints: objId => {
    const { localAnnotations } = get()
    const updated = { ...localAnnotations }
    delete updated[objId]
    set({ localAnnotations: updated })
  },

  clearLocalPointsForFrame: (objId, frameIdx) => {
    const { localAnnotations } = get()
    const objFrames = localAnnotations[objId]
    if (!objFrames) return
    const updated = { ...objFrames }
    delete updated[String(frameIdx)]
    set({ localAnnotations: { ...localAnnotations, [objId]: updated } })
  },

  setCurrentFrameMasks: (masks, frame) => set({ currentFrameMasks: masks, currentFrameMasksFrame: frame ?? null }),

  setSavedMask: (fidx, masks) => {
    const { savedMaskCache } = get()

    _savedMaskCacheOrder = _savedMaskCacheOrder.filter(f => f !== fidx)
    _savedMaskCacheOrder.push(fidx)

    const newCache = { ...savedMaskCache, [fidx]: masks }

    const b64sToEvict: string[] = []
    while (_savedMaskCacheOrder.length > MAX_SAVED_MASK_FRAMES) {
      const oldest = _savedMaskCacheOrder.shift()!
      const oldMasks = newCache[oldest]
      if (oldMasks) {
        b64sToEvict.push(...Object.values(oldMasks))
        delete newCache[oldest]
      }
    }
    if (b64sToEvict.length > 0) evictMaskImages(b64sToEvict)

    set({ savedMaskCache: newCache })
  },

  clearSavedMaskCache: () => {
    const { savedMaskCache } = get()
    const b64s = Object.values(savedMaskCache).flatMap(m => Object.values(m))
    if (b64s.length > 0) evictMaskImages(b64s)
    clearCompositeCache()
    _savedMaskCacheOrder = []
    set({ savedMaskCache: {} })
  },

  setPendingInferenceFrame: f => set({ pendingInferenceFrame: f }),
  setPlaying: v => set({ isPlaying: v }),
  setPropagationStatus: s => set({ propagationStatus: s }),
  setPropagationProgress: (p, frame) => set({ propagationProgress: p, propagationCurrentFrame: frame }),
  setPropagationPausedAtFrame: f => set({ propagationPausedAtFrame: f }),
  setDrawerOpen: v => set({ drawerOpen: v }),
  setUploadModalOpen: v => set({ uploadModalOpen: v }),
  setViewerTab: tab => set({ viewerTab: tab }),
  setSessionInitialized: v => set({ sessionInitialized: v }),

  setConfig: updates => {
    const next = { ...get().config, ...updates }
    set({ config: next, configDirty: true })
  },

  persistConfig: () => {
    const { config } = get()
    writeConfig(config)
    set({ config, configDirty: false })
  },

  revertConfig: () => {
    const saved = loadConfig()
    set({ config: saved, configDirty: false })
  },

  addToast: (message, type = 'info') => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`
    set(s => ({ toasts: [...s.toasts, { id, message, type }] }))
  },

  removeToast: id => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })),

  setPropagationStartFrame: f => set({ propagationStartFrame: f }),

  updateVideo: updates => {
    const { project, currentVideoId } = get()
    if (!project || !currentVideoId) return
    set({
      project: {
        ...project,
        videos: {
          ...project.videos,
          [currentVideoId]: {
            ...project.videos[currentVideoId],
            ...updates,
          },
        },
      },
    })
  },

  resetVideoState: () => {
    _savedMaskCacheOrder = []
    set({
      currentFrame: 0,
      propagationStartFrame: 0,
      currentObjectId: null,
      pointMode: null,
      localAnnotations: {},
      currentFrameMasks: {},
      currentFrameMasksFrame: null,
      savedMaskCache: {},
      pendingInferenceFrame: null,
      isPlaying: false,
      propagationStatus: 'idle',
      propagationProgress: 0,
      propagationPausedAtFrame: -1,
      sessionInitialized: false,
      anchorPhase: false,
      anchorFrames: [],
      currentAnchorIndex: 0,
      annotatedAnchorIndices: [],
      anchorRemainderInferencing: false,
      anchorRemainderAwaitingCommit: false,
    })
  },

  // Anchor phase actions
  setAnchorPhase: v => set({ anchorPhase: v }),
  setAnchorFrames: frames => set({ anchorFrames: frames }),
  setCurrentAnchorIndex: i => set({ currentAnchorIndex: i }),
  addAnnotatedAnchor: index => {
    const { annotatedAnchorIndices } = get()
    if (!annotatedAnchorIndices.includes(index)) {
      set({ annotatedAnchorIndices: [...annotatedAnchorIndices, index] })
    }
  },
  setAnnotatedAnchorIndices: indices =>
    set({ annotatedAnchorIndices: [...new Set(indices)].sort((a, b) => a - b) }),
  setAnchorRemainderInferencing: v => set({ anchorRemainderInferencing: v }),
  setAnchorRemainderAwaitingCommit: v => set({ anchorRemainderAwaitingCommit: v }),
  invalidateSavedMaskFrame: fidx => {
    const { savedMaskCache, project, currentVideoId } = get()
    const pid = project?.id
    const vid = currentVideoId
    if (pid && vid) invalidateMaskLoaderFrame(pid, vid, fidx)
    if (savedMaskCache[fidx] === undefined) return
    const b64s = Object.values(savedMaskCache[fidx] ?? {})
    const next = { ...savedMaskCache }
    delete next[fidx]
    _savedMaskCacheOrder = _savedMaskCacheOrder.filter(f => f !== fidx)
    if (b64s.length > 0) evictMaskImages(b64s)
    set({ savedMaskCache: next })
  },
  resetAnchorState: () => set({
    anchorPhase: false,
    anchorFrames: [],
    currentAnchorIndex: 0,
    annotatedAnchorIndices: [],
    anchorRemainderInferencing: false,
    anchorRemainderAwaitingCommit: false,
  }),
}))

// Derived selectors
export const currentVideo = (state: AppState) =>
  state.project && state.currentVideoId
    ? state.project.videos[state.currentVideoId] ?? null
    : null
