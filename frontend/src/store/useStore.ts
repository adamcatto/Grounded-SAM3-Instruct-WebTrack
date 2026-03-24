import { create } from 'zustand'
import type { Project, VideoMeta, ObjectDef, MaskData, UncertaintyData, CorrectionRecord, ClassifierResults } from '../types'
import { evictMaskImages } from '../utils/maskUtils'

// Max number of frames to keep in the in-memory mask cache.
// Each frame holds N base64-encoded PNGs (~50 KB each compressed).
// The corresponding decoded ImageBitmaps are bounded separately in maskUtils.
const MAX_SAVED_MASK_FRAMES = 200

// Insertion-order tracking for FIFO eviction of savedMaskCache.
// Module-level (not in Zustand state) since it's purely an implementation detail.
let _savedMaskCacheOrder: number[] = []

export type PointMode = 'add' | 'remove' | null
export type PropagationStatus = 'idle' | 'running' | 'paused' | 'done' | 'error'
export type AppStep = 'upload' | 'annotate' | 'review'
export type ViewerTab = 'annotate' | 'player' | 'inference' | 'config' | 'uncertainty' | 'corrections' | 'overlaps' | 'classifier'

// ─── App config ───────────────────────────────────────────────────────────────

export interface AppConfig {
  usePrevFrameMask: boolean
  showMasks: boolean
  maskOpacity: number   // 0–1
  pointSize: number     // scale factor relative to default (1.0)
  correctionMethod: 'swap' | 'repropagate'  // default method for applying identity corrections
}

const CONFIG_KEY = 'sam3wt_config'

const CONFIG_DEFAULTS: AppConfig = {
  usePrevFrameMask: true,
  showMasks: true,
  maskOpacity: 0.85,
  pointSize: 1.0,
  correctionMethod: 'swap',
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
  pendingInferenceFrame: number | null    // frame awaiting "save to inference state" confirmation

  // Playback
  isPlaying: boolean

  // Propagation
  propagationStatus: PropagationStatus
  propagationProgress: number
  propagationCurrentFrame: number
  propagationStartFrame: number
  propagationPausedAtFrame: number

  // UI
  viewerTab: ViewerTab
  drawerOpen: boolean
  uploadModalOpen: boolean
  sessionInitialized: boolean
  frameJump: number
  setFrameJump: (n: number) => void

  // Identity tracking
  uncertaintyData: UncertaintyData | null
  corrections: CorrectionRecord[]
  setUncertaintyData: (data: UncertaintyData | null) => void
  setCorrections: (records: CorrectionRecord[]) => void

  // Classifier
  classifierResults: ClassifierResults | null
  showClassifierOverlay: boolean
  setClassifierResults: (r: ClassifierResults | null) => void
  setShowClassifierOverlay: (v: boolean) => void

  // Config
  config: AppConfig
  configDirty: boolean         // true when in-memory config differs from last localStorage save
  setConfig: (updates: Partial<AppConfig>) => void
  persistConfig: () => void    // write current config to localStorage
  revertConfig: () => void     // restore config + propagationStartFrame from localStorage

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
  setCurrentFrameMasks: (masks: MaskData, frame?: number | null) => void
  setSavedMask: (fidx: number, masks: MaskData) => void
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
  // Starts at 0; overwritten by setCurrentVideo using the per-video start_frame from config.json
  propagationStartFrame: 0,
  propagationPausedAtFrame: -1,
  drawerOpen: false,
  uploadModalOpen: false,
  viewerTab: 'annotate' as ViewerTab,
  sessionInitialized: false,
  frameJump: 1,
  setFrameJump: (n: number) => set({ frameJump: Math.max(1, Math.round(n)) }),
  uncertaintyData: null,
  corrections: [],
  config: _initialConfig,
  configDirty: false,
  toasts: [],

  setUncertaintyData: data => set({ uncertaintyData: data }),
  setCorrections: records => set({ corrections: records }),

  classifierResults: null,
  showClassifierOverlay: false,
  setClassifierResults: r => set({ classifierResults: r }),
  setShowClassifierOverlay: v => set({ showClassifierOverlay: v }),

  setProject: p => set({ project: p }),

  setCurrentVideo: vid => {
    const prev = get().currentVideoId
    if (prev !== vid) {
      const vidData = vid ? get().project?.videos[vid] : undefined
      const alreadyPropagated = vidData?.propagation_complete ?? false

      // Restore saved point prompts into localAnnotations so annotated frames
      // show their points immediately without requiring a page session.
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
      // Use the per-video persisted start frame; fall back to 0
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
        uncertaintyData: null,
        corrections: [],
      })
    }
  },

  // IMPORTANT: Never allow currentFrame to go below propagationStartFrame
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

  setCurrentFrameMasks: (masks, frame) => set({ currentFrameMasks: masks, currentFrameMasksFrame: frame ?? null }),

  setSavedMask: (fidx, masks) => {
    const { savedMaskCache } = get()

    // Remove fidx from order tracking if already present (re-insert at end)
    _savedMaskCacheOrder = _savedMaskCacheOrder.filter(f => f !== fidx)
    _savedMaskCacheOrder.push(fidx)

    const newCache = { ...savedMaskCache, [fidx]: masks }

    // Evict oldest frames when over the limit
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

  setPendingInferenceFrame: f => set({ pendingInferenceFrame: f }),

  setPlaying: v => set({ isPlaying: v }),

  setPropagationStatus: s => set({ propagationStatus: s }),

  setPropagationProgress: (p, frame) =>
    set({ propagationProgress: p, propagationCurrentFrame: frame }),

  setPropagationPausedAtFrame: f => set({ propagationPausedAtFrame: f }),

  setDrawerOpen: v => set({ drawerOpen: v }),
  setUploadModalOpen: v => set({ uploadModalOpen: v }),
  setViewerTab: tab => set({ viewerTab: tab }),
  setSessionInitialized: v => set({ sessionInitialized: v }),

  // setConfig: update in-memory only, mark dirty. Does NOT auto-save to localStorage.
  setConfig: updates => {
    const next = { ...get().config, ...updates }
    set({ config: next, configDirty: true })
  },

  // persistConfig: write current in-memory config to localStorage, clear dirty flag.
  persistConfig: () => {
    const { config } = get()
    writeConfig(config)
    set({ config, configDirty: false })
  },

  // revertConfig: reload the last saved config from localStorage, clear dirty flag.
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
      propagationStartFrame: 0,  // Reset start frame to 0 when starting over
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
    })
  },
}))

// Derived selectors
export const currentVideo = (state: AppState) =>
  state.project && state.currentVideoId
    ? state.project.videos[state.currentVideoId] ?? null
    : null
