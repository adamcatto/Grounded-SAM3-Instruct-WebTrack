import { create } from 'zustand'
import type { Project, VideoMeta, ObjectDef, MaskData } from '../types'

export type PointMode = 'add' | 'remove' | null
export type PropagationStatus = 'idle' | 'running' | 'done' | 'error'
export type AppStep = 'upload' | 'annotate' | 'review'
export type ViewerTab = 'annotate' | 'player' | 'inference' | 'config'

// ─── App config ───────────────────────────────────────────────────────────────

export interface AppConfig {
  usePrevFrameMask: boolean
  showMasks: boolean
  maskOpacity: number   // 0–1
  pointSize: number     // scale factor relative to default (1.0)
  startFrame: number    // persisted propagation start frame
}

const CONFIG_KEY = 'sam3wt_config'

const CONFIG_DEFAULTS: AppConfig = {
  usePrevFrameMask: true,
  showMasks: true,
  maskOpacity: 0.85,
  pointSize: 1.0,
  startFrame: 0,
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

  // UI
  viewerTab: ViewerTab
  drawerOpen: boolean
  uploadModalOpen: boolean
  sessionInitialized: boolean

  // Config
  config: AppConfig
  configDirty: boolean         // true when in-memory config differs from last localStorage save
  setConfig: (updates: Partial<AppConfig>) => void
  persistConfig: () => void    // write current config + propagationStartFrame to localStorage
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
  // Initialize from persisted config so it survives page refreshes
  propagationStartFrame: _initialConfig.startFrame,
  drawerOpen: false,
  uploadModalOpen: false,
  viewerTab: 'annotate' as ViewerTab,
  sessionInitialized: false,
  config: _initialConfig,
  configDirty: false,
  toasts: [],

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

      set({
        currentVideoId: vid,
        currentFrame: 0,
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
      })
    }
  },

  setCurrentFrame: f => set({ currentFrame: f }),
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
    set({ savedMaskCache: { ...savedMaskCache, [fidx]: masks } })
  },

  setPendingInferenceFrame: f => set({ pendingInferenceFrame: f }),

  setPlaying: v => set({ isPlaying: v }),

  setPropagationStatus: s => set({ propagationStatus: s }),

  setPropagationProgress: (p, frame) =>
    set({ propagationProgress: p, propagationCurrentFrame: frame }),

  setDrawerOpen: v => set({ drawerOpen: v }),
  setUploadModalOpen: v => set({ uploadModalOpen: v }),
  setViewerTab: tab => set({ viewerTab: tab }),
  setSessionInitialized: v => set({ sessionInitialized: v }),

  // setConfig: update in-memory only, mark dirty. Does NOT auto-save to localStorage.
  setConfig: updates => {
    const next = { ...get().config, ...updates }
    set({ config: next, configDirty: true })
  },

  // persistConfig: capture current in-memory config + live propagationStartFrame,
  // write to localStorage, and clear the dirty flag.
  persistConfig: () => {
    const { config, propagationStartFrame } = get()
    const toSave: AppConfig = { ...config, startFrame: propagationStartFrame }
    writeConfig(toSave)
    set({ config: toSave, configDirty: false })
  },

  // revertConfig: reload the last saved config from localStorage, restore
  // propagationStartFrame to the saved startFrame, and clear dirty flag.
  revertConfig: () => {
    const saved = loadConfig()
    set({ config: saved, propagationStartFrame: saved.startFrame, configDirty: false })
  },

  addToast: (message, type = 'info') => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`
    set(s => ({ toasts: [...s.toasts, { id, message, type }] }))
  },

  removeToast: id => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })),

  // Mark dirty so unsaved-changes guard fires if user tries to leave Settings.
  setPropagationStartFrame: f => set({ propagationStartFrame: f, configDirty: true }),

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

  resetVideoState: () => set({
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
    sessionInitialized: false,
  }),
}))

// Derived selectors
export const currentVideo = (state: AppState) =>
  state.project && state.currentVideoId
    ? state.project.videos[state.currentVideoId] ?? null
    : null
