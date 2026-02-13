import { create } from 'zustand'
import type { Project, VideoMeta, ObjectDef, MaskData } from '../types'

export type PointMode = 'add' | 'remove' | null
export type PropagationStatus = 'idle' | 'running' | 'done' | 'error'
export type AppStep = 'upload' | 'annotate' | 'review'

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

  // Playback
  isPlaying: boolean

  // Propagation
  propagationStatus: PropagationStatus
  propagationProgress: number
  propagationCurrentFrame: number
  propagationStartFrame: number

  // UI
  drawerOpen: boolean
  uploadModalOpen: boolean
  sessionInitialized: boolean

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
  setPlaying: (v: boolean) => void
  setPropagationStatus: (s: PropagationStatus) => void
  setPropagationProgress: (p: number, frame: number) => void
  setDrawerOpen: (v: boolean) => void
  setUploadModalOpen: (v: boolean) => void
  setSessionInitialized: (v: boolean) => void
  setPropagationStartFrame: (f: number) => void
  updateVideo: (updates: Partial<VideoMeta>) => void
  resetVideoState: () => void
}

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
  isPlaying: false,
  propagationStatus: 'idle',
  propagationProgress: 0,
  propagationCurrentFrame: 0,
  propagationStartFrame: 0,
  drawerOpen: false,
  uploadModalOpen: false,
  sessionInitialized: false,

  setProject: p => set({ project: p }),

  setCurrentVideo: vid => {
    const prev = get().currentVideoId
    if (prev !== vid) {
      set({
        currentVideoId: vid,
        currentFrame: 0,
        currentObjectId: null,
        pointMode: null,
        localAnnotations: {},
        currentFrameMasks: {},
        currentFrameMasksFrame: null,
        savedMaskCache: {},
        isPlaying: false,
        propagationStatus: 'idle',
        propagationProgress: 0,
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

  setPlaying: v => set({ isPlaying: v }),

  setPropagationStatus: s => set({ propagationStatus: s }),

  setPropagationProgress: (p, frame) =>
    set({ propagationProgress: p, propagationCurrentFrame: frame }),

  setDrawerOpen: v => set({ drawerOpen: v }),
  setUploadModalOpen: v => set({ uploadModalOpen: v }),
  setSessionInitialized: v => set({ sessionInitialized: v }),
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

  resetVideoState: () => set({
    currentFrame: 0,
    currentObjectId: null,
    pointMode: null,
    localAnnotations: {},
    currentFrameMasks: {},
    currentFrameMasksFrame: null,
    savedMaskCache: {},
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
