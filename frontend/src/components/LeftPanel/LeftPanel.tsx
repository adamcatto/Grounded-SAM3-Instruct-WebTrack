import React, { useEffect, useRef, useState } from 'react'
import { Plus, RotateCcw, ChevronRight, ChevronDown, Loader, Download, X, Pause, Play, SkipBack, SkipForward, Trash2 } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import {
  addObject, initSession, startPropagationSSE, startExportSSE, getProject, resetVideo,
  clearFrameMasks, clearMasksBulk, getPropagationStatus, pausePropagation, updateVideoMeta,
  resumeFromFrame, getAnchorFrames, commitAnchorFrame, swapObjectMasks,
  type ClearMasksMode,
} from '../../api/client'
import { getObjectColor } from '../../utils/colors'
import { clearMaskCache } from '../../utils/maskUtils'
import ObjectCard from './ObjectCard'
import StepIndicator from './StepIndicator'
import type { PropagationEvent } from '../../types'

export default function LeftPanel() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project, currentVideoId, currentFrame,
    currentObjectId, setCurrentObject,
    propagationStatus, setPropagationStatus, setPropagationProgress,
    propagationPausedAtFrame, setPropagationPausedAtFrame,
    sessionInitialized, setSessionInitialized,
    propagationStartFrame, setPropagationStartFrame, setCurrentFrame,
    resetVideoState, updateVideo,
    setProject, setSavedMask, savedMaskCache, clearSavedMaskCache,
    addToast,
    config,
    // Anchor phase
    anchorPhase, setAnchorPhase,
    anchorFrames, setAnchorFrames,
    currentAnchorIndex, setCurrentAnchorIndex,
    annotatedAnchorIndices, addAnnotatedAnchor,
    resetAnchorState,
  } = store

  const [addingObject, setAddingObject] = useState(false)
  const [newObjName, setNewObjName] = useState('')
  const [initializingSession, setInitializingSession] = useState(false)
  const [trackingError, setTrackingError] = useState('')
  const [trackingRetryMsg, setTrackingRetryMsg] = useState('')
  const [extractingPhase, setExtractingPhase] = useState(false)
  const [totalBatchesRef] = useState({ current: 1 })
  const [extractedFrameCount, setExtractedFrameCount] = useState(0)
  const [totalFramesToProcess, setTotalFramesToProcess] = useState(0)
  const [actualStartFrame, setActualStartFrame] = useState(0)
  const [trackFrame, setTrackFrame] = useState(0)
  const activeEsRef = useRef<EventSource | null>(null)

  /** Anchor index → wall ms when user landed on that anchor to label it (for duration analytics). */
  const anchorEnteredMsRef = useRef<Record<number, number>>({})

  // Clear masks modal state
  const [showClearMasksModal, setShowClearMasksModal] = useState(false)
  const [clearRangeFrom, setClearRangeFrom] = useState('')
  const [clearRangeTo, setClearRangeTo] = useState('')

  // Paused resume modal
  const [showResumeModal, setShowResumeModal] = useState(false)
  const [resumeModalInput, setResumeModalInput] = useState('')

  // Track range modal
  const [showTrackRangeModal, setShowTrackRangeModal] = useState(false)
  const [trackRangeStart, setTrackRangeStart] = useState('')
  const [trackRangeEnd, setTrackRangeEnd] = useState('')

  // Tracking method modal (shown when all anchors are labeled)
  const [showTrackMethodModal, setShowTrackMethodModal] = useState(false)
  const [trackMethodChoice, setTrackMethodChoice] = useState<'sequential' | 'all_anchors'>('sequential')

  // Next-video prompt (shown when all anchors labeled and project has more videos)
  const [showNextVideoModal, setShowNextVideoModal] = useState(false)

  // Swap masks modal
  const [showSwapModal, setShowSwapModal] = useState(false)
  const [swapObjA, setSwapObjA] = useState('')
  const [swapObjB, setSwapObjB] = useState('')
  const [swapFromFrame, setSwapFromFrame] = useState('')
  const [swapToFrame, setSwapToFrame] = useState('')
  const [swapping, setSwapping] = useState(false)

  type ExportStatus = 'idle' | 'running' | 'done' | 'error'
  const [exportStatus, setExportStatus] = useState<ExportStatus>('idle')
  const [exportProgress, setExportProgress] = useState(0)
  const [exportPath, setExportPath] = useState<string | null>(null)
  const [exportError, setExportError] = useState('')

  const objects = video ? Object.values(video.objects) : []
  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  // Next video in project order (if any)
  const nextVideoId = (() => {
    if (!project || !currentVideoId) return null
    const ids = Object.keys(project.videos)
    const idx = ids.indexOf(currentVideoId)
    return idx >= 0 && idx + 1 < ids.length ? ids[idx + 1] : null
  })()

  const isTracking = propagationStatus === 'running'
  const isPaused = propagationStatus === 'paused'
  const isDone = propagationStatus === 'done'
  const hasObjects = objects.length > 0
  const allAnchorsLabeled = anchorFrames.length > 0 && annotatedAnchorIndices.length >= anchorFrames.length

  // ── Restore progress on video load ──────────────────────────────────────────

  useEffect(() => {
    if (!pid || !vid) return
    getPropagationStatus(pid, vid).then(status => {
      const startF = status.start_frame || 0
      const total = status.total_frames || 1
      const lastFrame = status.last_frame >= 0 ? status.last_frame : startF
      setTrackFrame(lastFrame + 1)
      setExtractedFrameCount(lastFrame + 1)
      setActualStartFrame(startF)
      totalBatchesRef.current = Math.max(1, Math.ceil((total - startF) / 1000))

      if (status.propagation_complete) {
        if (useStore.getState().propagationStatus !== 'done') {
          setPropagationStatus('done')
          setPropagationProgress(1, lastFrame)
        }
      } else if (status.is_paused) {
        setPropagationStatus('paused')
        setPropagationPausedAtFrame(status.paused_at_frame ?? lastFrame)
      } else if (status.is_running) {
        setPropagationStatus('running')
        setTrackingError('')
        setTrackingRetryMsg('')
        setExtractingPhase(false)
        _connectSSE(0)
      }
    }).catch(() => { /* ignore — backend may not be ready yet */ })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid, vid])

  useEffect(() => {
    anchorEnteredMsRef.current = {}
  }, [vid])

  // ── Add Object ──────────────────────────────────────────────────────────────

  async function handleAddObject() {
    if (!newObjName.trim()) return
    const idx = objects.length
    const color = getObjectColor(idx)
    const obj = await addObject(pid, vid, newObjName.trim(), color)
    updateVideo({ objects: { ...(video?.objects ?? {}), [obj.id]: obj } })
    setCurrentObject(obj.id)
    setNewObjName('')
    setAddingObject(false)
  }

  // ── Session Init ─────────────────────────────────────────────────────────────

  async function handleInitSession() {
    setInitializingSession(true)
    try {
      await initSession(pid, vid)
      setSessionInitialized(true)
    } catch (e: unknown) {
      console.error('Session init failed:', e)
      throw e
    } finally {
      setInitializingSession(false)
    }
  }

  // ── Anchor Annotation Phase ──────────────────────────────────────────────────

  async function handleStartAnchorAnnotation() {
    if (!hasObjects) return
    setTrackingError('')

    // Fetch anchor frames
    try {
      const { anchor_frames } = await getAnchorFrames(pid, vid)
      setAnchorFrames(anchor_frames)
      setCurrentAnchorIndex(0)
      setAnchorPhase(true)
      anchorEnteredMsRef.current = {}
      anchorEnteredMsRef.current[0] = Date.now()
      // Navigate to first anchor frame
      setCurrentFrame(anchor_frames[0] ?? propagationStartFrame)
      addToast(
        "You're being timed — per-anchor and whole-video durations are saved in your project.",
        'success',
      )
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to get anchor frames'
      setTrackingError(msg)
    }
  }

  async function handleCommitAnchor() {
    if (!anchorPhase || anchorFrames.length === 0) return
    const frameIdx = anchorFrames[currentAnchorIndex]
    const ai = currentAnchorIndex
    const finished_ms = Date.now()
    let entered_ms = anchorEnteredMsRef.current[ai]
    if (entered_ms === undefined) entered_ms = finished_ms
    const labelingTiming = { entered_ms, finished_ms }
    const frameSecStr = Math.max(0, (finished_ms - entered_ms) / 1000).toFixed(2)

    let wholeVideoWallMs: number | undefined
    try {
      await commitAnchorFrame(pid, vid, frameIdx, ai, labelingTiming)
      addAnnotatedAnchor(ai)
      try {
        const fresh = await getProject(pid)
        setProject(fresh)
        const w = fresh.videos[vid]?.anchor_labeling_timing?.video?.whole_video_labeling_wall_ms
        if (typeof w === 'number') wholeVideoWallMs = w
      } catch {
        /* keep local state — timing is persisted server-side anyway */
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to commit anchor frame'
      setTrackingError(msg)
      return
    }

    const nextIndex = ai + 1
    const isLastAnchor = nextIndex >= anchorFrames.length

    if (isLastAnchor && wholeVideoWallMs != null) {
      const totalSecStr = (wholeVideoWallMs / 1000).toFixed(2)
      addToast(
        `This anchor frame: ${frameSecStr}s\nTotal for video: ${totalSecStr}s`,
        'success',
      )
    } else {
      addToast(`This anchor frame: ${frameSecStr}s`, 'success')
    }

    if (isLastAnchor) {
      setAnchorPhase(false)
      setCurrentAnchorIndex(anchorFrames.length)
      if (nextVideoId) {
        setShowNextVideoModal(true)
      }
      return
    }

    anchorEnteredMsRef.current[nextIndex] = Date.now()
    setCurrentAnchorIndex(nextIndex)
    setCurrentFrame(anchorFrames[nextIndex])
  }

  function handleStartTracking() {
    // When all anchors are labeled, prompt the user to choose the tracking method.
    // Otherwise start immediately with the default (sequential) approach.
    if (!hasObjects) return
    if (allAnchorsLabeled) {
      setTrackMethodChoice(config.useAllAnchors ? 'all_anchors' : 'sequential')
      setShowTrackMethodModal(true)
    } else {
      _doStartTracking(false)
    }
  }

  function _doStartTracking(useAllAnchors: boolean) {
    setShowTrackMethodModal(false)
    if (!hasObjects) return
    setTrackingError('')
    setTrackingRetryMsg('')
    resetAnchorState()
    setPropagationStatus('running')
    setExtractingPhase(false)
    setTrackFrame(propagationStartFrame)
    setExtractedFrameCount(propagationStartFrame)
    setTotalFramesToProcess(0)
    setActualStartFrame(propagationStartFrame)
    totalBatchesRef.current = 1
    _connectSSE(0, undefined, -1, useAllAnchors)
  }

  function handleStartTrackRange() {
    const from = parseInt(trackRangeStart)
    const to = parseInt(trackRangeEnd)
    const maxFrame = (video?.num_frames ?? 1) - 1
    if (isNaN(from) || isNaN(to) || from > to || from < 0 || to > maxFrame) return
    setShowTrackRangeModal(false)
    setTrackingError('')
    setTrackingRetryMsg('')
    resetAnchorState()
    setPropagationStatus('running')
    setExtractingPhase(false)
    setTrackFrame(from)
    setExtractedFrameCount(from)
    setTotalFramesToProcess(to - from + 1)
    setActualStartFrame(from)
    totalBatchesRef.current = Math.max(1, Math.ceil((to - from + 1) / 1000))
    _connectSSE(0, from, to)
  }

  async function handleSwapMasks() {
    if (!swapObjA || !swapObjB || swapObjA === swapObjB) return
    setSwapping(true)
    try {
      const fromF = swapFromFrame !== '' ? parseInt(swapFromFrame) : undefined
      const toF = swapToFrame !== '' ? parseInt(swapToFrame) : undefined
      const result = await swapObjectMasks(pid, vid, swapObjA, swapObjB, fromF, toF)
      setShowSwapModal(false)
      clearSavedMaskCache()
      clearMaskCache()
      addToast(`Swapped masks for ${result.frames_swapped} frame(s)`, 'success')
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Swap failed'
      addToast(msg, 'error')
    } finally {
      setSwapping(false)
    }
  }

  // ── SSE Propagation ──────────────────────────────────────────────────────────

  function _connectSSE(retryCount: number, explicitStart?: number, endFrame = -1, useAllAnchors = false) {
    const MAX_RETRIES = 3
    activeEsRef.current?.close()

    if (!pid || !vid) {
      setTrackingError('Internal error: project/video ID missing')
      setPropagationStatus('error')
      return
    }
    const effectiveStart = explicitStart ?? propagationStartFrame
    const es = startPropagationSSE(pid, vid, effectiveStart, -1, endFrame, useAllAnchors)
    activeEsRef.current = es

    es.addEventListener('init', (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      setTotalFramesToProcess(data.frames_to_process ?? 0)
      setActualStartFrame(data.actual_start ?? 0)
      totalBatchesRef.current = data.total_batches ?? 1
      setExtractedFrameCount(data.actual_start ?? 0)
    })
    es.addEventListener('catch_up', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      const total: number = data.total_frames ?? 1
      const last: number = data.last_frame ?? -1
      const startF: number = data.start_frame ?? propagationStartFrame
      setTrackFrame(last + 1)
      setExtractedFrameCount(last + 1)
      setTotalFramesToProcess(total - startF)
      setActualStartFrame(startF)
      totalBatchesRef.current = Math.max(1, Math.ceil(total / 1000))
    })
    es.addEventListener('batch_start', (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      totalBatchesRef.current = data.total_batches ?? 1
      if (data.status === 'extracting') {
        setExtractingPhase(true)
        setExtractedFrameCount(data.batch_start ?? 0)
      } else if (data.status === 'initializing_session') {
        setExtractedFrameCount(data.batch_end ?? 0)
        setExtractingPhase(false)
      }
    })
    es.addEventListener('extract_progress', (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      setExtractedFrameCount((data.batch_start ?? 0) + (data.extracted ?? 0))
    })
    es.addEventListener('progress', (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      setExtractingPhase(false)
      setTrackFrame(data.frame + 1)
      setPropagationProgress(data.progress, data.frame)
    })
    es.addEventListener('done', async (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      setExtractingPhase(false)
      setPropagationStatus('done')
      setPropagationProgress(1, data.frame ?? 0)
      setTrackingRetryMsg('')
      es.close()
      activeEsRef.current = null
      clearSavedMaskCache()
      clearMaskCache()
      const fresh = await getProject(pid)
      setProject(fresh)
    })
    es.addEventListener('error', (e: Event) => {
      try {
        const data = JSON.parse((e as MessageEvent).data ?? '{}')
        setTrackingError(data.error ?? 'Propagation failed')
      } catch {
        setTrackingError('Propagation error')
      }
      setExtractingPhase(false)
      setPropagationStatus('error')
      es.close()
      activeEsRef.current = null
    })
    es.onerror = () => {
      const status = useStore.getState().propagationStatus
      if (status === 'done') return
      es.close()
      activeEsRef.current = null
      setExtractingPhase(false)
      if (retryCount < MAX_RETRIES) {
        const attempt = retryCount + 1
        setTrackingRetryMsg(`Connection lost. Retrying (${attempt}/${MAX_RETRIES})...`)
        setPropagationStatus('running')
        setTimeout(() => _connectSSE(attempt, explicitStart, endFrame, useAllAnchors), 3000)
      } else {
        setPropagationStatus('error')
        setTrackingError(`Connection lost after ${MAX_RETRIES} retries. Check that backend is running.`)
        setTrackingRetryMsg('')
      }
    }
  }

  function _connectSSEResume(resumeFrom: number, retryCount: number) {
    const MAX_RETRIES = 3
    activeEsRef.current?.close()
    const es = startPropagationSSE(pid, vid, propagationStartFrame, resumeFrom)
    activeEsRef.current = es

    es.addEventListener('init', (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      setTotalFramesToProcess(data.frames_to_process ?? 0)
      setActualStartFrame(data.actual_start ?? 0)
      totalBatchesRef.current = data.total_batches ?? 1
      setExtractedFrameCount(data.actual_start ?? 0)
    })
    es.addEventListener('batch_start', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      totalBatchesRef.current = data.total_batches ?? 1
      if (data.status === 'extracting') {
        setExtractingPhase(true)
        setExtractedFrameCount(data.batch_start ?? 0)
      } else if (data.status === 'initializing_session') {
        setExtractedFrameCount(data.batch_end ?? 0)
        setExtractingPhase(false)
      }
    })
    es.addEventListener('extract_progress', (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      setExtractedFrameCount((data.batch_start ?? 0) + (data.extracted ?? 0))
    })
    es.addEventListener('progress', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setExtractingPhase(false)
      setTrackFrame(data.frame + 1)
      setPropagationProgress(data.progress, data.frame)
    })
    es.addEventListener('done', async (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setExtractingPhase(false)
      setPropagationStatus('done')
      setPropagationProgress(1, data.frame ?? 0)
      setTrackingRetryMsg('')
      es.close()
      activeEsRef.current = null
      clearSavedMaskCache()
      clearMaskCache()
      const fresh = await getProject(pid)
      setProject(fresh)
    })
    es.addEventListener('error', (e: Event) => {
      try {
        const data = JSON.parse((e as MessageEvent).data ?? '{}')
        setTrackingError(data.error ?? 'Propagation failed')
      } catch {
        setTrackingError('Propagation error')
      }
      setExtractingPhase(false)
      setPropagationStatus('error')
      es.close()
      activeEsRef.current = null
    })
    es.onerror = () => {
      const status = useStore.getState().propagationStatus
      if (status === 'done') return
      es.close()
      activeEsRef.current = null
      setExtractingPhase(false)
      if (retryCount < MAX_RETRIES) {
        const attempt = retryCount + 1
        setTrackingRetryMsg(`Connection lost. Retrying (${attempt}/${MAX_RETRIES})...`)
        setPropagationStatus('running')
        setTimeout(() => _connectSSEResume(resumeFrom, attempt), 3000)
      } else {
        setPropagationStatus('error')
        setTrackingError(`Connection lost after ${MAX_RETRIES} retries`)
        setTrackingRetryMsg('')
      }
    }
  }

  async function handlePause() {
    activeEsRef.current?.close()
    activeEsRef.current = null
    try {
      const result = await pausePropagation(pid, vid)
      setPropagationStatus('paused')
      setPropagationPausedAtFrame(result.paused_at_frame)
    } catch (e: unknown) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if ((e as any)?.response?.status === 400) {
        setPropagationStatus('idle')
        return
      }
      const msg = e instanceof Error ? e.message : 'Pause failed'
      addToast(msg, 'error')
    }
  }

  function handleResume() {
    setTrackingError('')
    setTrackingRetryMsg('')
    setPropagationStatus('running')
    setExtractingPhase(false)
    const resumeFrom = propagationPausedAtFrame
    setTrackFrame(resumeFrom + 1)
    setExtractedFrameCount(resumeFrom + 1)
    setTotalFramesToProcess(0)
    setActualStartFrame(resumeFrom + 1)
    _connectSSEResume(resumeFrom, 0)
  }

  function handleRestartFromStart() {
    setTrackingError('')
    setTrackingRetryMsg('')
    setPropagationStatus('running')
    setExtractingPhase(false)
    setTrackFrame(propagationStartFrame)
    setExtractedFrameCount(propagationStartFrame)
    setTotalFramesToProcess(0)
    setActualStartFrame(propagationStartFrame)
    totalBatchesRef.current = 1
    _connectSSE(0)
  }

  async function handleResumeFromFrame() {
    const f = parseInt(resumeModalInput)
    if (isNaN(f) || f < 0) return
    setShowResumeModal(false)
    setTrackingError('')
    setTrackingRetryMsg('')
    try {
      await resumeFromFrame(pid, vid, f, true)
    } catch {
      setTrackingError('Failed to clear masks for the selected frame')
      return
    }
    setPropagationStatus('running')
    setExtractingPhase(false)
    setTrackFrame(f)
    setExtractedFrameCount(f)
    setTotalFramesToProcess(0)
    setActualStartFrame(f)
    _connectSSEResume(f - 1, 0)
  }

  // ── Start frame ─────────────────────────────────────────────────────────────

  async function _saveStartFrame(f: number) {
    if (!pid || !vid) return
    try {
      await updateVideoMeta(pid, vid, { start_frame: f })
      updateVideo({ start_frame: f })
    } catch { /* non-critical */ }
  }

  function _commitStartFrame() {
    const max = video?.num_frames ? video.num_frames - 1 : 0
    const clamped = Math.max(0, Math.min(max, propagationStartFrame))
    setPropagationStartFrame(clamped)
    setCurrentFrame(clamped)
    _saveStartFrame(clamped)
  }

  // ── Clear Masks ──────────────────────────────────────────────────────────────

  function handleClearFrameMasks() {
    setShowClearMasksModal(true)
  }

  async function handleConfirmClearMasks(mode: ClearMasksMode | 'this_frame') {
    setShowClearMasksModal(false)
    try {
      if (mode === 'this_frame') {
        await clearFrameMasks(pid, vid, currentFrame)
        setSavedMask(currentFrame, {})
      } else if (mode === 'from_frame') {
        await clearMasksBulk(pid, vid, 'from_frame', currentFrame)
      } else if (mode === 'range') {
        const from = parseInt(clearRangeFrom)
        const to = parseInt(clearRangeTo)
        if (!isNaN(from) && !isNaN(to)) {
          await clearMasksBulk(pid, vid, 'range', from, to)
        }
      } else if (mode === 'all') {
        await clearMasksBulk(pid, vid, 'all')
      }
      clearMaskCache()
      const fresh = await getProject(pid)
      setProject(fresh)
      addToast('Masks cleared', 'success')
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to clear masks'
      addToast(msg, 'error')
    }
  }

  // ── Export ───────────────────────────────────────────────────────────────────

  function handleExport() {
    if (!pid || !vid) return
    setExportStatus('running')
    setExportProgress(0)
    setExportPath(null)
    setExportError('')
    const es = startExportSSE(pid, vid)
    es.addEventListener('progress', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setExportProgress(data.progress ?? 0)
    })
    es.addEventListener('done', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setExportStatus('done')
      setExportProgress(1)
      setExportPath(data.path ?? null)
      es.close()
    })
    es.addEventListener('error', (e: Event) => {
      try {
        const data = JSON.parse((e as MessageEvent).data ?? '{}')
        setExportError(data.error ?? 'Export failed')
      } catch {
        setExportError('Export failed')
      }
      setExportStatus('error')
      es.close()
    })
    es.onerror = () => {
      setExportError('Connection lost during export')
      setExportStatus('error')
      es.close()
    }
  }

  // ── Start Over ───────────────────────────────────────────────────────────────

  async function handleStartOver() {
    if (!pid || !vid) return
    activeEsRef.current?.close()
    activeEsRef.current = null
    clearMaskCache()
    await resetVideo(pid, vid)
    const fresh = await getProject(pid)
    setProject(fresh)
    resetVideoState()
  }

  if (!video) return null

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Step indicator */}
      <StepIndicator
        step={isDone ? 2 : 1}
        total={2}
        title={isDone ? 'Review tracked objects' : 'Select objects'}
        subtitle={
          isDone
            ? 'Review your selected objects across the video. Continue to edit if needed.'
            : 'Add objects and draw points on the frame, then annotate anchor frames to track.'
        }
      />

      {/* Anchor annotation phase header */}
      {anchorPhase && (
        <div className="mx-3 mt-2 mb-1 rounded-lg border border-blue-700/40 bg-blue-500/10 px-3 py-2 flex-shrink-0">
          <p className="text-xs text-blue-300 font-medium">
            Anchor Frame {currentAnchorIndex + 1} / {anchorFrames.length}
            <span className="text-blue-400/60 ml-1.5 font-normal">(frame #{anchorFrames[currentAnchorIndex]})</span>
          </p>
          <p className="text-xs text-[#666] mt-0.5">
            Annotate objects on this frame, then click "Done, next frame".
          </p>
        </div>
      )}

      {/* All anchors labeled banner */}
      {!anchorPhase && allAnchorsLabeled && propagationStatus === 'idle' && (
        <div className="mx-3 mt-2 mb-1 rounded-lg border border-emerald-700/40 bg-emerald-500/10 px-3 py-2 flex-shrink-0">
          <p className="text-xs text-emerald-300 font-medium">All anchor frames labeled</p>
          <p className="text-xs text-[#666] mt-0.5">Click "Start Tracking" to run bidirectional propagation.</p>
        </div>
      )}

      {/* Objects list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {objects.map(obj => (
          <ObjectCard
            key={obj.id}
            objId={obj.id}
            name={obj.name}
            color={obj.color}
            isActive={currentObjectId === obj.id}
            onSelect={() => {
              setCurrentObject(currentObjectId === obj.id ? null : obj.id)
            }}
            description={obj.description}
          />
        ))}

        {/* Add new object */}
        {!isTracking && !anchorPhase && (
          addingObject ? (
            <div className="rounded-xl border border-[#333] bg-[#1a1a1a] p-3">
              <p className="text-xs text-[#666] mb-2">Name this object</p>
              <input
                type="text"
                value={newObjName}
                onChange={e => setNewObjName(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') handleAddObject()
                  if (e.key === 'Escape') { setAddingObject(false); setNewObjName('') }
                }}
                placeholder="e.g. Person, Car, Ball..."
                className="w-full mb-2"
                autoFocus
              />
              <div className="flex gap-2">
                <button
                  onClick={() => { setAddingObject(false); setNewObjName('') }}
                  className="btn btn-secondary flex-1 text-xs py-1"
                >
                  Cancel
                </button>
                <button
                  onClick={handleAddObject}
                  disabled={!newObjName.trim()}
                  className="btn btn-primary flex-1 text-xs py-1"
                >
                  Add
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setAddingObject(true)}
              className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-dashed border-[#333] text-[#666] hover:border-[#555] hover:text-[#aaa] transition-colors text-sm"
            >
              <div className="w-8 h-8 rounded-lg border border-[#333] flex items-center justify-center">
                <Plus size={14} />
              </div>
              Add another object
            </button>
          )
        )}
      </div>

      {/* SAM loading indicator */}
      {initializingSession && (
        <div className="mx-3 mb-2 flex items-center gap-2 text-xs text-[#888]">
          <Loader size={12} className="animate-spin" />
          Loading SAM model...
        </div>
      )}

      {/* Propagation progress */}
      {isTracking && (() => {
        const totalFrames = totalFramesToProcess > 0
          ? totalFramesToProcess
          : Math.max(1, (video?.num_frames ?? 1) - propagationStartFrame)
        const extractDone = Math.min(totalFrames, Math.max(0, extractedFrameCount - actualStartFrame))
        const extractPct = totalFrames > 0 ? Math.min(1, extractDone / totalFrames) : 0
        const trackDone = Math.min(totalFrames, Math.max(0, trackFrame - actualStartFrame))
        const trackPct = totalFrames > 0 ? Math.min(1, trackDone / totalFrames) : 0
        return (
          <div className="mx-3 mb-2 space-y-2">
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-[#888]">
                <span>Extracting frames</span>
                <span className={!extractingPhase && extractDone >= totalFrames ? 'text-emerald-500' : ''}>
                  {extractDone}/{totalFrames}
                </span>
              </div>
              <div className="h-1.5 bg-[#2a2a2a] rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-150 ${extractingPhase ? 'bg-amber-500' : 'bg-emerald-600'}`}
                  style={{ width: `${extractPct * 100}%` }}
                />
              </div>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-[#888]">
                <span>Tracking objects</span>
                <span>{trackDone}/{totalFrames}</span>
              </div>
              <div className="h-1.5 bg-[#2a2a2a] rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-150"
                  style={{ width: `${trackPct * 100}%` }}
                />
              </div>
            </div>
            {trackingRetryMsg && (
              <p className="text-xs text-amber-400">{trackingRetryMsg}</p>
            )}
          </div>
        )
      })()}

      {/* Error */}
      {trackingError && (
        <p className="mx-3 mb-2 text-xs text-red-400 bg-red-400/10 rounded-lg px-3 py-2">
          {trackingError}
        </p>
      )}

      {/* Export progress */}
      {exportStatus === 'running' && (
        <div className="mx-3 mb-2 space-y-1">
          <div className="flex justify-between text-xs text-[#888]">
            <span>Exporting video...</span>
            <span>{Math.round(exportProgress * 100)}%</span>
          </div>
          <div className="h-1.5 bg-[#2a2a2a] rounded-full overflow-hidden">
            <div
              className="h-full bg-emerald-500 transition-all duration-300"
              style={{ width: `${exportProgress * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Export error */}
      {exportStatus === 'error' && exportError && (
        <p className="mx-3 mb-2 text-xs text-red-400 bg-red-400/10 rounded-lg px-3 py-2">
          {exportError}
        </p>
      )}

      {/* Export done */}
      {exportStatus === 'done' && exportPath && (
        <div className="mx-3 mb-2 rounded-lg bg-emerald-400/10 border border-emerald-400/20 px-3 py-2">
          <div className="flex items-start gap-2">
            <div className="flex-1 min-w-0">
              <p className="text-xs text-emerald-400 font-medium mb-1">Export saved</p>
              <p className="text-xs text-[#888] break-all font-mono leading-relaxed">{exportPath}</p>
            </div>
            <button
              onClick={() => setExportStatus('idle')}
              className="text-[#555] hover:text-[#aaa] flex-shrink-0 mt-0.5"
            >
              <X size={12} />
            </button>
          </div>
        </div>
      )}

      {/* Start frame */}
      <div className="px-3 pb-2 flex items-center gap-2 flex-shrink-0">
        <label className="text-xs text-[#666] whitespace-nowrap">Start frame</label>
        <input
          type="number"
          min={0}
          max={video.num_frames - 1}
          value={propagationStartFrame}
          onChange={e => setPropagationStartFrame(Math.max(0, parseInt(e.target.value) || 0))}
          onKeyDown={e => { if (e.key === 'Enter') _commitStartFrame() }}
          onBlur={_commitStartFrame}
          disabled={isTracking || anchorPhase}
          className="w-full text-xs py-1 px-2 rounded bg-[#1a1a1a] border border-[#333] text-[#ccc] disabled:opacity-40"
        />
      </div>

      {/* Paused state actions */}
      {isPaused && (
        <div className="mx-2 mb-2 rounded-lg border border-amber-800/40 bg-amber-400/5 p-2.5 flex-shrink-0">
          <p className="text-xs text-amber-400 mb-2">
            Paused at frame {propagationPausedAtFrame}.
          </p>
          <div className="flex gap-1.5">
            <button
              onClick={handleResume}
              disabled={!hasObjects}
              className="btn btn-primary flex-1 flex items-center justify-center gap-1 text-xs py-1.5 disabled:opacity-40"
            >
              <Play size={10} />
              Resume
            </button>
            <button
              onClick={() => { setResumeModalInput(String(propagationPausedAtFrame)); setShowResumeModal(true) }}
              disabled={!hasObjects}
              className="btn btn-ghost flex-1 flex items-center justify-center gap-1 text-xs py-1.5 disabled:opacity-40"
            >
              <SkipForward size={10} />
              From frame
            </button>
            <button
              onClick={handleRestartFromStart}
              disabled={!hasObjects}
              className="btn btn-ghost flex-1 flex items-center justify-center gap-1 text-xs py-1.5 disabled:opacity-40"
            >
              <SkipBack size={10} />
              Restart
            </button>
          </div>
        </div>
      )}

      {/* Bottom buttons */}
      <div
        className="flex items-stretch gap-1 px-1.5 py-1 border-t border-[#2a2a2a] flex-shrink-0 overflow-hidden"
        style={{ containerType: 'inline-size' } as React.CSSProperties}
      >
        <button
          onClick={handleStartOver}
          disabled={isTracking}
          className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2"
          style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
        >
          <RotateCcw size={10} />
          <span>Start</span>
          <span>over</span>
        </button>

        {(video.propagated_frames?.length ?? 0) > 0 && (
          <button
            onClick={handleExport}
            disabled={exportStatus === 'running' || isTracking}
            className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40"
            style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
            title="Export annotated MP4"
          >
            {exportStatus === 'running'
              ? <Loader size={10} className="animate-spin" />
              : <Download size={10} />
            }
            <span>Export</span>
          </button>
        )}

        {savedMaskCache[currentFrame] && Object.keys(savedMaskCache[currentFrame]).length > 0 && (
          <button
            onClick={handleClearFrameMasks}
            disabled={isTracking}
            className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40 text-red-400 hover:text-red-300"
            style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
            title={`Clear saved masks for frame ${currentFrame}`}
          >
            <Trash2 size={10} />
            <span>Clear</span>
            <span>masks</span>
          </button>
        )}

        {objects.length >= 2 && (video.propagated_frames?.length ?? 0) > 0 && !isTracking && (
          <button
            onClick={() => {
              setSwapObjA(objects[0]?.id ?? '')
              setSwapObjB(objects[1]?.id ?? '')
              setSwapFromFrame('')
              setSwapToFrame('')
              setShowSwapModal(true)
            }}
            className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2"
            style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
            title="Swap two objects' masks"
          >
            <RotateCcw size={10} />
            <span>Swap</span>
            <span>masks</span>
          </button>
        )}

        {/* Anchor annotation / track button */}
        {isTracking ? (
          <button
            onClick={handlePause}
            className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 text-amber-400 hover:text-amber-300"
            style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
          >
            <Pause size={10} />
            <span>Pause</span>
          </button>
        ) : anchorPhase ? (
          <button
            onClick={handleCommitAnchor}
            disabled={!hasObjects}
            className="btn btn-primary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40"
            style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
          >
            <ChevronDown size={10} />
            <span>Done,</span>
            <span>next</span>
          </button>
        ) : allAnchorsLabeled ? (
          <button
            onClick={handleStartTracking}
            disabled={!hasObjects}
            className="btn btn-primary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40"
            style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
          >
            <ChevronRight size={10} />
            <span>Start</span>
            <span>tracking</span>
          </button>
        ) : (
          <>
            <button
              onClick={handleStartAnchorAnnotation}
              disabled={!hasObjects || initializingSession || isPaused}
              className="btn btn-secondary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40"
              style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
            >
              {initializingSession
                ? <Loader size={10} className="animate-spin" />
                : <ChevronRight size={10} />
              }
              <span>Annotate</span>
              <span>anchors</span>
            </button>
            <button
              onClick={() => {
                setTrackRangeStart(String(propagationStartFrame))
                setTrackRangeEnd(String((video?.num_frames ?? 1) - 1))
                setShowTrackRangeModal(true)
              }}
              disabled={!hasObjects || initializingSession || isPaused}
              className="btn btn-primary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40"
              style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
            >
              <ChevronRight size={10} />
              <span>Track</span>
              <span>frames</span>
            </button>
          </>
        )}
      </div>

      {/* Resume from frame modal */}
      {showResumeModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-80 shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-1">Resume from frame</h3>
            <p className="text-xs text-[#666] mb-4 leading-relaxed">
              Masks from this frame onwards will be cleared and tracking will restart from here.
            </p>
            <input
              type="number"
              min={0}
              max={video ? video.num_frames - 1 : 999999}
              value={resumeModalInput}
              onChange={e => setResumeModalInput(e.target.value)}
              className="w-full mb-4 text-sm py-1.5 px-2.5 rounded bg-[#111] border border-[#333] text-[#ccc]"
              autoFocus
              onKeyDown={e => { if (e.key === 'Enter') handleResumeFromFrame() }}
            />
            <div className="flex gap-2">
              <button onClick={() => setShowResumeModal(false)} className="btn btn-ghost flex-1 py-2 text-xs">
                Cancel
              </button>
              <button onClick={handleResumeFromFrame} className="btn btn-primary flex-1 py-2 text-xs font-medium">
                Resume
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Track Range Modal */}
      {showTrackRangeModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-80 shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-1">Track frame range</h3>
            <p className="text-xs text-[#666] mb-4 leading-relaxed">
              Annotate frames in the range below, then track them. Only frames with point prompts will be used as seeds.
            </p>
            <div className="flex items-center gap-2 mb-4">
              <div className="flex-1">
                <label className="block text-[10px] text-[#666] mb-1">Start frame</label>
                <input
                  type="number"
                  min={0}
                  max={video ? video.num_frames - 1 : 0}
                  value={trackRangeStart}
                  onChange={e => setTrackRangeStart(e.target.value)}
                  className="w-full text-sm py-1.5 px-2.5 rounded bg-[#111] border border-[#333] text-[#ccc]"
                  autoFocus
                />
              </div>
              <span className="text-xs text-[#555] mt-4">–</span>
              <div className="flex-1">
                <label className="block text-[10px] text-[#666] mb-1">End frame</label>
                <input
                  type="number"
                  min={0}
                  max={video ? video.num_frames - 1 : 0}
                  value={trackRangeEnd}
                  onChange={e => setTrackRangeEnd(e.target.value)}
                  className="w-full text-sm py-1.5 px-2.5 rounded bg-[#111] border border-[#333] text-[#ccc]"
                  onKeyDown={e => { if (e.key === 'Enter') handleStartTrackRange() }}
                />
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => setShowTrackRangeModal(false)} className="btn btn-ghost flex-1 py-2 text-xs">
                Cancel
              </button>
              <button onClick={handleStartTrackRange} className="btn btn-primary flex-1 py-2 text-xs font-medium">
                Track
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Swap Masks Modal */}
      {showSwapModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-80 shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-1">Swap object masks</h3>
            <p className="text-xs text-[#666] mb-4 leading-relaxed">
              Swap the mask assignments between two objects across all (or a range of) frames.
            </p>
            <div className="space-y-3 mb-4">
              <div>
                <label className="block text-[10px] text-[#666] mb-1">Object A</label>
                <select
                  value={swapObjA}
                  onChange={e => setSwapObjA(e.target.value)}
                  className="w-full text-sm py-1.5 px-2.5 rounded bg-[#111] border border-[#333] text-[#ccc]"
                >
                  {objects.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-[#666] mb-1">Object B</label>
                <select
                  value={swapObjB}
                  onChange={e => setSwapObjB(e.target.value)}
                  className="w-full text-sm py-1.5 px-2.5 rounded bg-[#111] border border-[#333] text-[#ccc]"
                >
                  {objects.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-[#666] mb-1">Frame range (optional — leave blank for all frames)</label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={0}
                    max={video ? video.num_frames - 1 : 0}
                    value={swapFromFrame}
                    onChange={e => setSwapFromFrame(e.target.value)}
                    placeholder="From"
                    className="flex-1 text-xs py-1.5 px-2 rounded bg-[#111] border border-[#333] text-[#ccc]"
                  />
                  <span className="text-xs text-[#555]">–</span>
                  <input
                    type="number"
                    min={0}
                    max={video ? video.num_frames - 1 : 0}
                    value={swapToFrame}
                    onChange={e => setSwapToFrame(e.target.value)}
                    placeholder="To"
                    className="flex-1 text-xs py-1.5 px-2 rounded bg-[#111] border border-[#333] text-[#ccc]"
                  />
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => setShowSwapModal(false)} className="btn btn-ghost flex-1 py-2 text-xs" disabled={swapping}>
                Cancel
              </button>
              <button
                onClick={handleSwapMasks}
                disabled={swapping || !swapObjA || !swapObjB || swapObjA === swapObjB}
                className="btn btn-primary flex-1 py-2 text-xs font-medium disabled:opacity-40"
              >
                {swapping ? 'Swapping...' : 'Swap'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Clear Masks Modal */}
      {showClearMasksModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-84 shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-1">Clear masks</h3>
            <div className="flex flex-col gap-2 mt-3">
              <button
                onClick={() => handleConfirmClearMasks('this_frame')}
                className="btn btn-ghost w-full py-2 text-xs text-left px-3"
              >
                This frame only <span className="text-[#555]">(frame {currentFrame})</span>
              </button>
              <button
                onClick={() => handleConfirmClearMasks('from_frame')}
                className="btn btn-ghost w-full py-2 text-xs text-left px-3"
              >
                This frame and all future frames <span className="text-[#555]">(frame {currentFrame} →)</span>
              </button>
              <div className="rounded border border-[#2a2a2a] p-2.5 flex flex-col gap-2">
                <p className="text-xs text-[#888]">Range of frames</p>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={0}
                    max={video ? video.num_frames - 1 : 0}
                    value={clearRangeFrom}
                    onChange={e => setClearRangeFrom(e.target.value)}
                    className="flex-1 text-xs py-1 px-2 rounded bg-[#111] border border-[#333] text-[#ccc]"
                    placeholder="From"
                  />
                  <span className="text-xs text-[#555]">–</span>
                  <input
                    type="number"
                    min={0}
                    max={video ? video.num_frames - 1 : 0}
                    value={clearRangeTo}
                    onChange={e => setClearRangeTo(e.target.value)}
                    className="flex-1 text-xs py-1 px-2 rounded bg-[#111] border border-[#333] text-[#ccc]"
                    placeholder="To"
                  />
                  <button
                    onClick={() => handleConfirmClearMasks('range')}
                    className="btn btn-ghost text-xs py-1 px-2 flex-shrink-0"
                  >
                    Clear
                  </button>
                </div>
              </div>
              <button
                onClick={() => handleConfirmClearMasks('all')}
                className="btn btn-ghost w-full py-2 text-xs text-red-400 hover:text-red-300 text-left px-3"
              >
                All frames
              </button>
            </div>
            <button
              onClick={() => setShowClearMasksModal(false)}
              className="mt-3 btn btn-ghost w-full py-2 text-xs text-[#555] hover:text-[#aaa]"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Next-video prompt — shown when all anchors are labeled and another video exists */}
      {showNextVideoModal && nextVideoId && project && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-84 shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-1">All anchor frames labeled</h3>
            <p className="text-xs text-[#666] mb-5 leading-relaxed">
              Would you like to start tracking this video now, or move on to the next video
              (<span className="text-[#aaa]">{project.videos[nextVideoId]?.name}</span>) to label its anchor frames first?
            </p>
            <div className="flex flex-col gap-2">
              <button
                onClick={() => {
                  setShowNextVideoModal(false)
                  handleStartTracking()
                }}
                className="btn btn-primary w-full py-2 text-xs font-medium"
              >
                Start tracking this video
              </button>
              <button
                onClick={() => {
                  setShowNextVideoModal(false)
                  store.setCurrentVideo(nextVideoId)
                }}
                className="btn btn-secondary w-full py-2 text-xs"
              >
                Go to next video &rarr; {project.videos[nextVideoId]?.name}
              </button>
              <button
                onClick={() => setShowNextVideoModal(false)}
                className="btn btn-ghost w-full py-2 text-xs text-[#555] hover:text-[#aaa]"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tracking method modal */}
      {showTrackMethodModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-84 shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-1">Choose tracking method</h3>
            <p className="text-xs text-[#555] mb-4 leading-relaxed">
              All anchor frames have been labeled. Select how SAM seeds each propagation batch.
            </p>

            <div className="space-y-2 mb-5">
              {/* Option A: sequential */}
              <button
                onClick={() => setTrackMethodChoice('sequential')}
                className={`w-full text-left rounded-lg border px-3 py-2.5 transition-colors ${
                  trackMethodChoice === 'sequential'
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-[#333] bg-[#111] hover:border-[#444]'
                }`}
              >
                <p className="text-xs font-medium text-[#ddd] mb-0.5">Sequential batches</p>
                <p className="text-[10px] text-[#666] leading-relaxed">
                  Each 1 000-frame batch seeds from its own labeled frames and the previous
                  batch's last mask. Fast and memory-efficient.
                </p>
              </button>

              {/* Option B: all anchors */}
              <button
                onClick={() => setTrackMethodChoice('all_anchors')}
                className={`w-full text-left rounded-lg border px-3 py-2.5 transition-colors ${
                  trackMethodChoice === 'all_anchors'
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-[#333] bg-[#111] hover:border-[#444]'
                }`}
              >
                <p className="text-xs font-medium text-[#ddd] mb-0.5">All anchor frames as context</p>
                <p className="text-[10px] text-[#666] leading-relaxed">
                  Every labeled anchor frame is loaded into each batch's SAM session as a
                  global context seed. May improve accuracy when objects change appearance
                  across batches. Slightly slower per batch.
                </p>
              </button>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setShowTrackMethodModal(false)}
                className="btn btn-ghost flex-1 py-2 text-xs"
              >
                Cancel
              </button>
              <button
                onClick={() => _doStartTracking(trackMethodChoice === 'all_anchors')}
                className="btn btn-primary flex-1 py-2 text-xs font-medium"
              >
                Start Tracking
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
