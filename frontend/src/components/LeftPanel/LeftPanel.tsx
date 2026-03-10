import React, { useEffect, useRef, useState } from 'react'
import { Plus, RotateCcw, ChevronRight, Loader, Download, X, Zap, Save, Pause, Play, SkipBack, SkipForward, ArrowRightLeft, Trash2 } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { addObject, initSession, startPropagationSSE, startExportSSE, getProject, resetVideo, predictFrame, saveFrameInference, clearFrameMasks, clearMasksBulk, getPropagationStatus, pausePropagation, updateVideoMeta, swapMasks, startSwapAllSSE, getUncertainty, getCorrections, resumeFromFrame, type SwapMode, type ClearMasksMode } from '../../api/client'
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
    setProject, setSavedMask, savedMaskCache,
    pendingInferenceFrame, setPendingInferenceFrame,
    config, addToast,
    setUncertaintyData, setCorrections, setViewerTab,
  } = store

  const [addingObject, setAddingObject] = useState(false)
  const [newObjName, setNewObjName] = useState('')
  const [initializingSession, setInitializingSession] = useState(false)
  const [trackingError, setTrackingError] = useState('')
  const [trackingRetryMsg, setTrackingRetryMsg] = useState('')
  const [extractingPhase, setExtractingPhase] = useState(false)
  const [totalBatchesRef] = useState({ current: 1 })
  const [extractedFrameCount, setExtractedFrameCount] = useState(0)  // real-time per-frame count (absolute frame idx)
  const [totalFramesToProcess, setTotalFramesToProcess] = useState(0) // from backend init event
  const [actualStartFrame, setActualStartFrame] = useState(0)         // real start (may differ from propagationStartFrame)
  const [trackFrame, setTrackFrame] = useState(0)
  const activeEsRef = useRef<EventSource | null>(null)
  const [predictingFrame, setPredictingFrame] = useState(false)
  // Clear masks modal state
  const [showClearMasksModal, setShowClearMasksModal] = useState(false)
  const [clearRangeFrom, setClearRangeFrom] = useState('')
  const [clearRangeTo, setClearRangeTo] = useState('')
  // Swap masks state
  const [swapObjA, setSwapObjA] = useState<string>('')
  const [swapObjB, setSwapObjB] = useState<string>('')
  const [swapping, setSwapping] = useState(false)
  const [showSwapModal, setShowSwapModal] = useState(false)
  const [showTrackModal, setShowTrackModal] = useState(false)
  const [trackModalInput, setTrackModalInput] = useState('')
  const [trackModalBelowStart, setTrackModalBelowStart] = useState(false)
  const [swapProgress, setSwapProgress] = useState(0)   // 0–1
  const [swapProgressDone, setSwapProgressDone] = useState(0)
  const [swapProgressTotal, setSwapProgressTotal] = useState(0)

  type ExportStatus = 'idle' | 'running' | 'done' | 'error'
  const [exportStatus, setExportStatus] = useState<ExportStatus>('idle')
  const [exportProgress, setExportProgress] = useState(0)
  const [exportPath, setExportPath] = useState<string | null>(null)
  const [exportError, setExportError] = useState('')

  const objects = video ? Object.values(video.objects) : []
  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  // ── Restore progress from server on video load ────────────────────────────

  useEffect(() => {
    if (!pid || !vid) return
    getPropagationStatus(pid, vid).then(status => {
      const startF = status.start_frame || 0
      const total = status.total_frames || 1
      const lastFrame = status.last_frame >= 0 ? status.last_frame : startF
      // Seed progress bars from saved state
      setTrackFrame(lastFrame + 1)
      setExtractedFrameCount(lastFrame + 1)
      setActualStartFrame(startF)

      totalBatchesRef.current = Math.max(1, Math.ceil((total - startF) / 1000))

      if (status.propagation_complete) {
        // Already done — make sure store reflects this
        if (useStore.getState().propagationStatus !== 'done') {
          setPropagationStatus('done')
          setPropagationProgress(1, lastFrame)
        }
        // Load uncertainty and corrections (they're only set in the done-event
        // handler during live propagation, so we must restore them on page load)
        Promise.all([getUncertainty(pid, vid), getCorrections(pid, vid)])
          .then(([uncertainty, corrections]) => {
            setUncertaintyData(uncertainty)
            setCorrections(corrections)
          })
          .catch(() => { /* non-critical */ })
      } else if (status.is_paused) {
        setPropagationStatus('paused')
        setPropagationPausedAtFrame(status.paused_at_frame ?? lastFrame)
      } else if (status.is_running) {
        // Propagation running on backend — reconnect SSE
        setPropagationStatus('running')
        setTrackingError('')
        setTrackingRetryMsg('')
        setExtractingPhase(false)
        _connectSSE(0)
      }
    }).catch(() => { /* status endpoint not yet available or error — ignore */ })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid, vid])

  // ── Add Object ──────────────────────────────────────────────────────────────

  async function handleAddObject() {
    if (!newObjName.trim()) return
    const idx = objects.length
    const color = getObjectColor(idx)
    const obj = await addObject(pid, vid, newObjName.trim(), color)
    updateVideo({
      objects: {
        ...(video?.objects ?? {}),
        [obj.id]: obj,
      },
    })
    setCurrentObject(obj.id)
    setNewObjName('')
    setAddingObject(false)
    // Session is auto-initialized on first annotation click (no need here)
  }

  // ── Init Session ─────────────────────────────────────────────────────────────

  async function handleInitSession() {
    setInitializingSession(true)
    try {
      await initSession(pid, vid)
      setSessionInitialized(true)
    } catch (e: unknown) {
      console.error('Session init failed:', e)
      throw e  // re-throw so callers (handleConfirmTrack) can detect failure
    } finally {
      setInitializingSession(false)
    }
  }

  // ── Track Objects ─────────────────────────────────────────────────────────────

  function _connectSSE(retryCount: number) {
    const MAX_RETRIES = 3
    // Close any existing connection
    activeEsRef.current?.close()

    if (!pid || !vid) {
      console.error('[SSE] Cannot connect: pid or vid is empty', { pid, vid })
      setTrackingError('Internal error: project/video ID missing')
      setPropagationStatus('error')
      return
    }
    const sseUrl = `/api/projects/${pid}/videos/${vid}/propagate?start_frame=${propagationStartFrame}`
    console.log(`[SSE] Connecting to propagation SSE (attempt ${retryCount + 1})...`, { pid, vid, propagationStartFrame, sseUrl })
    const es = startPropagationSSE(pid, vid, propagationStartFrame)
    activeEsRef.current = es

    es.onopen = () => {
      console.log('[SSE] Connection opened successfully')
    }

    es.addEventListener('init', (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      setTotalFramesToProcess(data.frames_to_process ?? 0)
      setActualStartFrame(data.actual_start ?? 0)
      totalBatchesRef.current = data.total_batches ?? 1
      setExtractedFrameCount(data.actual_start ?? 0)
    })
    es.addEventListener('catch_up', (e: MessageEvent) => {
      // Server sent current position when reconnecting mid-run
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
        // Extraction for this batch is done — confirm it
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
      // data.progress = total_propagated / num_frames (absolute, not per-batch)
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
      const fresh = await getProject(pid)
      setProject(fresh)
      // Load uncertainty and corrections, then switch to uncertainty tab
      try {
        const [uncertainty, corrections] = await Promise.all([
          getUncertainty(pid, vid),
          getCorrections(pid, vid),
        ])
        setUncertaintyData(uncertainty)
        setCorrections(corrections)
        if (Object.keys(uncertainty.per_frame ?? {}).length > 0) {
          setViewerTab('uncertainty')
        }
      } catch { /* non-critical */ }
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
    es.onerror = (event) => {
      const status = useStore.getState().propagationStatus
      console.error('[SSE] Connection error:', { readyState: es.readyState, status, event, url: sseUrl })
      if (status === 'done') return
      es.close()
      activeEsRef.current = null
      setExtractingPhase(false)
      if (retryCount < MAX_RETRIES) {
        const attempt = retryCount + 1
        console.log(`[SSE] Retrying connection (${attempt}/${MAX_RETRIES})...`)
        setTrackingRetryMsg(`Connection lost. Retrying (${attempt}/${MAX_RETRIES})...`)
        setPropagationStatus('running')
        setTimeout(() => _connectSSE(attempt), 3000)
      } else {
        setPropagationStatus('error')
        setTrackingError(`Connection lost after ${MAX_RETRIES} retries. Check that backend is running.`)
        setTrackingRetryMsg('')
      }
    }
  }

  async function _saveStartFrame(f: number) {
    if (!pid || !vid) return
    try {
      await updateVideoMeta(pid, vid, { start_frame: f })
      updateVideo({ start_frame: f })
    } catch { /* non-critical */ }
  }

  function _commitStartFrame() {
    const clamped = Math.max(0, Math.min(video?.num_frames ? video.num_frames - 1 : 0, propagationStartFrame))
    setPropagationStartFrame(clamped)
    setCurrentFrame(clamped)
    _saveStartFrame(clamped)
  }

  function handleTrack() {
    setTrackModalInput(String(propagationStartFrame))
    setTrackModalBelowStart(false)
    setShowTrackModal(true)
  }

  async function handleConfirmTrack() {
    const parsed = parseInt(trackModalInput)
    const resolvedFrame = isNaN(parsed) ? propagationStartFrame : parsed
    if (resolvedFrame < propagationStartFrame) {
      if (!trackModalBelowStart) {
        setTrackModalBelowStart(true)
        return
      }
      // User confirmed despite warning — proceed from propagationStartFrame
    }
    setTrackModalBelowStart(false)
    setShowTrackModal(false)
    setTrackingError('')
    setTrackingRetryMsg('')

    if (!sessionInitialized) {
      try {
        await handleInitSession()
      } catch {
        setTrackingError('Failed to initialize SAM session')
        return
      }
    }

    const startFrom = Math.max(propagationStartFrame, resolvedFrame)

    if (startFrom > propagationStartFrame) {
      // Clear masks from startFrom onwards, then resume from there
      try {
        await resumeFromFrame(pid, vid, startFrom, true)
      } catch {
        setTrackingError('Failed to clear masks for the selected start frame')
        return
      }
      setPropagationStatus('running')
      setExtractingPhase(false)
      setTrackFrame(startFrom)
      setExtractedFrameCount(startFrom)
      setTotalFramesToProcess(0)
      setActualStartFrame(startFrom)
      // Seed from the last kept frame (startFrom - 1), so backend's actual_start = startFrom
      _connectSSEResume(startFrom - 1, 0)
    } else {
      // Normal track from the configured start frame
      setPropagationStatus('running')
      setExtractingPhase(false)
      setTrackFrame(propagationStartFrame)
      setExtractedFrameCount(propagationStartFrame)
      setTotalFramesToProcess(0)
      setActualStartFrame(propagationStartFrame)
      totalBatchesRef.current = 1
      console.log('[Track] About to call _connectSSE(0)', { pid, vid, propagationStartFrame, sessionInitialized })
      _connectSSE(0)
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
      // 400 means propagation wasn't running on the backend — the task likely
      // crashed before the user hit pause.  Just reset to idle silently.
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
      const fresh = await getProject(pid)
      setProject(fresh)
      try {
        const [uncertainty, corrections] = await Promise.all([
          getUncertainty(pid, vid),
          getCorrections(pid, vid),
        ])
        setUncertaintyData(uncertainty)
        setCorrections(corrections)
        if (Object.keys(uncertainty.per_frame ?? {}).length > 0) {
          setViewerTab('uncertainty')
        }
      } catch { /* non-critical */ }
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

  // ── Export Video ─────────────────────────────────────────────────────────────

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
      // Use functional updater to read current status without stale closure
      setExportStatus(prev => {
        if (prev !== 'done') {
          setExportError('Connection lost during export')
          es.close()
          return 'error'
        }
        return prev
      })
    }
  }

  // ── Predict Frame ────────────────────────────────────────────────────────────

  async function handlePredictFrame() {
    if (!pid || !vid) return
    setPredictingFrame(true)
    setPendingInferenceFrame(null)
    const frame = currentFrame
    const usePrev = config.usePrevFrameMask

    try {
      const result = await predictFrame(pid, vid, frame, usePrev)
      if (Object.keys(result.masks).length > 0) {
        setSavedMask(result.frame_idx, result.masks)
        // Auto-save to inference state so it anchors future tracking
        await saveFrameInference(pid, vid, result.frame_idx)
        addToast(`Frame ${result.frame_idx} predicted and saved.`, 'success')
        // Refresh project to pick up any newly registered instance objects
        const fresh = await getProject(pid)
        setProject(fresh)
      } else {
        addToast(`Frame ${frame}: no masks returned from prediction.`, 'error')
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Prediction failed'
      addToast(msg, 'error')
    } finally {
      setPredictingFrame(false)
    }
  }

  function handleClearFrameMasks() {
    setClearRangeFrom(String(currentFrame))
    setClearRangeTo(String(video ? video.num_frames - 1 : currentFrame))
    setShowClearMasksModal(true)
  }

  async function handleConfirmClearMasks(mode: ClearMasksMode | 'this_frame') {
    if (!pid || !vid) return
    setShowClearMasksModal(false)
    try {
      if (mode === 'this_frame') {
        const result = await clearFrameMasks(pid, vid, currentFrame)
        if (result.status === 'protected') {
          addToast(`Frame ${currentFrame} is a seed keyframe and cannot be cleared.`, 'error')
          return
        }
        setSavedMask(currentFrame, {})
        store.setCurrentFrameMasks({})
        clearMaskCache()
        addToast(`Masks cleared for frame ${currentFrame}.`, 'success')
      } else if (mode === 'from_frame') {
        const result = await clearMasksBulk(pid, vid, 'from_frame', currentFrame)
        clearMaskCache()
        store.resetVideoState()
        addToast(`Cleared ${result.deleted_frames} frame(s) from frame ${currentFrame} onwards (seed frames preserved).`, 'success')
      } else if (mode === 'range') {
        const from = parseInt(clearRangeFrom)
        const to = parseInt(clearRangeTo)
        if (isNaN(from) || isNaN(to) || from > to) {
          addToast('Invalid range.', 'error')
          return
        }
        const result = await clearMasksBulk(pid, vid, 'range', from, to)
        clearMaskCache()
        store.resetVideoState()
        addToast(`Cleared ${result.deleted_frames} frame(s) in range ${from}–${to} (seed frames preserved).`, 'success')
      } else {
        // all
        const result = await clearMasksBulk(pid, vid, 'all')
        clearMaskCache()
        store.resetVideoState()
        addToast(`Cleared ${result.deleted_frames} frame(s) (seed frames preserved).`, 'success')
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Clear failed'
      addToast(msg, 'error')
    }
  }

  async function handleSaveInference() {
    if (!pid || !vid || pendingInferenceFrame === null) return
    try {
      await saveFrameInference(pid, vid, pendingInferenceFrame)
      addToast(`Frame ${pendingInferenceFrame} saved to inference state.`, 'success')
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Save failed'
      addToast(msg, 'error')
    } finally {
      setPendingInferenceFrame(null)
    }
  }

  // ── Swap Masks ────────────────────────────────────────────────────────────────

  function openSwapModal() {
    if (!swapObjA || !swapObjB || swapObjA === swapObjB) return
    setShowSwapModal(true)
  }

  async function handleSwapMasks(mode: SwapMode) {
    if (!swapObjA || !swapObjB || swapObjA === swapObjB) return
    setShowSwapModal(false)
    setSwapping(true)

    if (mode === 'this_frame') {
      try {
        const result = await swapMasks(pid, vid, currentFrame, swapObjA, swapObjB, mode)
        if (result.masks) setSavedMask(currentFrame, result.masks)
        addToast(`Swapped masks for frame ${currentFrame}`, 'success')
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Swap failed'
        addToast(msg, 'error')
      } finally {
        setSwapping(false)
      }
      return
    }

    // all_future: use SSE stream so we get progress and avoid proxy timeouts
    const totalFrames = (video?.num_frames ?? 1) - currentFrame
    setSwapProgress(0)
    setSwapProgressDone(0)
    setSwapProgressTotal(totalFrames)

    const es = startSwapAllSSE(pid, vid, currentFrame, swapObjA, swapObjB)

    es.addEventListener('progress', (e: MessageEvent) => {
      const data = JSON.parse(e.data) as { done: number; total: number; swapped: number }
      setSwapProgressDone(data.done)
      setSwapProgressTotal(data.total)
      setSwapProgress(data.total > 0 ? data.done / data.total : 0)
    })

    es.addEventListener('done', (e: MessageEvent) => {
      es.close()
      const data = JSON.parse(e.data) as { swapped_frames: number; frame_idx: number; masks: Record<string, string> }
      if (data.masks) setSavedMask(currentFrame, data.masks)
      clearMaskCache()
      setSwapping(false)
      setSwapProgress(0)
      addToast(`Swapped masks for ${data.swapped_frames} frames (frame ${currentFrame} onwards)`, 'success')
    })

    es.addEventListener('error', () => {
      es.close()
      setSwapping(false)
      setSwapProgress(0)
      addToast('Swap failed', 'error')
    })
  }

  // ── Start Over ────────────────────────────────────────────────────────────────

  async function handleStartOver() {
    if (!confirm('Clear all annotations and masks for this video?')) return
    if (!pid || !currentVideoId) return
    try {
      await resetVideo(pid, currentVideoId)
      clearMaskCache()
      resetVideoState()
      // Reload project so objects/prompts are cleared in store
      const updated = await getProject(pid)
      setProject(updated)
    } catch (e) {
      console.error('Reset failed', e)
    }
  }

  // ─── Empty state ──────────────────────────────────────────────────────────────

  if (!video) {
    return (
      <div className="flex flex-col h-full items-center justify-center p-6 text-center">
        <p className="text-[#555] text-sm leading-relaxed">
          Click an object in the video to start
        </p>
        <p className="text-[#444] text-xs mt-2 leading-relaxed">
          You'll be able to label and track objects across all video frames.
        </p>
      </div>
    )
  }

  const isTracking = propagationStatus === 'running'
  const isPaused = propagationStatus === 'paused'
  const isDone = propagationStatus === 'done'
  const hasObjects = objects.length > 0

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
            : 'Add objects and draw points on the frame. Press "Track objects" to propagate.'
        }
      />

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
            minInstances={obj.min_instances}
            maxInstances={obj.max_instances}
          />
        ))}

        {/* Add new object */}
        {addingObject ? (
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
            disabled={isTracking}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-dashed border-[#333] text-[#666] hover:border-[#555] hover:text-[#aaa] transition-colors text-sm"
          >
            <div className="w-8 h-8 rounded-lg border border-[#333] flex items-center justify-center">
              <Plus size={14} />
            </div>
            Add another object
          </button>
        )}
      </div>

      {/* Session initializing indicator */}
      {initializingSession && (
        <div className="mx-3 mb-2 flex items-center gap-2 text-xs text-[#888]">
          <Loader size={12} className="animate-spin" />
          Loading SAM model...
        </div>
      )}

      {/* Propagation progress */}
      {isTracking && (() => {
        // Use backend-reported total; fall back to local estimate until init event arrives
        const totalFrames = totalFramesToProcess > 0
          ? totalFramesToProcess
          : Math.max(1, (video?.num_frames ?? 1) - propagationStartFrame)
        // Extraction: real-time count relative to actual start
        const extractDone = Math.min(totalFrames, Math.max(0, extractedFrameCount - actualStartFrame))
        const extractPct = totalFrames > 0 ? Math.min(1, extractDone / totalFrames) : 0
        // Tracking: frames propagated so far relative to actual start
        const trackDone = Math.min(totalFrames, Math.max(0, trackFrame - actualStartFrame))
        const trackPct = totalFrames > 0 ? Math.min(1, trackDone / totalFrames) : 0
        return (
          <div className="mx-3 mb-2 space-y-2">
            {/* Extracting frames bar */}
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
            {/* Tracking objects bar */}
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
          disabled={isTracking}
          className="w-full text-xs py-1 px-2 rounded bg-[#1a1a1a] border border-[#333] text-[#ccc] disabled:opacity-40"
        />
      </div>

      {/* Swap all-future progress bar */}
      {swapping && swapProgressTotal > 0 && (
        <div className="mx-3 mb-2 space-y-1 flex-shrink-0">
          <div className="flex justify-between text-xs text-[#888]">
            <span>Swapping identities...</span>
            <span>{swapProgressDone}/{swapProgressTotal}</span>
          </div>
          <div className="h-1.5 bg-[#2a2a2a] rounded-full overflow-hidden">
            <div
              className="h-full bg-amber-500 transition-all duration-200"
              style={{ width: `${swapProgress * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Swap masks UI — shown when frame has saved masks and 2+ objects exist */}
      {objects.length >= 2 && savedMaskCache[currentFrame] && (
        <div className="mx-2 mb-2 rounded-lg border border-[#333] bg-[#111] p-2.5 flex-shrink-0">
          <p className="text-xs text-[#666] mb-2 flex items-center gap-1.5">
            <ArrowRightLeft size={10} />
            Swap masks — frame {currentFrame}
          </p>
          <div className="flex items-center gap-1.5">
            <select
              value={swapObjA}
              onChange={e => setSwapObjA(e.target.value)}
              className="flex-1 text-xs py-1 px-1.5 rounded bg-[#1a1a1a] border border-[#333] text-[#ccc] min-w-0"
            >
              <option value="">Object A</option>
              {objects.map(o => (
                <option key={o.id} value={o.id}>{o.name}</option>
              ))}
            </select>
            <ArrowRightLeft size={10} className="text-[#555] flex-shrink-0" />
            <select
              value={swapObjB}
              onChange={e => setSwapObjB(e.target.value)}
              className="flex-1 text-xs py-1 px-1.5 rounded bg-[#1a1a1a] border border-[#333] text-[#ccc] min-w-0"
            >
              <option value="">Object B</option>
              {objects.map(o => (
                <option key={o.id} value={o.id}>{o.name}</option>
              ))}
            </select>
            <button
              onClick={openSwapModal}
              disabled={swapping || !swapObjA || !swapObjB || swapObjA === swapObjB}
              className="btn btn-ghost text-xs py-1 px-2 disabled:opacity-40 flex-shrink-0"
            >
              {swapping ? <Loader size={10} className="animate-spin" /> : 'Swap'}
            </button>
          </div>
        </div>
      )}

      {/* Paused state actions */}
      {isPaused && (
        <div className="mx-2 mb-2 rounded-lg border border-amber-800/40 bg-amber-400/5 p-2.5 flex-shrink-0">
          <p className="text-xs text-amber-400 mb-2">
            Paused at frame {propagationPausedAtFrame}. Annotate new frames, then:
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
              onClick={handleTrack}
              disabled={!hasObjects}
              className="btn btn-ghost flex-1 flex items-center justify-center gap-1 text-xs py-1.5 disabled:opacity-40"
              title="Choose a frame to start tracking from"
            >
              <SkipForward size={10} />
              From frame
            </button>
            <button
              onClick={handleRestartFromStart}
              disabled={!hasObjects}
              className="btn btn-ghost flex-1 flex items-center justify-center gap-1 text-xs py-1.5 disabled:opacity-40"
              title="Restart tracking from the start frame with updated inference state"
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
        <button
          onClick={handlePredictFrame}
          disabled={predictingFrame || isTracking || !hasObjects}
          className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40"
          style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
          title={`Predict mask for frame ${currentFrame}`}
        >
          {predictingFrame
            ? <Loader size={10} className="animate-spin" />
            : <Zap size={10} />
          }
          <span>Predict</span>
        </button>
        {isTracking ? (
          <button
            onClick={handlePause}
            className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 text-amber-400 hover:text-amber-300"
            style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
          >
            <Pause size={10} />
            <span>Pause</span>
          </button>
        ) : (
          <button
            onClick={handleTrack}
            disabled={isPaused || !hasObjects || initializingSession}
            className="btn btn-primary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40"
            style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
          >
            <ChevronRight size={10} />
            <span>Track</span>
            <span>objects</span>
          </button>
        )}
      </div>

      {/* Track start-frame modal */}
      {showTrackModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-80 shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-1">Start tracking</h3>
            <p className="text-xs text-[#666] mb-4 leading-relaxed">
              Choose which frame to begin tracking from. Frames before this will keep their existing masks.
            </p>
            <label className="text-xs text-[#888] mb-1 block">Start from frame</label>
            <input
              type="number"
              min={0}
              max={video ? video.num_frames - 1 : 999999}
              value={trackModalInput}
              onChange={e => { setTrackModalInput(e.target.value); setTrackModalBelowStart(false) }}
              className="w-full mb-2 text-sm py-1.5 px-2.5 rounded bg-[#111] border border-[#333] text-[#ccc]"
              autoFocus
              onKeyDown={e => { if (e.key === 'Enter') handleConfirmTrack() }}
            />
            {trackModalBelowStart && (
              <p className="text-xs text-amber-400 mb-3 leading-relaxed">
                Frame {trackModalInput} is before the configured start frame ({propagationStartFrame}), so tracking will begin at frame {propagationStartFrame}. Cancel to change the start frame first.
              </p>
            )}
            {!trackModalBelowStart && <div className="mb-4" />}
            <div className="flex gap-2">
              <button
                onClick={() => setShowTrackModal(false)}
                className="btn btn-ghost flex-1 py-2 text-xs"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmTrack}
                className="btn btn-primary flex-1 py-2 text-xs font-medium"
              >
                Start tracking
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
            <p className="text-xs text-[#666] mb-4 leading-relaxed">
              Seed keyframe masks (from "Predict") are always preserved.
            </p>
            <div className="flex flex-col gap-2">
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

      {/* Swap Mode Modal */}
      {showSwapModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-80 shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-3">Swap Labels</h3>
            <p className="text-xs text-[#888] mb-4">
              Swap the labels of <span className="text-[#ccc] font-medium">{objects.find(o => o.id === swapObjA)?.name}</span> and{' '}
              <span className="text-[#ccc] font-medium">{objects.find(o => o.id === swapObjB)?.name}</span>.
            </p>
            <p className="text-xs text-[#888] mb-4">
              Apply this swap to:
            </p>
            <div className="flex flex-col gap-2">
              <button
                onClick={() => handleSwapMasks('this_frame')}
                className="btn btn-primary w-full py-2 text-xs font-medium"
              >
                This frame only (frame {currentFrame})
              </button>
              <button
                onClick={() => handleSwapMasks('all_future')}
                className="btn btn-ghost w-full py-2 text-xs font-medium border border-amber-600/50 text-amber-400 hover:bg-amber-600/10"
              >
                All future frames (frame {currentFrame}+)
              </button>
              <button
                onClick={() => setShowSwapModal(false)}
                className="btn btn-ghost w-full py-2 text-xs text-[#666] hover:text-[#aaa]"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
