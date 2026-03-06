import React, { useEffect, useRef, useState } from 'react'
import { Plus, RotateCcw, ChevronRight, Loader, Download, X, Zap, Save } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { addObject, initSession, startPropagationSSE, startExportSSE, getProject, resetVideo, predictFrame, saveFrameInference, getPropagationStatus } from '../../api/client'
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
    sessionInitialized, setSessionInitialized,
    propagationStartFrame, setPropagationStartFrame, setCurrentFrame,
    resetVideoState, updateVideo,
    setProject, setSavedMask,
    pendingInferenceFrame, setPendingInferenceFrame,
    config, persistConfig, addToast,
  } = store

  const [addingObject, setAddingObject] = useState(false)
  const [newObjName, setNewObjName] = useState('')
  const [initializingSession, setInitializingSession] = useState(false)
  const [trackingError, setTrackingError] = useState('')
  const [trackingRetryMsg, setTrackingRetryMsg] = useState('')
  const [extractingPhase, setExtractingPhase] = useState(false)
  const [trackingOverallProgress, setTrackingOverallProgress] = useState(0)
  const [totalBatchesRef] = useState({ current: 1 })
  // extractedUpTo: frames confirmed fully extracted (updated on initializing_session)
  // extractCurrentEnd: end of the batch currently being extracted (updated on extracting)
  const [extractedUpTo, setExtractedUpTo] = useState(0)
  const [extractCurrentEnd, setExtractCurrentEnd] = useState(0)
  const [trackFrame, setTrackFrame] = useState(0)
  const activeEsRef = useRef<EventSource | null>(null)
  const [predictingFrame, setPredictingFrame] = useState(false)

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
      const done = status.frames_done || 0
      const lastFrame = status.last_frame >= 0 ? status.last_frame : startF
      // Seed progress bars from saved state
      setTrackFrame(lastFrame + 1)
      setExtractedUpTo(lastFrame + 1)
      setExtractCurrentEnd(lastFrame + 1)
      setTrackingOverallProgress(done / total)
      totalBatchesRef.current = Math.max(1, Math.ceil((total - startF) / 1000))

      if (status.propagation_complete) {
        // Already done — make sure store reflects this
        if (useStore.getState().propagationStatus !== 'done') {
          setPropagationStatus('done')
          setPropagationProgress(1, lastFrame)
        }
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
    } finally {
      setInitializingSession(false)
    }
  }

  // ── Track Objects ─────────────────────────────────────────────────────────────

  function _connectSSE(retryCount: number) {
    const MAX_RETRIES = 3
    // Close any existing connection
    activeEsRef.current?.close()

    const es = startPropagationSSE(pid, vid, propagationStartFrame)
    activeEsRef.current = es

    es.addEventListener('catch_up', (e: MessageEvent) => {
      // Server sent current position when reconnecting mid-run
      const data = JSON.parse(e.data)
      const done: number = data.frames_done ?? 0
      const total: number = data.total_frames ?? 1
      const last: number = data.last_frame ?? -1
      setTrackFrame(last + 1)
      setExtractedUpTo(last + 1)
      setExtractCurrentEnd(last + 1)
      setTrackingOverallProgress(done / total)
      totalBatchesRef.current = Math.max(1, Math.ceil(total / 1000))
    })
    es.addEventListener('batch_start', (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      totalBatchesRef.current = data.total_batches ?? 1
      if (data.status === 'extracting') {
        setExtractingPhase(true)
        setExtractCurrentEnd(data.batch_end ?? 0)
      } else if (data.status === 'initializing_session') {
        // Extraction for this batch is done — confirm it
        setExtractedUpTo(data.batch_end ?? 0)
        setExtractingPhase(false)
      }
    })
    es.addEventListener('progress', (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      setExtractingPhase(false)
      setTrackFrame(data.frame + 1)
      // data.progress = total_propagated / num_frames (absolute, not per-batch)
      setTrackingOverallProgress(data.progress)
      setPropagationProgress(data.progress, data.frame)
    })
    es.addEventListener('done', async (e: MessageEvent) => {
      const data: PropagationEvent = JSON.parse(e.data)
      setExtractingPhase(false)
      setTrackingOverallProgress(1)
      setPropagationStatus('done')
      setPropagationProgress(1, data.frame ?? 0)
      setTrackingRetryMsg('')
      es.close()
      activeEsRef.current = null
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
        setTimeout(() => _connectSSE(attempt), 3000)
      } else {
        setPropagationStatus('error')
        setTrackingError(`Connection lost after ${MAX_RETRIES} retries`)
        setTrackingRetryMsg('')
      }
    }
  }

  async function handleTrack() {
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

    setPropagationStatus('running')
    setExtractingPhase(false)
    setTrackingOverallProgress(0)
    setExtractedUpTo(propagationStartFrame)
    setExtractCurrentEnd(propagationStartFrame)
    setTrackFrame(propagationStartFrame)
    totalBatchesRef.current = 1
    _connectSSE(0)
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
        setPendingInferenceFrame(result.frame_idx)
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
  const isDone = propagationStatus === 'done'
  const hasObjects = objects.length > 0
  const hasPrompts = Object.keys(video.point_prompts ?? {}).length > 0

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
        const totalFrames = (video?.num_frames ?? 1) - propagationStartFrame
        // During extraction: show target batch end; after: show confirmed extracted count
        const extractDisplayEnd = extractingPhase ? extractCurrentEnd : extractedUpTo
        const extractDone = Math.max(0, extractDisplayEnd - propagationStartFrame)
        const extractPct = totalFrames > 0 ? Math.min(1, extractDone / totalFrames) : 0
        // Tracking: frames processed so far
        const trackDone = Math.max(0, trackFrame - propagationStartFrame)
        const trackPct = Math.min(1, trackingOverallProgress)
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
                  className={`h-full transition-all duration-300 ${extractingPhase ? 'bg-amber-500' : 'bg-emerald-600'}`}
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
                  className="h-full bg-blue-500 transition-all duration-300"
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
          onKeyDown={e => {
            if (e.key === 'Enter') {
              const clamped = Math.max(0, Math.min(video.num_frames - 1, propagationStartFrame))
              setPropagationStartFrame(clamped)
              setCurrentFrame(clamped)
              persistConfig()
            }
          }}
          disabled={isTracking}
          className="w-full text-xs py-1 px-2 rounded bg-[#1a1a1a] border border-[#333] text-[#ccc] disabled:opacity-40"
        />
      </div>

      {/* Save-to-inference-state prompt */}
      {pendingInferenceFrame !== null && (
        <div className="mx-2 mb-2 p-2.5 rounded-lg border border-blue-800/50 bg-[#0d1a2a] flex-shrink-0">
          <div className="flex items-start justify-between gap-2 mb-2">
            <p className="text-xs text-[#aaa] leading-snug">
              <span className="text-blue-400 font-medium">Frame {pendingInferenceFrame} predicted.</span>
              {' '}Save as a keyframe so it anchors future tracking?
            </p>
            <button onClick={() => setPendingInferenceFrame(null)} className="text-[#555] hover:text-[#aaa] flex-shrink-0 mt-0.5">
              <X size={11} />
            </button>
          </div>
          <div className="flex gap-1.5">
            <button
              onClick={handleSaveInference}
              className="btn btn-primary flex items-center gap-1 text-xs py-1"
            >
              <Save size={10} />
              Save to inference state
            </button>
            <button
              onClick={() => setPendingInferenceFrame(null)}
              className="btn btn-ghost text-xs py-1"
            >
              Dismiss
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
        <button
          onClick={handlePredictFrame}
          disabled={predictingFrame || isTracking || !hasObjects}
          className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40"
          style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
          title={`Predict mask for frame ${store.currentFrame}`}
        >
          {predictingFrame
            ? <Loader size={10} className="animate-spin" />
            : <Zap size={10} />
          }
          <span>Predict</span>
        </button>
        <button
          onClick={handleTrack}
          disabled={isTracking || !hasObjects || initializingSession}
          className="btn btn-primary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-0.5 py-2 disabled:opacity-40"
          style={{ fontSize: 'clamp(7px, 4cqi, 11px)', lineHeight: 1.2 }}
        >
          {isTracking ? (
            <>
              <Loader size={10} className="animate-spin" />
              <span>Tracking...</span>
            </>
          ) : (
            <>
              <ChevronRight size={10} />
              <span>Track</span>
              <span>objects</span>
            </>
          )}
        </button>
      </div>
    </div>
  )
}
