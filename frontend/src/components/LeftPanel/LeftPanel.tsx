import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Plus, RotateCcw, ChevronRight, ChevronDown, Loader, Download, X, Pause, Play, SkipBack, SkipForward, Trash2, GripVertical, ArrowLeftRight, Undo2, Redo2, Sparkles } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import {
  addObject, addSubObject, initSession, startPropagationSSE, startExportSSE, getProject, resetVideo,
  clearFrameMasks, clearMasksBulk, getPropagationStatus, pausePropagation, updateVideoMeta,
  resumeFromFrame, getAnchorFrames, commitAnchorFrame, swapObjectMasks,
  startAnchorRemainderPredictionSSE,
  continueAnchorRemainderReview,
  getSavedMask,
  extractFrame,
  restoreMaskFrames,
  predictFrame,
  type ClearMasksMode,
} from '../../api/client'
import { getObjectColor } from '../../utils/colors'
import { clearMaskCache } from '../../utils/maskUtils'
import {
  ANCHOR_BATCH_SIZE_MAX,
  ANCHOR_BATCH_SIZE_MIN,
  annotatedAnchorIndicesFromFrames,
  computeAnchorFrames,
  firstUnlabeledAnchorIndex,
  normalizeAnchorBatchSize,
  videoAnchorBatchSize,
} from '../../utils/anchorFrames'
import NumericDraftInput from '../NumericDraftInput'
import ObjectCard from './ObjectCard'
import StepIndicator from './StepIndicator'
import type { PropagationEvent, ObjectKind } from '../../types'

function targetIsTypingContext(target: EventTarget | null): boolean {
  const el = target instanceof HTMLElement ? target : null
  if (!el) return false
  if (el.isContentEditable) return true
  const tag = el.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
  return false
}

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
    setCurrentFrameMasks,
    addToast,
    config,
    setConfig,
    // Anchor phase
    anchorPhase, setAnchorPhase,
    anchorFrames, setAnchorFrames,
    currentAnchorIndex, setCurrentAnchorIndex,
    annotatedAnchorIndices, addAnnotatedAnchor,
    resetAnchorState,
    anchorRemainderInferencing,
    setAnchorRemainderInferencing,
    anchorRemainderAwaitingCommit,
    setAnchorRemainderAwaitingCommit,
    setAnnotatedAnchorIndices,
    invalidateSavedMaskFrame,
    undoStack, redoStack, historyBusy, undoLast, redoLast,
    viewerTab,
  } = store

  const handleCommitAnchorRef = useRef<() => Promise<void>>(async () => {})

  const [addingObject, setAddingObject] = useState(false)
  const [newObjName, setNewObjName] = useState('')
  // Sub-object add form (keyed by the parent object being added to)
  const [subParentId, setSubParentId] = useState<string | null>(null)
  const [subName, setSubName] = useState('')
  const [subKind, setSubKind] = useState<ObjectKind>('segmentation')
  const [subBlobFrac, setSubBlobFrac] = useState(0.06)
  const [predicting, setPredicting] = useState(false)
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
  /** Backend constant: anchors [0,N) manual, anchors [N,end) inferred via sequential propagation. */
  const manualAnchorPrefixRef = useRef(5)
  const anchorRemainderEsRef = useRef<EventSource | null>(null)

  /** Backend paused inferencer until Good / Done after Edit */
  const [anchorReviewPrompt, setAnchorReviewPrompt] = useState<{
    frameIdx: number
    anchorIndex: number
    anchorsDone: number
    anchorsQueued: number
  } | null>(null)
  /** Shown while remainder inference runs but review UI / mask fetch is not active yet */
  const [anchorRemainderProgressText, setAnchorRemainderProgressText] = useState('')
  const [anchorReviewBusy, setAnchorReviewBusy] = useState(false)
  /** Fetching masks from API before showing the review chip */
  const [anchorReviewLoading, setAnchorReviewLoading] = useState(false)
  const [anchorReviewPanelOffsets, setAnchorReviewPanelOffsets] = useState({ right: 24, bottom: 24 })

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

  // Anchor sampling modal (first-time anchor labeling on a video)
  const [showAnchorSamplingModal, setShowAnchorSamplingModal] = useState(false)
  const [anchorSamplingInterval, setAnchorSamplingInterval] = useState(1000)
  const [anchorSamplingLive, setAnchorSamplingLive] = useState<number | null>(null)
  const [anchorSamplingSaving, setAnchorSamplingSaving] = useState(false)

  // Tracking method modal (shown when all anchors are labeled)
  const [showTrackMethodModal, setShowTrackMethodModal] = useState(false)
  const [trackMethodChoice, setTrackMethodChoice] = useState<'sequential' | 'all_anchors'>('sequential')

  // Next-video prompt (shown when all anchors labeled and project has more videos)
  const [showNextVideoModal, setShowNextVideoModal] = useState(false)

  // Swap masks modal
  const [showSwapModal, setShowSwapModal] = useState(false)
  const [swapObjA, setSwapObjA] = useState('')
  const [swapObjB, setSwapObjB] = useState('')
  type SwapMaskScope = 'current' | 'from_current' | 'all'
  const [swapMaskScope, setSwapMaskScope] = useState<SwapMaskScope>('current')
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
  const allAnchorsLabeled =
    (anchorFrames.length > 0 && annotatedAnchorIndices.length >= anchorFrames.length)
    || Boolean(video?.anchor_labeling_complete)

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
      const batchSz = videoAnchorBatchSize(video, useStore.getState().config.anchorBatchSize)
      totalBatchesRef.current = Math.max(1, Math.ceil((total - startF) / batchSz))

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

  useEffect(() => {
    return () => {
      anchorRemainderEsRef.current?.close()
      anchorRemainderEsRef.current = null
    }
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

  function openSubForm(parentId: string) {
    setSubParentId(parentId)
    setSubName('')
    setSubKind('segmentation')
    setSubBlobFrac(0.06)
  }

  async function handlePredictFrame() {
    if (!pid || !vid || !hasObjects || predicting) return
    setPredicting(true)
    try {
      // Make sure the frame is available, then predict all objects on it.
      try { await extractFrame(pid, vid, currentFrame) } catch { /* may already exist */ }
      const result = await predictFrame(pid, vid, currentFrame)
      const masks = result.masks ?? {}
      // Display only — predict_frame does not persist server-side.
      setCurrentFrameMasks(masks, currentFrame)
      setSavedMask(currentFrame, masks)
      const n = Object.keys(masks).length
      addToast(n > 0 ? `Predicted ${n} object${n === 1 ? '' : 's'} on frame #${currentFrame}` : 'No objects predicted on this frame', n > 0 ? 'success' : 'info')
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail ?? (e instanceof Error ? e.message : 'Prediction failed'), 'error')
    } finally {
      setPredicting(false)
    }
  }

  async function handleAddSubObject() {
    if (!subParentId || !subName.trim()) return
    const color = getObjectColor(objects.length)
    const obj = await addSubObject(pid, vid, subParentId, subName.trim(), subKind, {
      color,
      point_blob_frac: subKind === 'point' ? subBlobFrac : undefined,
    })
    updateVideo({ objects: { ...(video?.objects ?? {}), [obj.id]: obj } })
    setCurrentObject(obj.id)
    setSubParentId(null)
    setSubName('')
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

  function labeledAnchorFrameSet(extraIndices: number[] = []): Set<number> {
    const fromVideo = video?.annotated_anchors ?? []
    const frames = new Set(fromVideo)
    for (const i of extraIndices) {
      const fr = anchorFrames[i]
      if (fr !== undefined) frames.add(fr)
    }
    for (const i of annotatedAnchorIndices) {
      const fr = anchorFrames[i]
      if (fr !== undefined) frames.add(fr)
    }
    return frames
  }

  function goToFirstUnlabeledAnchor(
    frames: number[],
    labeledFrames: Set<number>,
    options?: { toastIfAlreadyThere?: boolean },
  ): boolean {
    const nextIndex = firstUnlabeledAnchorIndex(frames, labeledFrames)
    if (nextIndex >= frames.length) return false
    if (
      options?.toastIfAlreadyThere &&
      nextIndex === currentAnchorIndex &&
      currentFrame === frames[nextIndex]
    ) {
      addToast('You are already on the next anchor in the labeling queue.', 'info')
      return true
    }
    anchorEnteredMsRef.current[nextIndex] = Date.now()
    setCurrentAnchorIndex(nextIndex)
    setCurrentFrame(frames[nextIndex]!)
    return true
  }

  const anchorSamplingPreviewCount =
    video && showAnchorSamplingModal
      ? computeAnchorFrames(
          propagationStartFrame,
          video.num_frames,
          anchorSamplingLive ?? anchorSamplingInterval,
        ).length
      : null

  function handleAnnotateAnchorsClick() {
    if (!hasObjects) return
    const committed = video?.annotated_anchors ?? []
    if (committed.length === 0) {
      const initial = videoAnchorBatchSize(video, config.anchorBatchSize)
      setAnchorSamplingInterval(initial)
      setAnchorSamplingLive(initial)
      setShowAnchorSamplingModal(true)
      return
    }
    void handleStartAnchorAnnotation()
  }

  async function confirmAnchorSamplingAndStart() {
    const normalized = normalizeAnchorBatchSize(anchorSamplingInterval, config.anchorBatchSize)
    setAnchorSamplingSaving(true)
    setTrackingError('')
    try {
      if (pid && vid) {
        await updateVideoMeta(pid, vid, { anchor_batch_size: normalized })
        updateVideo({ anchor_batch_size: normalized })
      }
      setConfig({ anchorBatchSize: normalized })
      setAnchorSamplingInterval(normalized)
      setShowAnchorSamplingModal(false)
      await handleStartAnchorAnnotation()
    } catch (e: unknown) {
      const detail =
        e && typeof e === 'object' && 'response' in e
          ? (e as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : undefined
      const msg =
        typeof detail === 'string' ? detail : (e instanceof Error ? e.message : 'Could not save anchor interval')
      setTrackingError(msg)
      addToast(msg, 'error')
    } finally {
      setAnchorSamplingSaving(false)
    }
  }

  async function handleStartAnchorAnnotation() {
    if (!hasObjects) return
    setTrackingError('')

    try {
      const { anchor_frames, manual_anchor_prefix_before_infer: manualPre } = await getAnchorFrames(pid, vid)
      manualAnchorPrefixRef.current = typeof manualPre === 'number' && manualPre >= 1 ? manualPre : 5
      const annFr = video?.annotated_anchors ?? []
      const doneIndices = annotatedAnchorIndicesFromFrames(anchor_frames, annFr)
      setAnnotatedAnchorIndices(doneIndices)

      const nextIndex = firstUnlabeledAnchorIndex(anchor_frames, annFr)
      if (nextIndex >= anchor_frames.length) {
        addToast('All anchor frames are already labeled.', 'info')
        return
      }

      setAnchorFrames(anchor_frames)
      setAnchorPhase(true)
      anchorEnteredMsRef.current = {}
      goToFirstUnlabeledAnchor(anchor_frames, new Set(annFr))

      const resumed = nextIndex > 0
      addToast(
        resumed
          ? `Resuming anchor labeling at frame ${anchor_frames[nextIndex]} (${nextIndex + 1}/${anchor_frames.length}).`
          : "You're being timed — per-anchor and whole-video durations are saved in your project.",
        'success',
      )
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to get anchor frames'
      setTrackingError(msg)
    }
  }

  function handleGoToQueueAnchor() {
    if (!anchorPhase || anchorFrames.length === 0) return
    if (anchorRemainderInferencing && !anchorRemainderAwaitingCommit) return
    goToFirstUnlabeledAnchor(anchorFrames, labeledAnchorFrameSet(), { toastIfAlreadyThere: true })
  }

  const queueAnchorIndex =
    anchorPhase && anchorFrames.length > 0
      ? firstUnlabeledAnchorIndex(anchorFrames, labeledAnchorFrameSet())
      : anchorFrames.length
  const offQueueAnchor =
    queueAnchorIndex < anchorFrames.length &&
    (currentAnchorIndex !== queueAnchorIndex || currentFrame !== anchorFrames[queueAnchorIndex])

  async function runAnchorRemainderPrediction(): Promise<void> {
    setAnchorRemainderInferencing(true)
    setTrackingError('')
    setAnchorReviewPrompt(null)
    setAnchorRemainderAwaitingCommit(false)
    setAnchorReviewBusy(false)
    setAnchorReviewLoading(false)
    setAnchorReviewPanelOffsets({ right: 24, bottom: 24 })
    setAnchorRemainderProgressText('Connecting…')
    anchorRemainderEsRef.current?.close()
    const es = startAnchorRemainderPredictionSSE(pid, vid, true)
    anchorRemainderEsRef.current = es

    const applyAnnotatedIndices = async () => {
      const fresh = await getProject(pid)
      setProject(fresh)
      const annFr = fresh.videos[vid]?.annotated_anchors ?? []
      setAnnotatedAnchorIndices(annotatedAnchorIndicesFromFrames(anchorFrames, annFr))
    }

    const maskLoadTimeoutMs = 90_000
    function withTimeoutMs<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
      return new Promise((resolve, reject) => {
        const id = window.setTimeout(
          () => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)),
          ms,
        )
        p.then(
          v => {
            window.clearTimeout(id)
            resolve(v)
          },
          err => {
            window.clearTimeout(id)
            reject(err)
          },
        )
      })
    }

    try {
      await new Promise<void>((resolve, reject) => {
        let finished = false
        const finishOk = () => {
          if (finished) return
          finished = true
          es.close()
          if (anchorRemainderEsRef.current === es) anchorRemainderEsRef.current = null
          resolve()
        }
        const finishErr = (msg: string) => {
          if (finished) return
          finished = true
          es.close()
          if (anchorRemainderEsRef.current === es) anchorRemainderEsRef.current = null
          reject(new Error(msg))
        }

        es.addEventListener('anchor_predicted', () => {
          void applyAnnotatedIndices()
        })

        es.addEventListener('init', (evt: MessageEvent) => {
          try {
            const d = JSON.parse(evt.data ?? '{}') as { anchors_queued?: number }
            const q = typeof d.anchors_queued === 'number' ? d.anchors_queued : 0
            setAnchorRemainderProgressText(
              q > 0 ? `Predicting ${q} anchor frame${q === 1 ? '' : 's'}…` : 'Predicting anchors…',
            )
          } catch {
            setAnchorRemainderProgressText('Predicting anchors…')
          }
        })

        es.addEventListener('sam_propagate_start', (evt: MessageEvent) => {
          try {
            const d = JSON.parse(evt.data ?? '{}') as { frame_idx?: number; step?: number; steps?: number }
            const step = typeof d.step === 'number' ? d.step : '?'
            const steps = typeof d.steps === 'number' ? d.steps : '?'
            const fi = typeof d.frame_idx === 'number' ? d.frame_idx : '?'
            setAnchorRemainderProgressText(`Anchor ${step}/${steps} · SAM propagating · frame ${fi}`)
          } catch {
            setAnchorRemainderProgressText('SAM propagating…')
          }
        })

        es.addEventListener('review_prompt', (evt: MessageEvent) => {
          void (async () => {
            let d: {
              frame_idx: number
              anchor_index?: number
              anchors_done?: number
              anchors_queued?: number
            }
            try {
              d = JSON.parse(evt.data) as typeof d
            } catch {
              return
            }
            const ai = typeof d.anchor_index === 'number' ? d.anchor_index : 0
            const fidx = d.frame_idx
            setAnchorReviewPrompt(null)
            setAnchorReviewLoading(true)
            invalidateSavedMaskFrame(fidx)
            setCurrentFrame(fidx)
            setCurrentAnchorIndex(ai)
            setCurrentFrameMasks({}, null)
            try {
              await applyAnnotatedIndices()
              try {
                await withTimeoutMs(extractFrame(pid, vid, fidx), maskLoadTimeoutMs, 'extractFrame')
              } catch {
                /* frame may already exist or extract failed */
              }
              const data = await withTimeoutMs(getSavedMask(pid, vid, fidx), maskLoadTimeoutMs, 'getSavedMask')
              const masks = data.masks ?? {}
              setSavedMask(fidx, masks)
              setAnchorReviewPrompt({
                frameIdx: fidx,
                anchorIndex: ai,
                anchorsDone: d.anchors_done ?? 0,
                anchorsQueued: d.anchors_queued ?? 0,
              })
            } catch (err: unknown) {
              const msg = err instanceof Error ? err.message : 'Could not load predicted masks'
              addToast(msg, 'error')
              setAnchorReviewPrompt({
                frameIdx: fidx,
                anchorIndex: ai,
                anchorsDone: d.anchors_done ?? 0,
                anchorsQueued: d.anchors_queued ?? 0,
              })
            } finally {
              setAnchorReviewLoading(false)
            }
          })()
        })

        es.addEventListener('error', (evt: Event) => {
          if (!(evt instanceof MessageEvent)) return
          try {
            const d = JSON.parse(evt.data ?? '{}') as { error?: string }
            setAnchorReviewPrompt(null)
            setAnchorRemainderAwaitingCommit(false)
            setAnchorReviewLoading(false)
            finishErr(d.error ?? 'anchor remainder inference failed')
          } catch {
            setAnchorReviewPrompt(null)
            setAnchorRemainderAwaitingCommit(false)
            setAnchorReviewLoading(false)
            finishErr('anchor remainder inference failed')
          }
        })

        es.addEventListener('done', () => {
          setAnchorReviewPrompt(null)
          setAnchorRemainderAwaitingCommit(false)
          setAnchorReviewLoading(false)
          finishOk()
        })

        es.onerror = () => {
          if (finished) return
          if (es.readyState === EventSource.CLOSED) {
            setAnchorReviewPrompt(null)
            setAnchorRemainderAwaitingCommit(false)
            setAnchorReviewLoading(false)
            finishErr('Anchor inference connection closed unexpectedly.')
          }
        }
      })

      await applyAnnotatedIndices()
      setAnchorPhase(false)
      setCurrentAnchorIndex(anchorFrames.length)
      if (nextVideoId) setShowNextVideoModal(true)
      addToast('All anchor frames are ready.', 'success')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Anchor remainder inference failed'
      setTrackingError(msg)
      addToast(msg, 'error')
    } finally {
      setAnchorRemainderInferencing(false)
      setAnchorReviewPrompt(null)
      setAnchorRemainderAwaitingCommit(false)
      setAnchorReviewBusy(false)
      setAnchorReviewLoading(false)
      setAnchorRemainderProgressText('')
    }
  }

  async function handleAnchorReviewGood() {
    if (!pid || !vid) return
    setAnchorReviewBusy(true)
    setTrackingError('')
    try {
      await continueAnchorRemainderReview(pid, vid)
      setAnchorReviewPrompt(null)
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      const msg = detail ?? (e instanceof Error ? e.message : 'Could not continue anchor inference')
      addToast(msg, 'error')
    } finally {
      setAnchorReviewBusy(false)
    }
  }

  function handleAnchorReviewEdit() {
    const ai = anchorReviewPrompt?.anchorIndex
    setAnchorReviewPrompt(null)
    setAnchorRemainderAwaitingCommit(true)
    if (typeof ai === 'number' && anchorEnteredMsRef.current[ai] === undefined) {
      anchorEnteredMsRef.current[ai] = Date.now()
    }
  }

  function handleAnchorReviewDragStart(e: React.MouseEvent<HTMLDivElement>) {
    e.preventDefault()
    e.stopPropagation()
    const startX = e.clientX
    const startY = e.clientY
    const startRight = anchorReviewPanelOffsets.right
    const startBottom = anchorReviewPanelOffsets.bottom
    const approxW = 288
    const approxH = 220
    const pad = 8
    function move(ev: MouseEvent) {
      const dx = ev.clientX - startX
      const dy = ev.clientY - startY
      let nr = startRight - dx
      let nb = startBottom - dy
      nr = Math.max(pad, Math.min(window.innerWidth - pad - approxW, nr))
      nb = Math.max(pad, Math.min(window.innerHeight - pad - approxH, nb))
      setAnchorReviewPanelOffsets({ right: nr, bottom: nb })
    }
    function up() {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  async function handleCommitAnchor() {
    if (!anchorPhase || anchorFrames.length === 0) return
    const frameIdx = anchorFrames[currentAnchorIndex]
    const ai = currentAnchorIndex
    const awaitingRemainderCommit = anchorRemainderInferencing && anchorRemainderAwaitingCommit
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

    if (awaitingRemainderCommit) {
      setAnchorRemainderAwaitingCommit(false)
      addToast(`This anchor frame: ${frameSecStr}s`, 'success')
      try {
        await continueAnchorRemainderReview(pid, vid)
      } catch (e: unknown) {
        const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        addToast(detail ?? (e instanceof Error ? e.message : 'Could not resume inference'), 'error')
      }
      return
    }

    const nextIndex = ai + 1
    const isLastAnchor = nextIndex >= anchorFrames.length
    const manualN = manualAnchorPrefixRef.current
    const triggerRemainderInfer = anchorFrames.length > manualN && nextIndex === manualN

    if (isLastAnchor && wholeVideoWallMs != null) {
      const totalSecStr = (wholeVideoWallMs / 1000).toFixed(2)
      addToast(
        `This anchor frame: ${frameSecStr}s\nTotal for video: ${totalSecStr}s`,
        'success',
      )
    } else {
      addToast(`This anchor frame: ${frameSecStr}s`, 'success')
    }

    if (triggerRemainderInfer && config.autoInferAnchorRemainder) {
      addToast('Manual anchor prefix complete — inferring remaining anchors…', 'info')
      await runAnchorRemainderPrediction()
      return
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

  handleCommitAnchorRef.current = handleCommitAnchor

  const handleGoToPreviousAnchor = useCallback(() => {
    if (!anchorPhase || anchorFrames.length === 0) return
    if (anchorRemainderInferencing && !anchorRemainderAwaitingCommit) return
    if (currentAnchorIndex <= 0) return
    const prev = currentAnchorIndex - 1
    anchorEnteredMsRef.current[prev] = Date.now()
    setCurrentAnchorIndex(prev)
    setCurrentFrame(anchorFrames[prev]!)
  }, [
    anchorPhase,
    anchorFrames,
    currentAnchorIndex,
    anchorRemainderInferencing,
    anchorRemainderAwaitingCommit,
    setCurrentAnchorIndex,
    setCurrentFrame,
  ])

  const handleGoToPreviousAnchorRef = useRef(handleGoToPreviousAnchor)
  handleGoToPreviousAnchorRef.current = handleGoToPreviousAnchor

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (viewerTab !== 'annotate') return
      if (targetIsTypingContext(e.target)) return
      if (e.key !== 'Enter' || e.repeat) return
      if (e.metaKey || e.ctrlKey || e.altKey) return

      const sNow = anchorReviewPrompt
      if (sNow != null) return

      if (!anchorPhase || anchorFrames.length === 0) return
      const blockedInfer = anchorRemainderInferencing && !anchorRemainderAwaitingCommit
      if (blockedInfer) return

      if (e.shiftKey) {
        if (currentAnchorIndex <= 0) return
        e.preventDefault()
        handleGoToPreviousAnchorRef.current()
        return
      }

      if (!hasObjects) return
      e.preventDefault()
      void handleCommitAnchorRef.current()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [
    viewerTab,
    anchorPhase,
    anchorFrames.length,
    anchorRemainderInferencing,
    anchorRemainderAwaitingCommit,
    hasObjects,
    currentAnchorIndex,
    anchorReviewPrompt,
  ])

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
      const f = currentFrame
      let fromF: number | undefined
      let toF: number | undefined
      if (swapMaskScope === 'current') {
        fromF = f
        toF = f
      } else if (swapMaskScope === 'from_current') {
        fromF = f
        toF = undefined
      } else {
        fromF = undefined
        toF = undefined
      }
      const result = await swapObjectMasks(pid, vid, swapObjA, swapObjB, fromF, toF)
      setShowSwapModal(false)
      clearSavedMaskCache()
      clearMaskCache()
      addToast(`Swapped masks for ${result.frames_swapped} frame(s)`, 'success')
      const a = swapObjA
      const b = swapObjB
      const sf = fromF
      const st = toF
      useStore.getState().pushHistory({
        labelUndo: 'Swap masks',
        labelRedo: 'Swap masks',
        undo: async () => {
          try {
            await swapObjectMasks(pid, vid, a, b, sf, st)
            clearSavedMaskCache()
            clearMaskCache()
          } catch { /* ignore */ }
        },
        redo: async () => {
          try {
            await swapObjectMasks(pid, vid, a, b, sf, st)
            clearSavedMaskCache()
            clearMaskCache()
          } catch { /* ignore */ }
        },
      })
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

  async function _snapshotMasksForFrames(frames: number[]) {
    type MaskData = Record<string, string>
    const out: Record<string, MaskData> = {}
    for (const f of frames) {
      try {
        const d = await getSavedMask(pid, vid, f)
        if (d.masks && Object.keys(d.masks).length > 0) out[String(f)] = d.masks
      } catch { /* empty frame */ }
    }
    return out
  }

  function _framesToSnapshotForClear(mode: ClearMasksMode | 'this_frame'): number[] {
    const cap = (video?.num_frames ?? 1) - 1
    const f0 = currentFrame
    if (mode === 'this_frame') return [f0]
    const pool = new Set<number>([...(video?.propagated_frames ?? [])])
    for (const k of Object.keys(savedMaskCache)) pool.add(Number(k))
    const sorted = [...pool].filter(f => f >= 0 && f <= cap).sort((a, b) => a - b)
    if (mode === 'from_frame') return sorted.filter(f => f >= f0)
    if (mode === 'all') return sorted
    if (mode === 'range') {
      const from = parseInt(clearRangeFrom)
      const to = parseInt(clearRangeTo)
      if (isNaN(from) || isNaN(to)) return []
      return sorted.filter(f => f >= from && f <= to)
    }
    return []
  }

  // ── Clear Masks ──────────────────────────────────────────────────────────────

  function handleClearFrameMasks() {
    setShowClearMasksModal(true)
  }

  async function handleConfirmClearMasks(mode: ClearMasksMode | 'this_frame') {
    setShowClearMasksModal(false)
    const frames = _framesToSnapshotForClear(mode)
    const snapshots = await _snapshotMasksForFrames(frames)
    const f0 = currentFrame
    const rangeFrom = parseInt(clearRangeFrom)
    const rangeTo = parseInt(clearRangeTo)
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
      useStore.getState().pushHistory({
        labelUndo: 'Clear masks',
        labelRedo: 'Clear masks',
        undo: async () => {
          if (Object.keys(snapshots).length === 0) return
          await restoreMaskFrames(pid, vid, snapshots)
          clearMaskCache()
          setProject(await getProject(pid))
        },
        redo: async () => {
          try {
            if (mode === 'this_frame') {
              await clearFrameMasks(pid, vid, f0)
              setSavedMask(f0, {})
            } else if (mode === 'from_frame') {
              await clearMasksBulk(pid, vid, 'from_frame', f0)
            } else if (mode === 'range' && !isNaN(rangeFrom) && !isNaN(rangeTo)) {
              await clearMasksBulk(pid, vid, 'range', rangeFrom, rangeTo)
            } else if (mode === 'all') {
              await clearMasksBulk(pid, vid, 'all')
            }
          } catch { /* ignore */ }
          clearMaskCache()
          setProject(await getProject(pid))
        },
      })
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

  const manualNAnchors = manualAnchorPrefixRef.current
  const showInferRemainderButton =
    anchorPhase &&
    !anchorRemainderInferencing &&
    anchorFrames.length > manualNAnchors &&
    Array.from({ length: manualNAnchors }, (_, i) => i).every(i => annotatedAnchorIndices.includes(i)) &&
    anchorFrames.some((_, idx) => idx >= manualNAnchors && !annotatedAnchorIndices.includes(idx))

  // ── Object tree rendering ─────────────────────────────────────────────────────

  const objsById = video.objects
  const childrenOfParent = (parentId: string | null) =>
    Object.values(objsById)
      .filter(o => (o.parent_id ?? null) === parentId)
      .sort((a, b) => Number(a.id) - Number(b.id))

  const subObjectForm = (parentId: string) => (
    <div className="rounded-lg border border-[#333] bg-[#141414] p-2.5 ml-3.5" onClick={e => e.stopPropagation()}>
      <p className="text-[10px] text-[#666] mb-1.5 uppercase tracking-wide">New sub-object</p>
      <input
        type="text"
        value={subName}
        onChange={e => setSubName(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter') handleAddSubObject()
          if (e.key === 'Escape') setSubParentId(null)
        }}
        placeholder="e.g. head, left-ear, snout..."
        className="w-full mb-2 text-xs"
        autoFocus
      />
      <div className="flex gap-1.5 mb-2">
        {(['segmentation', 'point'] as ObjectKind[]).map(k => (
          <button
            key={k}
            onClick={() => setSubKind(k)}
            className={`flex-1 text-[11px] py-1 rounded-md border transition-colors
              ${subKind === k
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'border-[#333] text-[#888] hover:border-[#555]'}`}
          >
            {k === 'segmentation' ? 'Segmentation' : 'Point'}
          </button>
        ))}
      </div>
      {subKind === 'point' && (
        <label className="block text-[10px] text-[#666] mb-2">
          Blob size: {(subBlobFrac * 100).toFixed(0)}% of parent
          <input
            type="range" min={0.02} max={0.25} step={0.01}
            value={subBlobFrac}
            onChange={e => setSubBlobFrac(parseFloat(e.target.value))}
            className="w-full mt-1"
          />
        </label>
      )}
      <div className="flex gap-2">
        <button
          onClick={() => setSubParentId(null)}
          className="btn btn-secondary flex-1 text-xs py-1"
        >
          Cancel
        </button>
        <button
          onClick={handleAddSubObject}
          disabled={!subName.trim()}
          className="btn btn-primary flex-1 text-xs py-1"
        >
          Add
        </button>
      </div>
    </div>
  )

  const renderObjectNode = (obj: typeof objsById[string], depth: number): React.ReactNode => (
    <React.Fragment key={obj.id}>
      <ObjectCard
        objId={obj.id}
        name={obj.name}
        color={obj.color}
        isActive={currentObjectId === obj.id}
        onSelect={() => setCurrentObject(currentObjectId === obj.id ? null : obj.id)}
        description={obj.description}
        depth={depth}
        kind={obj.kind ?? 'segmentation'}
        onAddSub={!isTracking && !anchorPhase ? () => openSubForm(obj.id) : undefined}
      />
      {subParentId === obj.id && subObjectForm(obj.id)}
      {childrenOfParent(obj.id).map(child => renderObjectNode(child, depth + 1))}
    </React.Fragment>
  )

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
          {anchorRemainderInferencing ? (
            <>
              <p className="text-xs text-violet-300 font-medium">Automatic anchor inference</p>
              <p className="text-xs text-[#666] mt-0.5">
                Propagating masks from your manual prefix to each remaining anchor frame. You can keep this tab open.
              </p>
              {anchorRemainderAwaitingCommit && (
                <p className="text-xs text-amber-200/90 mt-1">
                  Editing this predicted anchor — adjust masks if needed, then click Done to continue.
                </p>
              )}
            </>
          ) : (
            <>
              <p className="text-xs text-blue-300 font-medium">
                Anchor Frame {currentAnchorIndex + 1} / {anchorFrames.length}
                <span className="text-blue-400/60 ml-1.5 font-normal">(frame #{anchorFrames[currentAnchorIndex]})</span>
              </p>
              <p className="text-xs text-[#666] mt-0.5">
                Annotate objects on this frame, then click “Done, next”, or press Enter (Shift+Enter = previous anchor).
                {anchorFrames.length > manualAnchorPrefixRef.current ? (
                  <>
                    {' '}
                    After anchor {manualAnchorPrefixRef.current},{' '}
                    {config.autoInferAnchorRemainder
                      ? 'remaining anchors run automatically.'
                      : 'label the rest manually or run inference below / enable auto in Settings → Tracking.'}
                  </>
                ) : null}
              </p>
              {offQueueAnchor && (
                <button
                  type="button"
                  onClick={handleGoToQueueAnchor}
                  className="mt-2 w-full text-xs py-1.5 px-2 rounded-md border border-blue-600/50 bg-blue-600/20 hover:bg-blue-600/30 text-blue-200 font-medium transition-colors"
                >
                  Go to next in queue — anchor {queueAnchorIndex + 1} (frame #{anchorFrames[queueAnchorIndex]})
                </button>
              )}
              {showInferRemainderButton && (
                <button
                  type="button"
                  onClick={() => void runAnchorRemainderPrediction()}
                  className="mt-2 w-full text-xs py-1.5 px-2 rounded-md bg-violet-600/80 hover:bg-violet-600 text-white font-medium transition-colors"
                >
                  Infer remaining anchors (SAM)
                </button>
              )}
            </>
          )}
        </div>
      )}

      {/* All anchors labeled banner */}
      {!anchorPhase && allAnchorsLabeled && propagationStatus === 'idle' && (
        <div className="mx-3 mt-2 mb-1 rounded-lg border border-emerald-700/40 bg-emerald-500/10 px-3 py-2 flex-shrink-0">
          <p className="text-xs text-emerald-300 font-medium">All anchor frames labeled</p>
          <p className="text-xs text-[#666] mt-0.5">Click "Start Tracking" to run bidirectional propagation.</p>
        </div>
      )}

      {/* Objects list (hierarchical: top-level objects + nested sub-objects) */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {childrenOfParent(null).map(obj => renderObjectNode(obj, 0))}

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

      {/* Predict current frame (preview all objects from the annotated state) */}
      {!isTracking && hasObjects && (
        <button
          type="button"
          onClick={() => void handlePredictFrame()}
          disabled={predicting}
          className="mx-3 mb-2 flex items-center justify-center gap-2 py-2 rounded-lg border border-[#333] bg-[#161616] text-sm text-[#ccc] hover:border-blue-500/50 hover:text-blue-300 disabled:opacity-50 transition-colors flex-shrink-0"
          title={`Predict masks for all objects on frame #${currentFrame} without saving`}
        >
          {predicting ? <Loader size={14} className="animate-spin" /> : <Sparkles size={14} />}
          {predicting ? 'Predicting…' : `Predict frame #${currentFrame}`}
        </button>
      )}

      {anchorRemainderInferencing && (
        <div className="mx-3 mb-2 flex items-center gap-2 text-xs text-violet-300">
          <Loader size={12} className="animate-spin" />
          Running automatic anchor inference (sequential SAM propagation)…
        </div>
      )}

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

      {/* Bottom buttons — two rows so controls are not squeezed in one line */}
      <div className="flex flex-col gap-1 px-1.5 py-1 border-t border-[#2a2a2a] flex-shrink-0 min-w-0">
        <div
          className="flex items-stretch gap-1 w-full min-w-0"
          style={{ containerType: 'inline-size' } as React.CSSProperties}
        >
          <button
            type="button"
            onClick={() => void undoLast()}
            disabled={!video || undoStack.length === 0 || isTracking || historyBusy}
            className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 disabled:opacity-40"
            style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
            title="Undo"
          >
            <Undo2 size={10} />
            <span>Undo</span>
          </button>
          <button
            type="button"
            onClick={() => void redoLast()}
            disabled={!video || redoStack.length === 0 || isTracking || historyBusy}
            className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 disabled:opacity-40"
            style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
            title="Redo"
          >
            <Redo2 size={10} />
            <span>Redo</span>
          </button>

          <button
            onClick={handleStartOver}
            disabled={isTracking}
            className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2"
            style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
          >
            <RotateCcw size={10} />
            <span>Start</span>
            <span>over</span>
          </button>

          {(video.propagated_frames?.length ?? 0) > 0 && (
            <button
              onClick={handleExport}
              disabled={exportStatus === 'running' || isTracking}
              className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 disabled:opacity-40"
              style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
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
              className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 disabled:opacity-40 text-red-400 hover:text-red-300"
              style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
              title={`Clear saved masks for frame ${currentFrame}`}
            >
              <Trash2 size={10} />
              <span>Clear</span>
              <span>masks</span>
            </button>
          )}

          {objects.length >= 2 && !isTracking && (
            <button
              onClick={() => {
                setSwapObjA(objects[0]?.id ?? '')
                setSwapObjB(objects[1]?.id ?? '')
                setSwapMaskScope('current')
                setShowSwapModal(true)
              }}
              disabled={anchorRemainderInferencing && !anchorRemainderAwaitingCommit}
              className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 disabled:opacity-40"
              style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
              title="Swap two objects' masks (saved .npz on disk for selected frame range)"
            >
              <ArrowLeftRight size={10} />
              <span>Swap</span>
              <span>masks</span>
            </button>
          )}
        </div>

        <div
          className="flex items-stretch gap-1 w-full min-w-0"
          style={{ containerType: 'inline-size' } as React.CSSProperties}
        >
          {isTracking ? (
            <button
              onClick={handlePause}
              className="btn btn-ghost flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 text-amber-400 hover:text-amber-300"
              style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
            >
              <Pause size={10} />
              <span>Pause</span>
            </button>
          ) : anchorPhase ? (
            <button
              type="button"
              onClick={handleCommitAnchor}
              disabled={!hasObjects || (anchorRemainderInferencing && !anchorRemainderAwaitingCommit)}
              title="Keyboard: Enter (when objects exist)"
              className="btn btn-primary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 disabled:opacity-40"
              style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
            >
              <ChevronDown size={10} />
              <span>Done,</span>
              <span>next</span>
            </button>
          ) : allAnchorsLabeled ? (
            <button
              onClick={handleStartTracking}
              disabled={!hasObjects}
              className="btn btn-primary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 disabled:opacity-40"
              style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
            >
              <ChevronRight size={10} />
              <span>Start</span>
              <span>tracking</span>
            </button>
          ) : (
            <>
              <button
                onClick={handleAnnotateAnchorsClick}
                disabled={!hasObjects || initializingSession || isPaused}
                className="btn btn-secondary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 disabled:opacity-40"
                style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
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
                className="btn btn-primary flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-1 py-2 disabled:opacity-40"
                style={{ fontSize: 'clamp(8px, 5cqi, 12px)', lineHeight: 1.2 }}
              >
                <ChevronRight size={10} />
                <span>Track</span>
                <span>frames</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Anchor remainder: SAM work or fetching masks before review chip */}
      {anchorRemainderInferencing && !anchorReviewPrompt && (
        <div className="fixed z-[65] bottom-6 right-6 flex items-center gap-2 rounded-full bg-black/75 border border-[#444] px-3 py-2 text-xs text-[#ddd] shadow-lg pointer-events-none max-w-[min(92vw,320px)]">
          <Loader size={14} className="animate-spin flex-shrink-0 text-violet-300" />
          <span className="truncate">
            {anchorReviewLoading ? 'Loading predicted masks…' : (anchorRemainderProgressText || 'Working…')}
          </span>
        </div>
      )}

      {/* Draggable Good / Edit chip (bottom-right by default) */}
      {anchorReviewPrompt && (
        <div
          role="dialog"
          aria-label="Predicted anchor review"
          className="fixed z-[70] w-[min(92vw,288px)] rounded-xl border border-[#444] bg-[#141414]/95 shadow-2xl backdrop-blur-sm overflow-hidden"
          style={{ right: anchorReviewPanelOffsets.right, bottom: anchorReviewPanelOffsets.bottom }}
        >
          <div
            className="flex items-center gap-2 px-3 py-2 border-b border-[#333] bg-[#1a1a1a]/90 cursor-grab active:cursor-grabbing select-none"
            onMouseDown={handleAnchorReviewDragStart}
          >
            <GripVertical size={14} className="text-[#666] flex-shrink-0" aria-hidden />
            <span className="text-xs font-medium text-[#ccc] truncate">
              Anchor {anchorReviewPrompt.anchorIndex + 1}/{anchorFrames.length} · frame {anchorReviewPrompt.frameIdx}
            </span>
          </div>
          <div className="p-3 space-y-3">
            <p className="text-[11px] text-[#888] leading-relaxed">
              Step {anchorReviewPrompt.anchorsDone}/{anchorReviewPrompt.anchorsQueued}. Accept masks or tap Edit to adjust, then Done.
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={anchorReviewBusy}
                onClick={handleAnchorReviewEdit}
                className="btn btn-secondary flex-1 py-2 text-xs font-medium"
              >
                Edit…
              </button>
              <button
                type="button"
                disabled={anchorReviewBusy}
                onClick={() => void handleAnchorReviewGood()}
                className="btn btn-primary flex-1 py-2 text-xs font-medium"
              >
                {anchorReviewBusy ? '…' : 'Good'}
              </button>
            </div>
          </div>
        </div>
      )}

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
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-[min(28rem,calc(100vw-1.5rem))] shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-1">Swap object masks</h3>
            <p className="text-xs text-[#666] mb-4 leading-relaxed">
              Swap saved mask assignments between two objects for the scope you choose (only frames that already have mask files are updated).
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
                <span className="block text-[10px] text-[#666] mb-2">Apply swap to</span>
                <div className="flex flex-col gap-2">
                  <label className="flex items-start gap-2 text-xs text-[#ccc] cursor-pointer">
                    <input
                      type="radio"
                      name="swapMaskScope"
                      checked={swapMaskScope === 'current'}
                      onChange={() => setSwapMaskScope('current')}
                      className="mt-0.5 accent-violet-500"
                    />
                    <span>
                      Current frame only{' '}
                      <span className="text-[#666]">(frame {currentFrame})</span>
                    </span>
                  </label>
                  <label className="flex items-start gap-2 text-xs text-[#ccc] cursor-pointer">
                    <input
                      type="radio"
                      name="swapMaskScope"
                      checked={swapMaskScope === 'from_current'}
                      onChange={() => setSwapMaskScope('from_current')}
                      className="mt-0.5 accent-violet-500"
                    />
                    <span>
                      This frame and all later frames{' '}
                      <span className="text-[#666]">
                        ({currentFrame}
                        {video != null ? `–${video.num_frames - 1}` : ''})
                      </span>
                    </span>
                  </label>
                  <label className="flex items-start gap-2 text-xs text-[#ccc] cursor-pointer">
                    <input
                      type="radio"
                      name="swapMaskScope"
                      checked={swapMaskScope === 'all'}
                      onChange={() => setSwapMaskScope('all')}
                      className="mt-0.5 accent-violet-500"
                    />
                    <span>Entire video (all frames that have masks)</span>
                  </label>
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

      {/* Anchor frame sampling modal (before first anchor labeling on a video) */}
      {showAnchorSamplingModal && video && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] p-5 w-[22rem] max-w-[calc(100vw-2rem)] shadow-xl">
            <h3 className="text-sm font-semibold text-[#eee] mb-1">Anchor frame sampling</h3>
            <p className="text-xs text-[#666] mb-4 leading-relaxed">
              Choose how often to place anchor frames across this video (from the start frame through
              the end). You can change this later in Settings only before any anchor is labeled.
            </p>

            <label htmlFor="anchor-sampling-interval" className="text-xs text-[#aaa] font-medium block mb-1.5">
              Frame interval (frames)
            </label>
            <div className="flex items-center gap-3 mb-3">
              <NumericDraftInput
                id="anchor-sampling-interval"
                value={anchorSamplingInterval}
                onChange={setAnchorSamplingInterval}
                onLiveChange={setAnchorSamplingLive}
                min={ANCHOR_BATCH_SIZE_MIN}
                max={ANCHOR_BATCH_SIZE_MAX}
                autoFocus
                className="w-28 text-sm py-2 px-3 rounded-lg bg-[#111] border border-[#333] text-[#eee] text-right"
              />
              <span className="text-xs text-[#555]">min {ANCHOR_BATCH_SIZE_MIN} · max {ANCHOR_BATCH_SIZE_MAX}</span>
            </div>

            <p className="text-xs text-blue-200/90 mb-5 rounded-lg border border-blue-700/30 bg-blue-500/10 px-3 py-2 leading-relaxed">
              {anchorSamplingPreviewCount != null ? (
                <>
                  This video will have{' '}
                  <span className="font-semibold text-blue-100">{anchorSamplingPreviewCount}</span>
                  {' '}anchor frame{anchorSamplingPreviewCount === 1 ? '' : 's'} to label.
                </>
              ) : (
                'Anchor count updates as you change the interval.'
              )}
            </p>

            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setShowAnchorSamplingModal(false)}
                disabled={anchorSamplingSaving}
                className="btn btn-ghost flex-1 py-2 text-xs"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void confirmAnchorSamplingAndStart()}
                disabled={anchorSamplingSaving}
                className="btn btn-primary flex-1 py-2 text-xs font-medium disabled:opacity-50"
              >
                {anchorSamplingSaving ? 'Starting…' : 'Start labeling'}
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
