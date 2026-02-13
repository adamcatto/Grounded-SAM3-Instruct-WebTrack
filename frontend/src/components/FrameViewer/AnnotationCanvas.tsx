import React, { useCallback, useEffect, useRef } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { extractFrame, addPoints, getSavedMask } from '../../api/client'
import { drawMasks, drawPoints } from '../../utils/maskUtils'

interface Props {
  width: number
  height: number
}

export default function AnnotationCanvas({ width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const store = useStore()
  const video = selectCurrentVideo(store)

  const {
    project, currentVideoId,
    currentFrame, currentObjectId, pointMode,
    localAnnotations, addLocalPoint,
    currentFrameMasks, currentFrameMasksFrame, setCurrentFrameMasks,
    savedMaskCache, setSavedMask,
    propagationStatus,
  } = store

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  // ── Render loop: draw masks + points onto canvas ──────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = width
    canvas.height = height
    ctx.clearRect(0, 0, width, height)

    // Determine which masks to show
    let masksToShow: typeof currentFrameMasks = {}
    if (propagationStatus === 'done' || propagationStatus === 'running') {
      const saved = savedMaskCache[currentFrame]
      if (saved) masksToShow = saved
    } else if (currentFrameMasksFrame === currentFrame) {
      // Only show live annotation masks for the frame they were computed on
      masksToShow = currentFrameMasks
    }

    // Draw masks
    drawMasks(ctx, masksToShow, width, height)

    // Draw all points for current frame
    const allPoints: { x: number; y: number; label: 0 | 1 }[] = []
    for (const [objId, framePts] of Object.entries(localAnnotations)) {
      const pts = framePts[String(currentFrame)]
      if (pts) {
        for (const p of pts.points) {
          allPoints.push(p)
        }
      }
    }
    drawPoints(ctx, allPoints, width, height)
  }, [width, height, currentFrameMasks, currentFrameMasksFrame, localAnnotations, currentFrame, savedMaskCache, propagationStatus])

  // ── Load saved masks when frame changes (post-propagation) ────────────────

  useEffect(() => {
    if ((propagationStatus === 'done' || propagationStatus === 'running') && !savedMaskCache[currentFrame]) {
      getSavedMask(pid, vid, currentFrame)
        .then(data => {
          if (data.masks && Object.keys(data.masks).length > 0) {
            setSavedMask(currentFrame, data.masks)
          }
        })
        .catch(() => { /* no mask for this frame yet */ })
    }
  }, [currentFrame, propagationStatus])

  // ── Click to add point ────────────────────────────────────────────────────

  const handleClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!pointMode || !currentObjectId || !video) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const scaleX = width / rect.width
    const scaleY = height / rect.height
    const px = (e.clientX - rect.left) * scaleX
    const py = (e.clientY - rect.top) * scaleY
    const nx = px / width
    const ny = py / height
    const label: 0 | 1 = pointMode === 'add' ? 1 : 0

    // Add to local state
    addLocalPoint(currentObjectId, currentFrame, nx, ny, label)

    // Get all accumulated points for this object on this frame
    const framePts = useStore.getState().localAnnotations[currentObjectId]?.[String(currentFrame)]
    const allPoints: [number, number][] = framePts ? framePts.points.map(p => [p.x, p.y]) : [[nx, ny]]
    const allLabels = framePts ? framePts.points.map(p => p.label as number) : [label]

    try {
      // Extract this single frame on the backend (into annotated_frames/)
      // so the SAM session can be initialized with just this frame.
      await extractFrame(pid, vid, currentFrame)

      const result = await addPoints(pid, vid, currentObjectId, currentFrame, allPoints, allLabels)
      if (result.masks) {
        // Read latest state after async call — only merge masks from the same frame
        const state = useStore.getState()
        const prevMasks = state.currentFrameMasksFrame === currentFrame ? state.currentFrameMasks : {}
        setCurrentFrameMasks({ ...prevMasks, ...result.masks }, currentFrame)
      }
    } catch (err) {
      console.error('Failed to add point:', err)
    }
  }, [pointMode, currentObjectId, video, currentFrame, width, height, pid, vid, addLocalPoint, setCurrentFrameMasks])

  const cursor = pointMode ? 'crosshair' : 'default'

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      onClick={handleClick}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        cursor,
        display: 'block',
      }}
    />
  )
}
