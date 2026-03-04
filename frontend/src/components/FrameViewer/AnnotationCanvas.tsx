import React, { useCallback, useEffect, useRef } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { extractFrame, addPoints, getSavedMask } from '../../api/client'
import { drawMasks, drawPoints } from '../../utils/maskUtils'

interface Props {
  width: number
  height: number
  videoRef: React.RefObject<HTMLVideoElement>
}

export default function AnnotationCanvas({ width, height, videoRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const store = useStore()
  const video = selectCurrentVideo(store)

  const {
    project, currentVideoId,
    currentFrame, setCurrentFrame, currentObjectId, pointMode,
    localAnnotations, addLocalPoint,
    currentFrameMasks, currentFrameMasksFrame, setCurrentFrameMasks,
    savedMaskCache, setSavedMask,
    propagationStatus,
    config,
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

    // Determine which masks to show.
    // Live annotation masks (from a recent SAM inference click) always take
    // priority over saved propagation masks — the user just re-annotated the
    // frame and the new result should be immediately visible.
    let masksToShow: typeof currentFrameMasks = {}
    if (currentFrameMasksFrame === currentFrame && Object.keys(currentFrameMasks).length > 0) {
      masksToShow = currentFrameMasks
    } else {
      const saved = savedMaskCache[currentFrame]
      if (saved) masksToShow = saved
    }

    // Collect points for current frame
    const allPoints: { x: number; y: number; label: 0 | 1 }[] = []
    for (const [, framePts] of Object.entries(localAnnotations)) {
      const pts = framePts[String(currentFrame)]
      if (pts) {
        for (const p of pts.points) {
          allPoints.push(p)
        }
      }
    }

    // Draw masks first (async), then points on top so they are always visible.
    // The stale flag prevents a superseded async draw from clobbering a newer
    // render that already ran its cleanup.
    let stale = false
    const masksToDraw = config.showMasks ? masksToShow : {}
    drawMasks(ctx, masksToDraw, width, height, config.maskOpacity).then(() => {
      if (stale) return
      drawPoints(ctx, allPoints, width, height, config.pointSize)
    })
    return () => { stale = true }
  }, [width, height, currentFrameMasks, currentFrameMasksFrame, localAnnotations, currentFrame, savedMaskCache, config.showMasks, config.maskOpacity, config.pointSize])

  // ── Load saved masks when frame changes ───────────────────────────────────

  useEffect(() => {
    if (!savedMaskCache[currentFrame]) {
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

    // Compute the true frame number from the video element's actual currentTime
    // at the moment of click, to avoid drift between store.currentFrame and what
    // the user sees on screen.
    const videoEl = videoRef.current
    const fps = video.fps || 30
    const trueFrame = videoEl
      ? Math.floor(videoEl.currentTime * fps)
      : currentFrame

    // Sync the store so the HUD and mask rendering use the same frame
    if (trueFrame !== currentFrame) {
      setCurrentFrame(trueFrame)
    }

    const rect = canvas.getBoundingClientRect()
    const scaleX = width / rect.width
    const scaleY = height / rect.height
    const px = (e.clientX - rect.left) * scaleX
    const py = (e.clientY - rect.top) * scaleY
    const nx = px / width
    const ny = py / height
    const label: 0 | 1 = pointMode === 'add' ? 1 : 0

    // Add to local state (use trueFrame so points line up)
    addLocalPoint(currentObjectId, trueFrame, nx, ny, label)

    // Get all accumulated points for this object on this frame
    const framePts = useStore.getState().localAnnotations[currentObjectId]?.[String(trueFrame)]
    const allPoints: [number, number][] = framePts ? framePts.points.map(p => [p.x, p.y]) : [[nx, ny]]
    const allLabels = framePts ? framePts.points.map(p => p.label as number) : [label]

    try {
      // Extract this single frame on the backend (into annotated_frames/)
      // so the SAM session can be initialized with just this frame.
      await extractFrame(pid, vid, trueFrame)

      const result = await addPoints(pid, vid, currentObjectId, trueFrame, allPoints, allLabels)
      if (result.masks) {
        // Read latest state after async call — only merge masks from the same frame
        const state = useStore.getState()
        const prevLive = state.currentFrameMasksFrame === trueFrame ? state.currentFrameMasks : {}
        const newLive = { ...prevLive, ...result.masks }
        setCurrentFrameMasks(newLive, trueFrame)
        // Keep savedMaskCache in sync: merge new masks over the existing saved
        // ones so that scrubbing away and back shows the updated result.
        const existingSaved = state.savedMaskCache[trueFrame] ?? {}
        setSavedMask(trueFrame, { ...existingSaved, ...result.masks })
      }
    } catch (err) {
      console.error('Failed to add point:', err)
    }
  }, [pointMode, currentObjectId, video, currentFrame, width, height, pid, vid, addLocalPoint, setCurrentFrameMasks, setCurrentFrame, videoRef])

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
