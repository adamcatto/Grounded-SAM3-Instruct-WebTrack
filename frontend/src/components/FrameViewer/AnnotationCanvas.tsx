import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { extractFrame, addPoints, getSavedMask } from '../../api/client'
import { drawMasks, drawPoints, loadMaskBitmap } from '../../utils/maskUtils'

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
    propagationStatus, propagationStartFrame,
    config, uncertaintyData,
  } = store

  const confusionScore = uncertaintyData?.per_frame?.[String(currentFrame)]?.confusion_score ?? 0

  const [hoverLabel, setHoverLabel] = useState<string | null>(null)
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null)
  const maskPixelDataRef = useRef<Map<string, ImageData>>(new Map())

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  // ── Derive masks to show ──────────────────────────────────────────────────

  const masksToShow = useMemo(() => {
    if (currentFrameMasksFrame === currentFrame && Object.keys(currentFrameMasks).length > 0) {
      console.log(`[MasksToShow] Frame ${currentFrame}: using live masks`, Object.keys(currentFrameMasks))
      return currentFrameMasks
    }
    const saved = savedMaskCache[currentFrame] ?? {}
    console.log(`[MasksToShow] Frame ${currentFrame}: using saved masks`, Object.keys(saved))
    return saved
  }, [currentFrame, currentFrameMasks, currentFrameMasksFrame, savedMaskCache])

  // ── Cache per-mask ImageData for hover hit-testing ────────────────────────

  useEffect(() => {
    if (width === 0 || height === 0) return
    const cache = maskPixelDataRef.current
    cache.clear()
    for (const [objId, b64] of Object.entries(masksToShow)) {
      loadMaskBitmap(b64).then(bitmap => {
        const c = document.createElement('canvas')
        c.width = width
        c.height = height
        const ctx2 = c.getContext('2d')
        if (!ctx2) return
        ctx2.drawImage(bitmap, 0, 0, width, height)
        cache.set(objId, ctx2.getImageData(0, 0, width, height))
      })
    }
  }, [masksToShow, width, height])

  // Build object name lookup for mask labels
  const objectNames = useMemo(() => {
    const names: Record<string, string> = {}
    if (video?.objects) {
      for (const [objId, obj] of Object.entries(video.objects)) {
        names[objId] = obj.name || objId
      }
    }
    return names
  }, [video?.objects])

  // ── Render loop: draw masks + points onto canvas ──────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = width
    canvas.height = height
    ctx.clearRect(0, 0, width, height)

    const masksToDraw = config.showMasks ? masksToShow : {}

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
    drawMasks(ctx, masksToDraw, width, height, config.maskOpacity, objectNames, config.showMasks).then(() => {
      if (stale) return
      drawPoints(ctx, allPoints, width, height, config.pointSize)
    })
    return () => { stale = true }
  }, [width, height, masksToShow, localAnnotations, currentFrame, config.showMasks, config.maskOpacity, config.pointSize, objectNames])

  // ── Load saved masks when frame changes ───────────────────────────────────
  // No debounce: start the fetch on the very next event loop tick.
  // The effect cleanup cancels the pending fetch for superseded frames,
  // so rapid arrow-key navigation only fires a request for each frame
  // the user briefly settles on (not every intermediate frame).

  useEffect(() => {
    console.log(`[MaskLoad] Frame ${currentFrame}: checking cache...`, { inCache: !!savedMaskCache[currentFrame], pid, vid })
    if (savedMaskCache[currentFrame]) {
      console.log(`[MaskLoad] Frame ${currentFrame}: already cached`)
      return  // already in memory
    }

    const frame = currentFrame
    const timer = setTimeout(() => {
      console.log(`[MaskLoad] Frame ${frame}: fetching from API...`)
      getSavedMask(pid, vid, frame)
        .then(data => {
          console.log(`[MaskLoad] Frame ${frame}: got response`, { maskCount: Object.keys(data.masks || {}).length })
          if (data.masks && Object.keys(data.masks).length > 0) {
            setSavedMask(frame, data.masks)
          }
        })
        .catch((err) => { 
          console.log(`[MaskLoad] Frame ${frame}: no mask or error`, err)
        })
    }, 0)

    return () => clearTimeout(timer)
  }, [currentFrame, propagationStatus, pid, vid])

  // ── Hover: show mask label tooltip ───────────────────────────────────────

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const px = Math.floor((e.clientX - rect.left) * (width / rect.width))
    const py = Math.floor((e.clientY - rect.top) * (height / rect.height))
    if (px < 0 || py < 0 || px >= width || py >= height) {
      setHoverLabel(null); setHoverPos(null); return
    }
    for (const [objId, imageData] of maskPixelDataRef.current) {
      const alpha = imageData.data[(py * width + px) * 4 + 3]
      if (alpha > 30) {
        setHoverLabel(video?.objects[objId]?.name ?? objId)
        setHoverPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
        return
      }
    }
    setHoverLabel(null); setHoverPos(null)
  }, [video, width, height])

  // ── Click to add point ────────────────────────────────────────────────────

  const handleClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!pointMode || !currentObjectId || !video) return

    const canvas = canvasRef.current
    if (!canvas) return

    // IMPORTANT: When adding point prompts, NEVER change the current frame.
    // Always use the store's currentFrame - that's what the user selected
    // and what they expect the point to be placed on.
    // The frame should never go below propagationStartFrame.
    const frameToUse = currentFrame

    const rect = canvas.getBoundingClientRect()
    const scaleX = width / rect.width
    const scaleY = height / rect.height
    const px = (e.clientX - rect.left) * scaleX
    const py = (e.clientY - rect.top) * scaleY
    const nx = px / width
    const ny = py / height
    const label: 0 | 1 = pointMode === 'add' ? 1 : 0

    // Add to local state
    addLocalPoint(currentObjectId, frameToUse, nx, ny, label)

    // Get all accumulated points for this object on this frame
    const framePts = useStore.getState().localAnnotations[currentObjectId]?.[String(frameToUse)]
    const allPoints: [number, number][] = framePts ? framePts.points.map(p => [p.x, p.y]) : [[nx, ny]]
    const allLabels = framePts ? framePts.points.map(p => p.label as number) : [label]

    try {
      // Extract this single frame on the backend (into annotated_frames/)
      // so the SAM session can be initialized with just this frame.
      await extractFrame(pid, vid, frameToUse)

      const result = await addPoints(pid, vid, currentObjectId, frameToUse, allPoints, allLabels)
      if (result.masks) {
        // Read latest state after async call — only merge masks from the same frame
        const state = useStore.getState()
        const prevLive = state.currentFrameMasksFrame === frameToUse ? state.currentFrameMasks : {}
        const newLive = { ...prevLive, ...result.masks }
        setCurrentFrameMasks(newLive, frameToUse)
        // Keep savedMaskCache in sync: merge new masks over the existing saved
        // ones so that scrubbing away and back shows the updated result.
        const existingSaved = state.savedMaskCache[frameToUse] ?? {}
        setSavedMask(frameToUse, { ...existingSaved, ...result.masks })
        
        // If new instance objects were created (multi-instance detection), merge them into state
        if (result.new_objects && result.new_objects.length > 0) {
          const currentObjs = state.project?.videos[vid]?.objects ?? {}
          const updatedObjs = { ...currentObjs }
          for (const newObj of result.new_objects) {
            updatedObjs[newObj.id] = newObj
          }
          useStore.getState().updateVideo({ objects: updatedObjs })
        }
        
        // If uncertainty was updated during annotation, merge it into store
        if (result.uncertainty_update) {
          useStore.getState().setUncertaintyData(result.uncertainty_update)
        }
      }
    } catch (err) {
      console.error('Failed to add point:', err)
    }
  }, [pointMode, currentObjectId, video, currentFrame, width, height, pid, vid, addLocalPoint, setCurrentFrameMasks, propagationStartFrame])

  const cursor = pointMode ? 'crosshair' : 'default'

  const uncertaintyBorderColor =
    confusionScore >= 0.7 ? 'rgba(239,68,68,0.8)' :
    confusionScore >= 0.4 ? 'rgba(245,158,11,0.7)' :
    null

  return (
    <>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => { setHoverLabel(null); setHoverPos(null) }}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          cursor,
          display: 'block',
          boxSizing: 'border-box',
          ...(uncertaintyBorderColor ? { boxShadow: `inset 0 0 0 3px ${uncertaintyBorderColor}` } : {}),
        }}
      />
      {uncertaintyBorderColor && (
        <div
          style={{
            position: 'absolute',
            top: 8,
            right: 8,
            pointerEvents: 'none',
            zIndex: 10,
            color: confusionScore >= 0.7 ? '#ef4444' : '#f59e0b',
            background: 'rgba(0,0,0,0.7)',
            borderRadius: '0.375rem',
            padding: '2px 8px',
            fontSize: '11px',
          }}
        >
          Identity uncertainty detected
        </div>
      )}
      {hoverLabel && hoverPos && (
        <div
          style={{
            position: 'absolute',
            left: hoverPos.x + 14,
            top: hoverPos.y - 28,
            pointerEvents: 'none',
            zIndex: 10,
          }}
          className="bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap"
        >
          {hoverLabel}
        </div>
      )}
    </>
  )
}
