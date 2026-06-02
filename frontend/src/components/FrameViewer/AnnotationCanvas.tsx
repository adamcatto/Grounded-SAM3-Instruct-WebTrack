import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { extractFrame, addPoints, getSavedMask, rebuildFromConfig, replaceFramePromptsData, removeObject } from '../../api/client'
import { drawMasks, drawPoints, loadMaskBitmap } from '../../utils/maskUtils'
import { applyRebuildMasksToStore, localAnnotationsToPointPrompts, stripObjectMaskFromFrameCaches } from '../../history/applyRebuild'

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
    config,
    anchorPhase,
    addToast,
    objectVisibility, isObjectVisible,
  } = store

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

    // Respect per-object visibility (cascade-aware: hidden if any ancestor is hidden).
    const masksToDraw = config.showMasks
      ? Object.fromEntries(Object.entries(masksToShow).filter(([oid]) => isObjectVisible(oid)))
      : {}

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
  }, [width, height, masksToShow, localAnnotations, currentFrame, config.showMasks, config.maskOpacity, config.pointSize, objectNames, objectVisibility])

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
  }, [currentFrame, propagationStatus, pid, vid, savedMaskCache])

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

    const oid = currentObjectId
    const key = String(frameToUse)
    const beforeSlot = useStore.getState().localAnnotations[oid]?.[key]
    const beforePts: [number, number][] = beforeSlot ? beforeSlot.points.map(p => [p.x, p.y]) : []
    const beforeLabs: number[] = beforeSlot ? beforeSlot.points.map(p => p.label as number) : []

    // Add to local state
    addLocalPoint(oid, frameToUse, nx, ny, label)

    // Get all accumulated points for this object on this frame
    const framePts = useStore.getState().localAnnotations[oid]?.[key]
    const allPoints: [number, number][] = framePts ? framePts.points.map(p => [p.x, p.y]) : [[nx, ny]]
    const allLabels = framePts ? framePts.points.map(p => p.label as number) : [label]
    const afterPts = [...allPoints]
    const afterLabs = [...allLabels]

    try {
      // Extract this single frame on the backend (into annotated_frames/)
      // so the SAM session can be initialized with just this frame.
      await extractFrame(pid, vid, frameToUse)

      const result = await addPoints(pid, vid, oid, frameToUse, allPoints, allLabels, anchorPhase)
      const addedObjects = result.new_objects ?? []

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
        if (addedObjects.length > 0) {
          const currentObjs = state.project?.videos[vid]?.objects ?? {}
          const updatedObjs = { ...currentObjs }
          for (const newObj of addedObjects) {
            updatedObjs[newObj.id] = newObj
          }
          useStore.getState().updateVideo({ objects: updatedObjs })
        }
      }

      useStore.getState().pushHistory({
        labelUndo: 'Point prompt',
        labelRedo: 'Point prompt',
        undo: async () => {
          if (useStore.getState().propagationStatus === 'running') return
          await replaceFramePromptsData(pid, vid, oid, frameToUse, beforePts, beforeLabs)
          for (const o of addedObjects) {
            try {
              await removeObject(pid, vid, o.id)
            } catch { /* ignore */ }
          }
          const la = JSON.parse(JSON.stringify(useStore.getState().localAnnotations)) as Record<string, Record<string, { points: { x: number; y: number; label: 0 | 1 }[] }>>
          if (beforePts.length === 0) {
            const of = la[oid]
            if (of) {
              delete of[key]
              if (Object.keys(of).length === 0) delete la[oid]
            }
          } else {
            la[oid] = { ...la[oid], [key]: { points: beforePts.map(([x, y], i) => ({ x, y, label: beforeLabs[i] as 0 | 1 })) } }
          }
          useStore.setState({ localAnnotations: la })
          const vidDatUndo = useStore.getState().project?.videos[vid]
          if (vidDatUndo) {
            const objs = { ...vidDatUndo.objects }
            for (const o of addedObjects) delete objs[o.id]
            useStore.getState().updateVideo({
              objects: objs,
              point_prompts: localAnnotationsToPointPrompts(la),
            })
          }
          const anchorNow = useStore.getState().anchorPhase
          const rb = await rebuildFromConfig(pid, vid, [frameToUse], anchorNow, anchorNow ? frameToUse : null)
          applyRebuildMasksToStore(rb.masks_by_frame)
          if (!la[oid]?.[key]) {
            stripObjectMaskFromFrameCaches(oid, frameToUse)
          }
        },
        redo: async () => {
          const st = useStore.getState()
          if (st.propagationStatus === 'running') return
          const la = JSON.parse(JSON.stringify(st.localAnnotations)) as typeof st.localAnnotations
          la[oid] = { ...(la[oid] ?? {}), [key]: { points: afterPts.map(([x, y], i) => ({ x, y, label: afterLabs[i] as 0 | 1 })) } }
          useStore.setState({ localAnnotations: la })
          const vidDat = st.project?.videos[vid]
          if (vidDat) {
            const objs = { ...vidDat.objects }
            for (const o of addedObjects) objs[o.id] = o
            useStore.getState().updateVideo({
              objects: objs,
              point_prompts: localAnnotationsToPointPrompts(la),
            })
          }
          await extractFrame(pid, vid, frameToUse)
          const anchorNow = useStore.getState().anchorPhase
          const res = await addPoints(pid, vid, oid, frameToUse, afterPts, afterLabs, anchorNow)
          if (res.masks) {
            const s2 = useStore.getState()
            const prevLive = s2.currentFrameMasksFrame === frameToUse ? s2.currentFrameMasks : {}
            s2.setCurrentFrameMasks({ ...prevLive, ...res.masks }, frameToUse)
            const existingSaved = s2.savedMaskCache[frameToUse] ?? {}
            s2.setSavedMask(frameToUse, { ...existingSaved, ...res.masks })
          }
          if (res.new_objects?.length) {
            const s2 = useStore.getState()
            const curObjs = s2.project?.videos[vid]?.objects ?? {}
            const uo = { ...curObjs }
            for (const n of res.new_objects) uo[n.id] = n
            s2.updateVideo({ objects: uo })
          }
        },
      })
    } catch (err: unknown) {
      console.error('Failed to add point:', err)
      // Extract the backend's detail message if available (e.g. 409 anchor-frame guard)
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail ?? 'Failed to add point', 'error')
    }
  }, [pointMode, currentObjectId, video, currentFrame, width, height, pid, vid, addLocalPoint, setCurrentFrameMasks, setSavedMask, anchorPhase, addToast])

  const cursor = pointMode ? 'crosshair' : 'default'

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
        }}
      />
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
