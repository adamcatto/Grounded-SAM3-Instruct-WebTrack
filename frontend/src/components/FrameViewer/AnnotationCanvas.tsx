import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { extractFrame, addPoints, addRegistrationPoints, rebuildFromConfig, replaceFramePromptsData, removeObject } from '../../api/client'
import { drawCompositeMask, drawMasks, drawPoints, loadMaskBitmap } from '../../utils/maskUtils'
import { getCompositeBitmap } from '../../utils/compositeMaskCache'
import { loadDisplayBitmap } from '../../utils/maskLoader'
import { applyRebuildMasksToStore, localAnnotationsToPointPrompts, stripObjectMaskFromFrameCaches } from '../../history/applyRebuild'

interface Props {
  width: number
  height: number
  videoRef: React.RefObject<HTMLVideoElement>
  scrubbing?: boolean
}

export default function AnnotationCanvas({ width, height, scrubbing = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const store = useStore()
  const video = selectCurrentVideo(store)

  const {
    project, currentVideoId,
    currentFrame, currentObjectId, pointMode,
    localAnnotations, addLocalPoint,
    currentFrameMasks, currentFrameMasksFrame, setCurrentFrameMasks,
    savedMaskCache, setSavedMask,
    config,
    anchorPhase,
    registrationMode,
    registrationTool,
    registrationPolygonPoints,
    addRegistrationPolygonPoint,
    registrationActiveEdge,
    registrationEdgePoints,
    addRegistrationEdgePoint,
    addToast,
  } = store

  const [hoverLabel, setHoverLabel] = useState<string | null>(null)
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null)
  const [displayReady, setDisplayReady] = useState(0)
  const maskPixelDataRef = useRef<Map<string, ImageData>>(new Map())

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  const masksToShow = useMemo(() => {
    if (currentFrameMasksFrame === currentFrame && Object.keys(currentFrameMasks).length > 0) {
      return currentFrameMasks
    }
    return savedMaskCache[currentFrame] ?? {}
  }, [currentFrame, currentFrameMasks, currentFrameMasksFrame, savedMaskCache])

  // Load pre-materialized display WebP when available (optional fast path).
  useEffect(() => {
    if (!config.showMasks || !pid || !vid || registrationMode) return
    if (getCompositeBitmap(pid, vid, currentFrame)) {
      setDisplayReady(t => t + 1)
      return
    }
    let cancelled = false
    void loadDisplayBitmap(pid, vid, currentFrame).then(() => {
      if (!cancelled) setDisplayReady(t => t + 1)
    })
    return () => { cancelled = true }
  }, [currentFrame, pid, vid, config.showMasks, registrationMode])

  // Per-object alpha maps for hover hit-testing (disabled while scrubbing).
  useEffect(() => {
    if (scrubbing || width === 0 || height === 0) return
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
  }, [masksToShow, width, height, scrubbing])

  const objectNames = useMemo(() => {
    const names: Record<string, string> = {}
    if (video?.objects) {
      for (const [objId, obj] of Object.entries(video.objects)) {
        names[objId] = obj.name || objId
      }
    }
    if (registrationMode) names.__registration_floor__ = 'Common floor'
    return names
  }, [video?.objects, registrationMode])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = width
    canvas.height = height
    ctx.clearRect(0, 0, width, height)

    let stale = false

    const finishPoints = () => {
      if (stale) return
      const allPoints: { x: number; y: number; label: 0 | 1 }[] = []
      for (const [, framePts] of Object.entries(localAnnotations)) {
        const pts = framePts[String(currentFrame)]
        if (pts) {
          for (const p of pts.points) {
            allPoints.push(p)
          }
        }
      }
      drawPoints(ctx, allPoints, width, height, config.pointSize)
      if (registrationMode && registrationPolygonPoints.length > 0) {
        ctx.save()
        ctx.strokeStyle = '#facc15'
        ctx.fillStyle = '#facc15'
        ctx.lineWidth = Math.max(2, Math.min(width, height) / 300)
        ctx.beginPath()
        registrationPolygonPoints.forEach(([x, y], i) => {
          const px = x * width
          const py = y * height
          if (i === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        })
        if (registrationPolygonPoints.length >= 3) ctx.closePath()
        ctx.stroke()
        for (const [x, y] of registrationPolygonPoints) {
          ctx.beginPath()
          ctx.arc(x * width, y * height, Math.max(3, ctx.lineWidth * 1.5), 0, Math.PI * 2)
          ctx.fill()
        }
        ctx.restore()
      }
      if (registrationMode && registrationTool === 'edges') {
        const colors = { top: '#facc15', right: '#22d3ee', bottom: '#f472b6', left: '#a3e635' }
        for (const edge of ['top', 'right', 'bottom', 'left'] as const) {
          const points = registrationEdgePoints[edge]
          if (points.length === 0) continue
          ctx.save()
          ctx.strokeStyle = colors[edge]
          ctx.fillStyle = colors[edge]
          ctx.globalAlpha = edge === registrationActiveEdge ? 1 : 0.65
          ctx.lineWidth = Math.max(2, Math.min(width, height) / 350)
          ctx.beginPath()
          points.forEach(([x, y], i) => {
            if (i === 0) ctx.moveTo(x * width, y * height)
            else ctx.lineTo(x * width, y * height)
          })
          ctx.stroke()
          for (const [x, y] of points) {
            ctx.beginPath()
            ctx.arc(x * width, y * height, Math.max(3, ctx.lineWidth * 1.5), 0, Math.PI * 2)
            ctx.fill()
          }
          ctx.restore()
        }
      }
    }

    const displayBitmap = config.showMasks && !registrationMode && pid && vid
      ? getCompositeBitmap(pid, vid, currentFrame)
      : undefined

    if (displayBitmap) {
      drawCompositeMask(ctx, displayBitmap, width, height, config.maskOpacity)
      finishPoints()
      return () => { stale = true }
    }

    const masksToDraw = config.showMasks ? masksToShow : {}
    const showLabels = config.showMasks && !scrubbing
    drawMasks(ctx, masksToDraw, width, height, config.maskOpacity, objectNames, showLabels).then(finishPoints)
    return () => { stale = true }
  }, [
    width, height, masksToShow, localAnnotations, currentFrame,
    config.showMasks, config.maskOpacity, config.pointSize, objectNames,
    scrubbing, pid, vid, displayReady, registrationMode, registrationTool,
    registrationPolygonPoints, registrationActiveEdge, registrationEdgePoints,
  ])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (scrubbing) {
      setHoverLabel(null)
      setHoverPos(null)
      return
    }
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
  }, [video, width, height, scrubbing])

  const handleClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!pointMode || !currentObjectId || !video) return

    const canvas = canvasRef.current
    if (!canvas) return

    const frameToUse = currentFrame

    const rect = canvas.getBoundingClientRect()
    const scaleX = width / rect.width
    const scaleY = height / rect.height
    const px = (e.clientX - rect.left) * scaleX
    const py = (e.clientY - rect.top) * scaleY
    // Browser layout/subpixel rounding can put border clicks a fraction outside
    // the image. Clamp normalized coordinates so edges are exactly 0 or 1.
    const nx = Math.max(0, Math.min(1, px / width))
    const ny = Math.max(0, Math.min(1, py / height))
    if (registrationMode && registrationTool === 'polygon') {
      addRegistrationPolygonPoint(nx, ny)
      return
    }
    if (registrationMode && registrationTool === 'edges') {
      addRegistrationEdgePoint(registrationActiveEdge, nx, ny)
      return
    }
    const label: 0 | 1 = pointMode === 'add' ? 1 : 0

    const oid = currentObjectId
    const key = String(frameToUse)
    const beforeSlot = useStore.getState().localAnnotations[oid]?.[key]
    const beforePts: [number, number][] = beforeSlot ? beforeSlot.points.map(p => [p.x, p.y]) : []
    const beforeLabs: number[] = beforeSlot ? beforeSlot.points.map(p => p.label as number) : []

    addLocalPoint(oid, frameToUse, nx, ny, label)

    const framePts = useStore.getState().localAnnotations[oid]?.[key]
    const allPoints: [number, number][] = framePts ? framePts.points.map(p => [p.x, p.y]) : [[nx, ny]]
    const allLabels = framePts ? framePts.points.map(p => p.label as number) : [label]
    const afterPts = [...allPoints]
    const afterLabs = [...allLabels]

    try {
      await extractFrame(pid, vid, frameToUse)

      if (registrationMode) {
        const result = await addRegistrationPoints(pid, vid, allPoints, allLabels)
        if (result.mask) {
          const masks = { __registration_floor__: result.mask }
          setCurrentFrameMasks(masks, 0)
          setSavedMask(0, masks)
        }
        const state = useStore.getState()
        const currentProject = state.project
        if (currentProject?.registration) {
          state.setProject({
            ...currentProject,
            registration: {
              ...currentProject.registration,
              status: 'labeling',
              videos: {
                ...currentProject.registration.videos,
                [vid]: {
                  ...currentProject.registration.videos[vid],
                  ...result.registration,
                  frame_idx: 0,
                },
              },
            },
          })
        }
        return
      }

      const result = await addPoints(pid, vid, oid, frameToUse, allPoints, allLabels, anchorPhase)
      const addedObjects = result.new_objects ?? []

      if (result.masks) {
        const state = useStore.getState()
        const prevLive = state.currentFrameMasksFrame === frameToUse ? state.currentFrameMasks : {}
        const newLive = { ...prevLive, ...result.masks }
        setCurrentFrameMasks(newLive, frameToUse)
        const existingSaved = state.savedMaskCache[frameToUse] ?? {}
        setSavedMask(frameToUse, { ...existingSaved, ...result.masks })

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
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail ?? 'Failed to add point', 'error')
    }
  }, [pointMode, currentObjectId, video, currentFrame, width, height, pid, vid, addLocalPoint, setCurrentFrameMasks, setSavedMask, anchorPhase, registrationMode, registrationTool, addRegistrationPolygonPoint, registrationActiveEdge, addRegistrationEdgePoint, addToast])

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
