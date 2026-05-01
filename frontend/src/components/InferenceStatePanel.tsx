import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { RefreshCw, AlertCircle, CheckCircle2, XCircle, ChevronRight, Trash2, Loader } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import { getSessionState, getSavedMask, frameUrl, clearFramePrompts, rebuildFromConfig, replaceFramePromptsData } from '../api/client'
import type { SessionState } from '../api/client'
import { applyRebuildMasksToStore, localAnnotationsToPointPrompts } from '../history/applyRebuild'

// ─── Collapsible section ──────────────────────────────────────────────────────

function Collapsible({ title, defaultOpen = false, children }: {
  title: string
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-1.5 py-1 group"
      >
        <ChevronRight
          size={11}
          className={`text-[#444] flex-shrink-0 transition-transform duration-150 group-hover:text-[#666]
            ${open ? 'rotate-90' : ''}`}
        />
        <span className="text-xs font-semibold text-[#555] uppercase tracking-wider group-hover:text-[#777] transition-colors">
          {title}
        </span>
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  )
}

// ─── Legend / chips ───────────────────────────────────────────────────────────

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: color }} />
      {label}
    </span>
  )
}

function Chip({ label, value, color }: {
  label: string; value: number
  color: 'blue' | 'orange' | 'purple' | 'green' | 'gray'
}) {
  const colors = {
    blue:   'bg-blue-500/10 text-blue-400 border-blue-500/20',
    orange: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    green:  'bg-green-500/10 text-green-400 border-green-500/20',
    gray:   'bg-[#2a2a2a] text-[#888] border-[#333]',
  }
  return (
    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs ${colors[color]}`}>
      <span className="font-semibold font-mono">{value}</span>
      <span className="opacity-70">{label}</span>
    </div>
  )
}

// ─── Frame coverage timeline ──────────────────────────────────────────────────

function FrameTimeline({ state }: { state: SessionState }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { num_frames, frame_map, cached_sam_indices, point_prompts, saved_mask_frames } = state

  const inSessionSet = useMemo(() => new Set(frame_map), [frame_map])
  const cachedRealSet = useMemo(
    () => new Set(cached_sam_indices.map(i => frame_map[i]).filter(v => v !== undefined)),
    [cached_sam_indices, frame_map]
  )
  const promptedSet = useMemo(() => {
    const s = new Set<number>()
    for (const fm of Object.values(point_prompts))
      for (const k of Object.keys(fm)) s.add(Number(k))
    return s
  }, [point_prompts])
  const savedSet = useMemo(() => new Set(saved_mask_frames), [saved_mask_frames])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || num_frames === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, W, H)
    for (let f = 0; f < num_frames; f++) {
      const x = Math.floor((f / num_frames) * W)
      const w = Math.max(1, Math.floor(((f + 1) / num_frames) * W) - x)
      const color = savedSet.has(f) ? '#22c55e'
        : cachedRealSet.has(f) ? '#f97316'
        : promptedSet.has(f) ? '#a78bfa'
        : inSessionSet.has(f) ? '#3b82f6'
        : null
      if (color) { ctx.fillStyle = color; ctx.fillRect(x, 0, w, H) }
    }
  }, [num_frames, inSessionSet, cachedRealSet, promptedSet, savedSet])

  if (num_frames === 0) return null
  return (
    <div>
      <canvas ref={canvasRef} width={800} height={14} className="w-full rounded"
        style={{ imageRendering: 'pixelated' }} />
      <div className="flex flex-wrap gap-3 mt-2 text-xs text-[#888]">
        <LegendItem color="#3b82f6" label="In session" />
        <LegendItem color="#a78bfa" label="Prompted" />
        <LegendItem color="#f97316" label="Cached output" />
        <LegendItem color="#22c55e" label="Saved mask" />
      </div>
    </div>
  )
}

// ─── Session frames table ─────────────────────────────────────────────────────

function SessionFramesTable({ state }: { state: SessionState }) {
  const { frame_map, cached_sam_indices, point_prompts } = state
  const cachedSet = new Set(cached_sam_indices)
  const promptsByRealFrame: Record<number, string[]> = {}
  for (const [oid, fm] of Object.entries(point_prompts)) {
    for (const [fidxStr, data] of Object.entries(fm)) {
      const f = Number(fidxStr)
      if (!promptsByRealFrame[f]) promptsByRealFrame[f] = []
      const nPos = data.labels.filter(l => l === 1).length
      const nNeg = data.labels.filter(l => l === 0).length
      promptsByRealFrame[f].push(`obj${oid}: ${nPos}+ ${nNeg}-`)
    }
  }
  if (frame_map.length === 0)
    return <p className="text-xs text-[#555] italic">No frames loaded in session.</p>
  return (
    <div className="overflow-x-auto max-h-56">
      <table className="w-full text-xs border-collapse">
        <thead className="sticky top-0 bg-[#0d0d0d]">
          <tr className="text-[#555] border-b border-[#2a2a2a]">
            <th className="text-left py-1 pr-4 font-medium">SAM idx</th>
            <th className="text-left py-1 pr-4 font-medium">Real frame</th>
            <th className="text-left py-1 pr-4 font-medium">Cached</th>
            <th className="text-left py-1 font-medium">Prompts</th>
          </tr>
        </thead>
        <tbody>
          {frame_map.map((realIdx, samIdx) => (
            <tr key={samIdx} className="border-b border-[#1e1e1e] hover:bg-[#1e1e1e]">
              <td className="py-1 pr-4 font-mono text-[#666]">{samIdx}</td>
              <td className="py-1 pr-4 font-mono text-[#ccc]">{realIdx}</td>
              <td className="py-1 pr-4">
                {cachedSet.has(samIdx)
                  ? <span className="text-orange-400">✓</span>
                  : <span className="text-[#444]">—</span>}
              </td>
              <td className="py-1 text-[#888]">
                {(promptsByRealFrame[realIdx] ?? []).join(', ') || <span className="text-[#444]">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Objects & prompts ────────────────────────────────────────────────────────

function ObjectsSection({ state }: { state: SessionState }) {
  const { objects, point_prompts, obj_ids_tracked } = state
  const trackedSet = new Set(obj_ids_tracked.map(String))
  const objectList = Object.values(objects)
  if (objectList.length === 0)
    return <p className="text-xs text-[#555] italic">No objects defined.</p>
  return (
    <div className="space-y-3">
      {objectList.map(obj => {
        const prompts = point_prompts[obj.id] ?? {}
        const frameCount = Object.keys(prompts).length
        return (
          <div key={obj.id} className="rounded-lg bg-[#1a1a1a] border border-[#2a2a2a] p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: obj.color }} />
              <span className="text-sm text-[#ddd] font-medium">{obj.name}</span>
              <span className="text-xs text-[#555] font-mono">id={obj.id}</span>
              {trackedSet.has(obj.id) && (
                <span className="ml-auto text-xs text-orange-400 bg-orange-400/10 px-1.5 py-0.5 rounded">
                  in SAM tracker
                </span>
              )}
            </div>
            {frameCount === 0 ? (
              <p className="text-xs text-[#555] italic">No prompts saved.</p>
            ) : (
              <div className="space-y-1">
                {Object.entries(prompts).sort(([a], [b]) => Number(a) - Number(b)).map(([fidxStr, data]) => {
                  const nPos = data.labels.filter(l => l === 1).length
                  const nNeg = data.labels.filter(l => l === 0).length
                  return (
                    <div key={fidxStr} className="flex items-center gap-3 text-xs">
                      <span className="font-mono text-[#666] w-16">frame {fidxStr}</span>
                      <span className="flex items-center gap-1">
                        {Array.from({ length: nPos }).map((_, i) => (
                          <span key={`p${i}`} className="w-2 h-2 rounded-full bg-green-500" />
                        ))}
                        {Array.from({ length: nNeg }).map((_, i) => (
                          <span key={`n${i}`} className="w-2 h-2 rounded-full bg-red-500" />
                        ))}
                      </span>
                      <span className="text-[#666]">
                        {nPos > 0 && `${nPos} pos`}{nPos > 0 && nNeg > 0 && ', '}{nNeg > 0 && `${nNeg} neg`}
                      </span>
                      <span className="text-[#444] font-mono text-[10px] ml-auto">
                        {data.points.map(([x, y]) => `(${x.toFixed(3)}, ${y.toFixed(3)})`).join(' ')}
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ─── Saved masks table ────────────────────────────────────────────────────────

function SavedMasksSection({ state }: { state: SessionState }) {
  const { saved_mask_frames, saved_mask_obj_counts } = state
  if (saved_mask_frames.length === 0)
    return <p className="text-xs text-[#555] italic">No saved masks on disk.</p>
  return (
    <div>
      <p className="text-xs text-[#666] mb-2">
        {saved_mask_frames.length} frames with masks ·{' '}
        {saved_mask_frames.slice(0, 12).join(', ')}{saved_mask_frames.length > 12 ? '…' : ''}
      </p>
      <div className="overflow-x-auto max-h-40">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="text-[#555] border-b border-[#2a2a2a]">
              <th className="text-left py-1 pr-6 font-medium">Frame</th>
              <th className="text-left py-1 font-medium">Objects in file</th>
            </tr>
          </thead>
          <tbody>
            {saved_mask_frames.map(f => (
              <tr key={f} className="border-b border-[#1e1e1e] hover:bg-[#1e1e1e]">
                <td className="py-0.5 pr-6 font-mono text-[#ccc]">{f}</td>
                <td className="py-0.5 text-[#888]">{saved_mask_obj_counts[f] ?? '?'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Labeled frame card ───────────────────────────────────────────────────────

// Internal canvas resolution — displayed at CSS width 100%
const CANVAS_W = 640

interface PointMeta {
  x: number; y: number       // normalized [0,1]
  label: number              // 1 = pos, 0 = neg
  objId: string
  name: string
  color: string
}

type TooltipState =
  | { type: 'point'; cssX: number; cssY: number; px: number; py: number; label: number; name: string; color: string }
  | { type: 'mask';  cssX: number; cssY: number; name: string; color: string; pct: number; bboxW: number; bboxH: number; bboxX: number; bboxY: number }

function ToggleBtn({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-0.5 rounded text-xs font-medium transition-colors border ${
        active
          ? 'bg-[#2a2a2a] text-[#ccc] border-[#444]'
          : 'text-[#555] border-transparent hover:text-[#888]'
      }`}
    >
      {label}
    </button>
  )
}

function LabeledFrameCard({
  frameIdx,
  objPrompts,
  objects,
  videoWidth,
  videoHeight,
  fps,
  pid,
  vid,
  onRemove,
}: {
  frameIdx: number
  objPrompts: Record<string, { points: [number, number][]; labels: number[] }>
  objects: SessionState['objects']
  videoWidth: number
  videoHeight: number
  fps: number
  pid: string
  vid: string
  onRemove: () => void
}) {
  const canvasH = videoWidth > 0 ? Math.round(CANVAS_W * videoHeight / videoWidth) : 360

  const [showImage, setShowImage] = useState(true)
  const [showMasks, setShowMasks] = useState(true)
  const [showPoints, setShowPoints] = useState(true)
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)
  const [renderTick, setRenderTick] = useState(0)
  const [removing, setRemoving] = useState(false)

  async function handleRemove() {
    if (!confirm(`Remove frame ${frameIdx} from inference state? This clears its point prompts.`)) return
    setRemoving(true)
    try {
      const st = useStore.getState()
      const perObj: Record<string, { points: [number, number][]; labels: number[] }> = {}
      for (const [objId, prompt] of Object.entries(objPrompts)) {
        perObj[objId] = { points: [...prompt.points], labels: [...prompt.labels] }
      }
      const laSnap = JSON.parse(JSON.stringify(st.localAnnotations)) as typeof st.localAnnotations
      const fKey = String(frameIdx)

      await clearFramePrompts(pid, vid, frameIdx)

      const la = JSON.parse(JSON.stringify(st.localAnnotations)) as typeof st.localAnnotations
      for (const objId of Object.keys(la)) {
        if (la[objId]?.[fKey]) {
          delete la[objId][fKey]
          if (Object.keys(la[objId]).length === 0) delete la[objId]
        }
      }
      useStore.setState({ localAnnotations: la })
      useStore.getState().updateVideo({ point_prompts: localAnnotationsToPointPrompts(la) })

      useStore.getState().pushHistory({
        labelUndo: 'Clear frame prompts',
        labelRedo: 'Clear frame prompts',
        undo: async () => {
          for (const [objId, pr] of Object.entries(perObj)) {
            await replaceFramePromptsData(pid, vid, objId, frameIdx, pr.points, pr.labels)
          }
          const restored = JSON.parse(JSON.stringify(laSnap)) as typeof laSnap
          useStore.setState({ localAnnotations: restored })
          useStore.getState().updateVideo({ point_prompts: localAnnotationsToPointPrompts(restored) })
          const anchorNow = useStore.getState().anchorPhase
          const rb = await rebuildFromConfig(pid, vid, [frameIdx], anchorNow, anchorNow ? frameIdx : null)
          applyRebuildMasksToStore(rb.masks_by_frame)
        },
        redo: async () => {
          await clearFramePrompts(pid, vid, frameIdx)
          const la2 = JSON.parse(JSON.stringify(useStore.getState().localAnnotations)) as typeof st.localAnnotations
          for (const oid of Object.keys(la2)) {
            if (la2[oid]?.[fKey]) {
              delete la2[oid][fKey]
              if (Object.keys(la2[oid]).length === 0) delete la2[oid]
            }
          }
          useStore.setState({ localAnnotations: la2 })
          useStore.getState().updateVideo({ point_prompts: localAnnotationsToPointPrompts(la2) })
        },
      })

      onRemove()
    } catch {
      setRemoving(false)
    } finally {
      setRemoving(false)
    }
  }

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameImgRef = useRef<HTMLImageElement | null>(null)
  const maskImgsRef = useRef<Record<string, HTMLImageElement>>({})
  const maskAlphaRef = useRef<Record<string, Uint8ClampedArray>>({})
  const maskStatsRef = useRef<Record<string, { pct: number; bboxX: number; bboxY: number; bboxW: number; bboxH: number }>>({})

  // All points with metadata, memoized so canvas effect deps are stable
  const allPoints = useMemo<PointMeta[]>(() => {
    const pts: PointMeta[] = []
    for (const [objId, { points, labels }] of Object.entries(objPrompts)) {
      const obj = objects[objId]
      for (let i = 0; i < points.length; i++) {
        pts.push({ x: points[i][0], y: points[i][1], label: labels[i],
          objId, name: obj?.name ?? objId, color: obj?.color ?? '#fff' })
      }
    }
    return pts
  }, [objPrompts, objects])

  // Load frame image
  useEffect(() => {
    frameImgRef.current = null
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => { frameImgRef.current = img; setRenderTick(t => t + 1) }
    img.onerror = () => setRenderTick(t => t + 1)
    img.src = frameUrl(pid, vid, frameIdx)
    return () => { img.onload = null; img.onerror = null }
  }, [pid, vid, frameIdx])

  // Load mask images and extract alpha channel for hit-testing
  useEffect(() => {
    maskImgsRef.current = {}
    maskAlphaRef.current = {}
    maskStatsRef.current = {}
    getSavedMask(pid, vid, frameIdx)
      .then(data => {
        const entries = Object.entries(data.masks ?? {})
        if (entries.length === 0) { setRenderTick(t => t + 1); return }
        let remaining = entries.length
        for (const [objId, b64] of entries) {
          const img = new Image()
          const done = () => { remaining--; if (remaining === 0) setRenderTick(t => t + 1) }
          img.onload = () => {
            maskImgsRef.current[objId] = img
            // Extract alpha at canvas resolution for hit-testing + stats
            try {
              const oc = document.createElement('canvas')
              oc.width = CANVAS_W; oc.height = canvasH
              const octx = oc.getContext('2d')
              if (octx) {
                octx.drawImage(img, 0, 0, CANVAS_W, canvasH)
                const idata = octx.getImageData(0, 0, CANVAS_W, canvasH).data
                const alpha = new Uint8ClampedArray(CANVAS_W * canvasH)
                let count = 0, minX = CANVAS_W, minY = canvasH, maxX = 0, maxY = 0
                for (let i = 0; i < CANVAS_W * canvasH; i++) {
                  const a = idata[i * 4 + 3]
                  alpha[i] = a
                  if (a > 127) {
                    count++
                    const x = i % CANVAS_W, y = Math.floor(i / CANVAS_W)
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                  }
                }
                maskAlphaRef.current[objId] = alpha
                if (count > 0) {
                  const sx = videoWidth / CANVAS_W, sy = videoHeight / canvasH
                  maskStatsRef.current[objId] = {
                    pct: (count / (CANVAS_W * canvasH)) * 100,
                    bboxX: Math.round(minX * sx), bboxY: Math.round(minY * sy),
                    bboxW: Math.round((maxX - minX + 1) * sx),
                    bboxH: Math.round((maxY - minY + 1) * sy),
                  }
                }
              }
            } catch { /* offscreen canvas unavailable */ }
            done()
          }
          img.onerror = done
          img.src = `data:image/png;base64,${b64}`
        }
      })
      .catch(() => setRenderTick(t => t + 1))
  }, [pid, vid, frameIdx, canvasH, videoWidth, videoHeight])

  // Draw to canvas whenever inputs change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#111111'
    ctx.fillRect(0, 0, W, H)

    if (showImage && frameImgRef.current)
      ctx.drawImage(frameImgRef.current, 0, 0, W, H)

    if (showMasks) {
      ctx.save()
      ctx.globalAlpha = 0.45
      for (const img of Object.values(maskImgsRef.current))
        ctx.drawImage(img, 0, 0, W, H)
      ctx.restore()
    }

    if (showPoints) {
      for (const pt of allPoints) {
        const cx = pt.x * W, cy = pt.y * H
        // Shadow ring
        ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fill()
        // Fill
        ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2)
        ctx.fillStyle = pt.label === 1 ? '#22c55e' : '#ef4444'; ctx.fill()
        // Border
        ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = 1.5; ctx.stroke()
        // +/- label
        ctx.fillStyle = 'white'
        ctx.font = 'bold 9px monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(pt.label === 1 ? '+' : '−', cx, cy)
      }
    }
  }, [renderTick, showImage, showMasks, showPoints, allPoints])

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    const cssX = e.clientX - rect.left
    const cssY = e.clientY - rect.top

    // Priority 1: point hit (within 14 CSS px)
    if (showPoints && allPoints.length > 0) {
      let found: TooltipState | null = null
      let minDist = 14
      for (const pt of allPoints) {
        const d = Math.hypot(cssX - pt.x * rect.width, cssY - pt.y * rect.height)
        if (d < minDist) {
          minDist = d
          found = {
            type: 'point',
            cssX: pt.x * rect.width, cssY: pt.y * rect.height,
            px: Math.round(pt.x * videoWidth), py: Math.round(pt.y * videoHeight),
            label: pt.label, name: pt.name, color: pt.color,
          }
        }
      }
      if (found) { setTooltip(found); return }
    }

    // Priority 2: mask hit (alpha channel lookup)
    if (showMasks) {
      const canvasX = Math.round(cssX * CANVAS_W / rect.width)
      const canvasY = Math.round(cssY * canvasH / rect.height)
      const idx = canvasY * CANVAS_W + canvasX
      const maskEntries = Object.entries(maskAlphaRef.current)
      // Iterate in reverse so the top-drawn mask wins
      for (let i = maskEntries.length - 1; i >= 0; i--) {
        const [objId, alpha] = maskEntries[i]
        if (idx >= 0 && idx < alpha.length && alpha[idx] > 127) {
          const stats = maskStatsRef.current[objId]
          if (stats) {
            const obj = objects[objId]
            setTooltip({
              type: 'mask', cssX, cssY,
              name: obj?.name ?? objId, color: obj?.color ?? '#fff',
              pct: stats.pct, bboxX: stats.bboxX, bboxY: stats.bboxY,
              bboxW: stats.bboxW, bboxH: stats.bboxH,
            })
            return
          }
        }
      }
    }

    setTooltip(null)
  }

  const timeSec = Math.floor(frameIdx / Math.max(fps, 1))
  const timeStr = `${Math.floor(timeSec / 60)}:${(timeSec % 60).toString().padStart(2, '0')}`
  const totalPoints = allPoints.length
  const nPos = allPoints.filter(p => p.label === 1).length
  const nNeg = allPoints.filter(p => p.label === 0).length

  return (
    <div className="rounded-xl border border-[#2a2a2a] overflow-hidden bg-[#111]">
      {/* Card header */}
      <div className="flex items-center gap-3 px-3 py-2 bg-[#161616] border-b border-[#2a2a2a]">
        <span className="font-mono text-sm font-semibold text-[#ddd]">Frame {frameIdx}</span>
        <span className="text-xs text-[#555]">{timeStr}</span>
        <span className="text-xs text-[#555]">
          {totalPoints} point{totalPoints !== 1 ? 's' : ''}
          {totalPoints > 0 && (
            <span> · <span className="text-green-500">{nPos}+</span> <span className="text-red-500">{nNeg}−</span></span>
          )}
        </span>
        <div className="flex-1" />
        <div className="flex gap-1 items-center">
          <ToggleBtn active={showImage} onClick={() => setShowImage(v => !v)} label="Image" />
          <ToggleBtn active={showMasks} onClick={() => setShowMasks(v => !v)} label="Masks" />
          <ToggleBtn active={showPoints} onClick={() => setShowPoints(v => !v)} label="Points" />
          <button
            onClick={handleRemove}
            disabled={removing}
            title="Remove from inference state"
            className="ml-1 p-1 rounded text-[#555] hover:text-red-400 hover:bg-red-400/10 transition-colors disabled:opacity-40"
          >
            {removing ? <Loader size={11} className="animate-spin" /> : <Trash2 size={11} />}
          </button>
        </div>
      </div>

      {/* Canvas with tooltip */}
      <div className="relative overflow-hidden" style={{ cursor: showPoints || showMasks ? 'crosshair' : 'default' }}>
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={canvasH}
          className="w-full block"
          style={{ imageRendering: 'auto' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
        />

        {tooltip && (
          <div
            className="absolute pointer-events-none z-20 rounded px-2 py-1 text-xs text-white whitespace-nowrap shadow-lg"
            style={{
              left: tooltip.cssX + 10,
              top: tooltip.cssY > 40 ? tooltip.cssY - 40 : tooltip.cssY + 14,
              background: 'rgba(0,0,0,0.85)',
              border: '1px solid rgba(255,255,255,0.15)',
            }}
          >
            <span className="font-medium" style={{ color: tooltip.color }}>{tooltip.name}</span>
            {tooltip.type === 'point' ? (
              <>
                <span className="mx-1.5 text-[#555]">·</span>
                <span className="font-mono">({tooltip.px}, {tooltip.py})px</span>
                <span className="mx-1.5 text-[#555]">·</span>
                <span className={tooltip.label === 1 ? 'text-green-400' : 'text-red-400'}>
                  {tooltip.label === 1 ? '+ positive' : '− negative'}
                </span>
              </>
            ) : (
              <>
                <span className="mx-1.5 text-[#555]">·</span>
                <span className="text-[#aaa]">{tooltip.pct.toFixed(1)}% area</span>
                <span className="mx-1.5 text-[#555]">·</span>
                <span className="font-mono text-[#888]">{tooltip.bboxW}×{tooltip.bboxH} @ ({tooltip.bboxX},{tooltip.bboxY})</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Per-object point legend below canvas */}
      {showPoints && allPoints.length > 0 && (
        <div className="px-3 py-2 border-t border-[#1e1e1e] flex flex-wrap gap-x-4 gap-y-1">
          {Object.entries(objPrompts).map(([objId, { points, labels }]) => {
            const obj = objects[objId]
            return (
              <div key={objId} className="flex items-center gap-2 text-xs">
                <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: obj?.color ?? '#888' }} />
                <span className="text-[#aaa]">{obj?.name ?? objId}</span>
                <span className="text-[#555]">
                  {labels.filter(l => l === 1).length}+ · {labels.filter(l => l === 0).length}−
                </span>
                <span className="font-mono text-[10px] text-[#444]">
                  {points.map(([x, y]) => `(${Math.round(x * (videoWidth || 1))},${Math.round(y * (videoHeight || 1))})`).join(' ')}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─── Labeled frames section ───────────────────────────────────────────────────

function LabeledFramesSection({ state, videoWidth, videoHeight, fps, pid, vid, onRefresh }: {
  state: SessionState
  videoWidth: number
  videoHeight: number
  fps: number
  pid: string
  vid: string
  onRefresh: () => void
}) {
  // Collect all frames that have at least one point prompt
  const frameToObjPrompts = useMemo(() => {
    const result: Record<number, Record<string, { points: [number, number][]; labels: number[] }>> = {}
    for (const [objId, frameMap] of Object.entries(state.point_prompts)) {
      for (const [fidxStr, promptData] of Object.entries(frameMap)) {
        const f = Number(fidxStr)
        if (!result[f]) result[f] = {}
        result[f][objId] = promptData
      }
    }
    return result
  }, [state.point_prompts])

  const sortedFrames = useMemo(
    () => Object.keys(frameToObjPrompts).map(Number).sort((a, b) => a - b),
    [frameToObjPrompts]
  )

  if (sortedFrames.length === 0)
    return <p className="text-xs text-[#555] italic">No labeled frames yet.</p>

  return (
    <div className="space-y-4">
      {sortedFrames.map(f => (
        <LabeledFrameCard
          key={f}
          frameIdx={f}
          objPrompts={frameToObjPrompts[f]}
          objects={state.objects}
          videoWidth={videoWidth}
          videoHeight={videoHeight}
          fps={fps}
          pid={pid}
          vid={vid}
          onRemove={onRefresh}
        />
      ))}
    </div>
  )
}

// ─── Main panel ───────────────────────────────────────────────────────────────

export default function InferenceStatePanel() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { project, currentVideoId } = store
  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  const [state, setState] = useState<SessionState | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    if (!pid || !vid) return
    setLoading(true)
    setError('')
    try {
      setState(await getSessionState(pid, vid))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load session state')
    } finally {
      setLoading(false)
    }
  }, [pid, vid])

  useEffect(() => { load() }, [load])

  if (!video) {
    return (
      <div className="flex-1 flex items-center justify-center text-[#555] text-sm">
        No video selected.
      </div>
    )
  }

  const promptedFrameCount = state
    ? new Set(Object.values(state.point_prompts).flatMap(m => Object.keys(m)).map(Number)).size
    : 0

  return (
    <div className="flex-1 overflow-y-auto bg-[#0d0d0d] text-[#ccc]">
      {/* Sticky header */}
      <div className="sticky top-0 z-10 bg-[#111111] border-b border-[#2a2a2a] px-4 py-2 flex items-center gap-3">
        {state && (
          <>
            <span className="text-xs font-mono font-semibold text-[#aaa] uppercase">
              {state.model !== 'none' ? state.model.toUpperCase() : 'No model'}
            </span>
            <span className="text-[#333]">·</span>
            {state.session_active ? (
              <span className="flex items-center gap-1.5 text-xs text-green-400">
                <CheckCircle2 size={12} /> Session active
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-xs text-[#555]">
                <XCircle size={12} /> No session
              </span>
            )}
            {state.session_id && (
              <>
                <span className="text-[#333]">·</span>
                <span className="text-xs font-mono text-[#555]" title={state.session_id}>
                  {state.session_id.slice(0, 24)}{state.session_id.length > 24 ? '…' : ''}
                </span>
              </>
            )}
          </>
        )}
        <div className="flex-1" />
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-1.5 text-xs text-[#666] hover:text-[#aaa] transition-colors disabled:opacity-40"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="mx-4 mt-4 flex items-center gap-2 text-xs text-red-400 bg-red-400/10 rounded-lg px-3 py-2">
          <AlertCircle size={12} /> {error}
        </div>
      )}
      {loading && !state && (
        <div className="flex items-center justify-center h-40 text-xs text-[#555]">Loading…</div>
      )}

      {state && (
        <div className="p-4 space-y-4">

          <Collapsible title="Summary" defaultOpen>
            <div className="flex flex-wrap gap-2 mt-1">
              <Chip label="Frames in session" value={state.frame_map.length} color="blue" />
              <Chip label="Cached outputs" value={state.cached_sam_indices.length} color="orange" />
              <Chip label="Prompted frames" value={promptedFrameCount} color="purple" />
              <Chip label="Saved mask frames" value={state.saved_mask_frames.length} color="green" />
              <Chip label="Action history" value={state.action_history_len} color="gray" />
            </div>
          </Collapsible>

          <Collapsible title="Frame Coverage" defaultOpen>
            <FrameTimeline state={state} />
            <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-[#666]">
              <span>Total video frames: <span className="text-[#aaa]">{state.num_frames.toLocaleString()}</span></span>
              <span>Annotated frames on disk: <span className="text-[#aaa]">{state.annotated_frame_files.length}</span></span>
              <span>SAM objects tracked: <span className="text-[#aaa]">{state.obj_ids_tracked.join(', ') || '—'}</span></span>
            </div>
          </Collapsible>

          <Collapsible
            title={`Labeled Frames (${promptedFrameCount})`}
            defaultOpen
          >
            <LabeledFramesSection
              state={state}
              videoWidth={video.width}
              videoHeight={video.height}
              fps={video.fps}
              pid={pid}
              vid={vid}
              onRefresh={load}
            />
          </Collapsible>

          <Collapsible title={`Session Frame Map (${state.frame_map.length})`}>
            <SessionFramesTable state={state} />
          </Collapsible>

          <Collapsible title={`Objects & Point Prompts (${Object.keys(state.objects).length})`}>
            <ObjectsSection state={state} />
          </Collapsible>

          <Collapsible title="Saved Masks on Disk">
            <SavedMasksSection state={state} />
          </Collapsible>

        </div>
      )}
    </div>
  )
}
