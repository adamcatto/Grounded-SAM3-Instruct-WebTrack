import { useEffect, useRef, useState, useCallback, useMemo, memo } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import { getUncertaintyFrameDetail, type UncertaintyFrameDetail } from '../api/client'
import { ChevronRight, ChevronDown, Eye, EyeOff, Loader } from 'lucide-react'

// ── Helpers ────────────────────────────────────────────────────────────────────

function scoreColor(score: number): string {
  if (score >= 0.7) return '#ef4444'
  if (score >= 0.4) return '#f59e0b'
  return '#22c55e'
}

function scoreLabel(score: number): string {
  if (score >= 0.7) return 'High'
  if (score >= 0.4) return 'Med'
  return 'Low'
}

// ── Heatmap canvas ────────────────────────────────────────────────────────────

interface HeatmapProps {
  perFrame: Record<string, { confusion_score: number }>
  total: number
  startFrame: number
  currentFrame: number
  onJump: (f: number) => void
}

const ConfusionHeatmap = memo(function ConfusionHeatmap({ perFrame, total, startFrame, currentFrame, onJump }: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rangeLen = total - 1 - startFrame

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || rangeLen <= 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const W = canvas.width
    const H = canvas.height
    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, W, H)

    for (const [fidxStr, { confusion_score }] of Object.entries(perFrame)) {
      const fidx = parseInt(fidxStr)
      if (fidx < startFrame) continue
      const x = Math.round(((fidx - startFrame) / rangeLen) * W)
      const intensity = Math.min(1, confusion_score)
      if (intensity < 0.05) continue
      if (intensity >= 0.7) ctx.fillStyle = `rgba(239,68,68,${intensity})`
      else if (intensity >= 0.4) ctx.fillStyle = `rgba(245,158,11,${intensity})`
      else ctx.fillStyle = `rgba(34,197,94,${intensity * 0.6})`
      ctx.fillRect(x, 0, Math.max(2, Math.round(W / rangeLen)), H)
    }

    const cx = Math.round(((currentFrame - startFrame) / rangeLen) * W)
    ctx.fillStyle = 'rgba(255,255,255,0.8)'
    ctx.fillRect(cx, 0, 1, H)
  }, [perFrame, rangeLen, currentFrame, startFrame])

  return (
    <canvas
      ref={canvasRef}
      width={1200}
      height={20}
      className="w-full h-5 rounded cursor-pointer"
      onClick={e => {
        const rect = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect()
        const pct = (e.clientX - rect.left) / rect.width
        onJump(Math.max(startFrame, Math.min(total - 1, Math.round(pct * rangeLen + startFrame))))
      }}
    />
  )
})

// ── Inline frame preview canvas ───────────────────────────────────────────────

function FramePreviewCanvas({ detail }: { detail: UncertaintyFrameDetail }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !detail.frame_image) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      for (const maskB64 of Object.values(detail.masks)) {
        const maskImg = new Image()
        maskImg.onload = () => {
          ctx.globalAlpha = 0.5
          ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height)
          ctx.globalAlpha = 1.0
        }
        maskImg.src = `data:image/png;base64,${maskB64}`
      }
    }
    img.src = `data:image/jpeg;base64,${detail.frame_image}`
  }, [detail])

  if (!detail.frame_image) {
    return (
      <div className="h-24 flex items-center justify-center text-[#555] text-xs rounded bg-black/40">
        Frame image unavailable
      </div>
    )
  }

  return <canvas ref={canvasRef} className="w-full h-auto rounded" />
}

// ── Frame row ─────────────────────────────────────────────────────────────────

interface PerFrameEntry {
  confusion_score: number
  confused_objects?: string[]
  temporal_rejections?: Record<string, string>
  per_object?: Record<string, { anomaly_score: number }>
}

interface FrameRowProps {
  fidx: number
  data: PerFrameEntry
  objects: { id: string; name: string; color: string }[]
  pid: string
  vid: string
  isExpanded: boolean
  onToggleExpand: () => void
  onJump: (f: number) => void
}

const FrameRow = memo(function FrameRow({ fidx, data, objects, pid, vid, isExpanded, onToggleExpand, onJump }: FrameRowProps) {
  const [showPreview, setShowPreview] = useState(false)
  const [detail, setDetail] = useState<UncertaintyFrameDetail | null>(null)
  const [loading, setLoading] = useState(false)

  const togglePreview = useCallback(async () => {
    const next = !showPreview
    setShowPreview(next)
    if (next && !detail) {
      setLoading(true)
      try {
        const d = await getUncertaintyFrameDetail(pid, vid, fidx)
        setDetail(d)
      } catch {
        // ignore
      } finally {
        setLoading(false)
      }
    }
  }, [showPreview, detail, pid, vid, fidx])

  const score = data.confusion_score

  return (
    <div style={{ borderBottom: '1px solid #1e1e1e' }}>
      {/* Row header — fixed height, must match COLLAPSED_H constant */}
      <div
        style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', cursor: 'pointer', userSelect: 'none', height: 36, boxSizing: 'border-box' }}
        onClick={onToggleExpand}
      >
        <span style={{ color: '#444', flexShrink: 0, width: 12 }}>
          {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </span>
        <span style={{ fontFamily: 'monospace', fontSize: 11, color: '#888', width: 52, flexShrink: 0 }}>{fidx}</span>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ flex: 1, height: 6, borderRadius: 3, background: '#1a1a1a', overflow: 'hidden' }}>
            <div style={{ height: '100%', borderRadius: 3, width: `${Math.min(100, score * 100)}%`, background: scoreColor(score), opacity: 0.7 }} />
          </div>
          <span style={{ fontSize: 10, fontFamily: 'monospace', flexShrink: 0, color: scoreColor(score) }}>{score.toFixed(3)}</span>
          <span style={{ fontSize: 9, padding: '1px 4px', borderRadius: 3, flexShrink: 0, background: `${scoreColor(score)}22`, color: scoreColor(score) }}>{scoreLabel(score)}</span>
        </div>
        {data.confused_objects && data.confused_objects.length > 0 && (
          <span style={{ fontSize: 9, color: '#f59e0b', flexShrink: 0 }}>confused</span>
        )}
        {data.temporal_rejections && Object.keys(data.temporal_rejections).length > 0 && (
          <span style={{ fontSize: 9, color: '#60a5fa', flexShrink: 0 }}>rejected</span>
        )}
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div style={{ padding: '8px 16px 12px', background: '#0a0a0a', display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 10, color: '#555' }}>Frame {fidx}</span>
            <button onClick={() => onJump(fidx)} style={{ fontSize: 10, color: '#60a5fa', background: 'none', border: 'none', cursor: 'pointer' }}>
              Jump to frame
            </button>
          </div>

          {data.per_object && Object.keys(data.per_object).length > 0 && (
            <div>
              <p style={{ fontSize: 9, color: '#555', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Per-object anomaly</p>
              {Object.entries(data.per_object).map(([objId, scores]) => {
                const obj = objects.find(o => o.id === objId)
                const anomaly = scores.anomaly_score ?? 0
                return (
                  <div key={objId} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', flexShrink: 0, background: obj?.color ?? '#888' }} />
                    <span style={{ fontSize: 12, color: '#aaa', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{obj?.name ?? `Obj ${objId}`}</span>
                    <div style={{ width: 64, height: 4, background: '#2a2a2a', borderRadius: 2, overflow: 'hidden', flexShrink: 0 }}>
                      <div style={{ height: '100%', width: `${Math.min(100, (anomaly / 5) * 100)}%`, background: anomaly > 2.5 ? '#ef4444' : anomaly > 1.5 ? '#f59e0b' : '#22c55e' }} />
                    </div>
                    <span style={{ fontSize: 10, fontFamily: 'monospace', color: '#666', width: 32, textAlign: 'right', flexShrink: 0 }}>{anomaly.toFixed(2)}</span>
                  </div>
                )
              })}
            </div>
          )}

          {data.temporal_rejections && Object.keys(data.temporal_rejections).length > 0 && (
            <div>
              <p style={{ fontSize: 9, color: '#555', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Temporal rejections</p>
              {Object.entries(data.temporal_rejections).map(([objId, reason]) => {
                const obj = objects.find(o => o.id === objId)
                return (
                  <div key={objId} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 10 }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', flexShrink: 0, background: obj?.color ?? '#888' }} />
                    <span style={{ color: '#888' }}>{obj?.name ?? `Obj ${objId}`}:</span>
                    <span style={{ color: '#93c5fd', fontFamily: 'monospace' }}>{reason}</span>
                  </div>
                )
              })}
            </div>
          )}

          {data.confused_objects && data.confused_objects.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 9, color: '#555', textTransform: 'uppercase' }}>Confused:</span>
              {data.confused_objects.map(objId => {
                const obj = objects.find(o => o.id === objId)
                return (
                  <span key={objId} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: '#fcd34d' }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: obj?.color ?? '#f59e0b' }} />
                    {obj?.name ?? `Obj ${objId}`}
                  </span>
                )
              })}
            </div>
          )}

          <div>
            <button
              onClick={togglePreview}
              style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: '#666', background: 'none', border: 'none', cursor: 'pointer' }}
            >
              {showPreview ? <EyeOff size={11} /> : <Eye size={11} />}
              {showPreview ? 'Hide frame preview' : 'Show frame preview'}
            </button>
            {showPreview && (
              <div style={{ marginTop: 8 }}>
                {loading ? (
                  <div style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, color: '#555', fontSize: 12 }}>
                    <Loader size={12} className="animate-spin" /> Loading...
                  </div>
                ) : detail ? (
                  <FramePreviewCanvas detail={detail} />
                ) : (
                  <div style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#555', fontSize: 12 }}>Failed to load</div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
})

// ── Virtual frame list ────────────────────────────────────────────────────────
// COLLAPSED_H must exactly match the height of a collapsed FrameRow header.
const COLLAPSED_H = 36
const EXPANDED_H  = 280
const OVERSCAN    = 5

interface VirtualFrameListProps {
  frameEntries: [number, PerFrameEntry][]
  expandedFrames: Set<number>
  objects: { id: string; name: string; color: string }[]
  pid: string
  vid: string
  onToggleExpand: (fidx: number) => void
  onJump: (f: number) => void
}

const VirtualFrameList = memo(function VirtualFrameList({
  frameEntries, expandedFrames, objects, pid, vid, onToggleExpand, onJump,
}: VirtualFrameListProps) {
  const outerRef = useRef<HTMLDivElement>(null)
  // scrollTop is read via ref during render to avoid async stale state
  const scrollTopRef = useRef(0)
  const [, forceRender] = useState(0)
  const [viewH, setViewH] = useState(600)

  useEffect(() => {
    const el = outerRef.current
    if (!el) return
    setViewH(el.clientHeight)
    const ro = new ResizeObserver(() => setViewH(el.clientHeight))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Offsets — only recomputed when entries or expanded set changes, not on scroll
  const { offsets, totalH } = useMemo(() => {
    const offs = new Float64Array(frameEntries.length + 1)
    for (let i = 0; i < frameEntries.length; i++) {
      offs[i + 1] = offs[i] + (expandedFrames.has(frameEntries[i][0]) ? EXPANDED_H : COLLAPSED_H)
    }
    return { offsets: offs, totalH: offs[frameEntries.length] }
  }, [frameEntries, expandedFrames])

  const scrollTop = scrollTopRef.current

  // Binary search for first visible row
  let lo = 0, hi = frameEntries.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (offsets[mid + 1] <= scrollTop) lo = mid + 1
    else hi = mid
  }
  const start = Math.max(0, lo - OVERSCAN)

  // Binary search for last visible row
  lo = start; hi = frameEntries.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (offsets[mid] < scrollTop + viewH) lo = mid
    else hi = mid - 1
  }
  const end = Math.min(frameEntries.length - 1, lo + OVERSCAN)

  const paddingTop = offsets[start]

  return (
    <div
      ref={outerRef}
      style={{ height: '60vh', overflowY: 'auto', position: 'relative' }}
      onScroll={e => {
        scrollTopRef.current = (e.currentTarget as HTMLDivElement).scrollTop
        forceRender(n => n + 1)
      }}
    >
      {/* Full-height spacer makes the scrollbar reflect total content */}
      <div style={{ height: totalH }}>
        {/* Only the visible slice, positioned at the right offset */}
        <div style={{ position: 'sticky', top: 0, height: 0 }}>
          <div style={{ position: 'absolute', top: paddingTop - scrollTop, left: 0, right: 0 }}>
            {frameEntries.slice(start, end + 1).map(([fidx, data]) => (
              <FrameRow
                key={fidx}
                fidx={fidx}
                data={data}
                objects={objects}
                pid={pid}
                vid={vid}
                isExpanded={expandedFrames.has(fidx)}
                onToggleExpand={() => onToggleExpand(fidx)}
                onJump={onJump}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
})

// ── Main panel ────────────────────────────────────────────────────────────────

export default function UncertaintyPanel() {
  // Use granular selectors to avoid re-rendering when unrelated store fields change
  const uncertaintyData = useStore(s => s.uncertaintyData)
  const currentFrame    = useStore(s => s.currentFrame)
  const setCurrentFrame = useStore(s => s.setCurrentFrame)
  const propagationStartFrame = useStore(s => s.propagationStartFrame)
  const video = useStore(s => selectCurrentVideo(s))
  const project = useStore(s => s.project)

  const [expandedFrames, setExpandedFrames] = useState<Set<number>>(new Set())

  // Compute sorted frame entries lazily so the first render isn't blocked.
  // Object.entries + sort on 50k keys is expensive; defer to after first paint.
  const [frameEntries, setFrameEntries] = useState<[number, PerFrameEntry][] | null>(null)

  const perFrame = uncertaintyData?.per_frame ?? null
  useEffect(() => {
    if (!perFrame) { setFrameEntries(null); return }
    setFrameEntries(null)  // clear stale entries while recomputing
    const tid = setTimeout(() => {
      const entries = Object.entries(perFrame)
        .map(([k, v]) => [parseInt(k), v] as [number, PerFrameEntry])
        .sort(([a], [b]) => a - b)
      setFrameEntries(entries)
    }, 0)
    return () => clearTimeout(tid)
  }, [perFrame])

  const simPairs = useMemo(() => {
    const pairs: { a: string; b: string; score: number }[] = []
    for (const [key, score] of Object.entries(uncertaintyData?.similarity_matrix ?? {})) {
      const [a, b] = key.split('_')
      pairs.push({ a, b, score })
    }
    pairs.sort((x, y) => y.score - x.score)
    return pairs
  }, [uncertaintyData?.similarity_matrix])

  const toggleExpand = useCallback((fidx: number) => {
    setExpandedFrames(prev => {
      const next = new Set(prev)
      if (next.has(fidx)) next.delete(fidx)
      else next.add(fidx)
      return next
    })
  }, [])

  const pid = project?.id ?? ''
  const vid = video?.id ?? ''

  if (!uncertaintyData || !uncertaintyData.per_frame) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0d0d0d] text-[#555] text-sm">
        No uncertainty data available. Run propagation to generate uncertainty analysis.
      </div>
    )
  }

  const { per_frame, confusion_windows } = uncertaintyData

  if (Object.keys(per_frame).length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0d0d0d] text-[#555] text-sm text-center px-8">
        <div>
          <p className="mb-1">Propagation completed but no per-frame uncertainty was recorded.</p>
          <p className="text-xs text-[#444]">Re-run propagation to generate a fresh uncertainty report.</p>
        </div>
      </div>
    )
  }

  const objects = video ? Object.values(video.objects) : []
  const total = video?.num_frames ?? 1
  const startFrame = propagationStartFrame
  const rangeLen = total - 1 - startFrame

  return (
    <div className="flex-1 overflow-y-auto bg-[#0d0d0d] text-[#ccc]">
      <div className="max-w-2xl mx-auto p-6 space-y-8">

        {/* Heatmap */}
        <div>
          <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-2">Confusion Heatmap</h2>
          <p className="text-xs text-[#666] mb-3">Color intensity shows identity confusion per frame. Click to jump.</p>
          <ConfusionHeatmap
            perFrame={per_frame}
            total={total}
            startFrame={startFrame}
            currentFrame={currentFrame}
            onJump={setCurrentFrame}
          />
          <div className="flex items-center gap-4 mt-2">
            {[['Low', '#22c55e'], ['Medium', '#f59e0b'], ['High', '#ef4444']].map(([label, color]) => (
              <div key={label} className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
                <span className="text-[10px] text-[#666]">{label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* All frames — virtualized list */}
        <div>
          <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-1">
            All Frames ({Object.keys(per_frame).length})
          </h2>
          <p className="text-xs text-[#666] mb-3">
            Click any frame to expand its metrics. Toggle the eye icon to load a frame image preview.
          </p>
          <div className="rounded-xl border border-[#2a2a2a] bg-[#111] overflow-hidden">
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 12px', background: '#151515', borderBottom: '1px solid #2a2a2a' }}>
              <span style={{ width: 12, flexShrink: 0 }} />
              <span style={{ fontSize: 10, color: '#555', width: 52, flexShrink: 0 }}>Frame</span>
              <span style={{ fontSize: 10, color: '#555', flex: 1 }}>Confusion score</span>
            </div>
            {frameEntries === null ? (
              <div style={{ height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, color: '#555', fontSize: 12 }}>
                <Loader size={14} className="animate-spin" />
                Sorting {Object.keys(per_frame).length.toLocaleString()} frames…
              </div>
            ) : (
              <VirtualFrameList
                frameEntries={frameEntries}
                expandedFrames={expandedFrames}
                objects={objects}
                pid={pid}
                vid={vid}
                onToggleExpand={toggleExpand}
                onJump={setCurrentFrame}
              />
            )}
          </div>
        </div>

        {/* Similarity matrix */}
        {simPairs.length > 0 && (
          <div>
            <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-2">Object Similarity Matrix</h2>
            <p className="text-xs text-[#666] mb-3">Objects with similarity &gt; 0.3 receive identity protection during tracking.</p>
            <div className="rounded-xl border border-[#2a2a2a] bg-[#111] divide-y divide-[#1e1e1e]">
              {simPairs.map(({ a, b, score }) => {
                const objA = objects.find(o => o.id === a)
                const objB = objects.find(o => o.id === b)
                return (
                  <div key={`${a}_${b}`} className="flex items-center gap-3 px-4 py-2.5">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      {objA && <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: objA.color }} />}
                      <span className="text-sm text-[#ddd] truncate">{objA?.name ?? `Object ${a}`}</span>
                    </div>
                    <span className="text-xs text-[#555]">↔</span>
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      {objB && <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: objB.color }} />}
                      <span className="text-sm text-[#ddd] truncate">{objB?.name ?? `Object ${b}`}</span>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className="text-xs font-mono font-semibold" style={{ color: scoreColor(score) }}>{score.toFixed(2)}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full font-medium" style={{ background: `${scoreColor(score)}22`, color: scoreColor(score) }}>{scoreLabel(score)}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Confusion windows */}
        {(confusion_windows ?? []).length > 0 && (
          <div>
            <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-2">
              Confusion Windows ({confusion_windows.length})
            </h2>
            <div className="space-y-2">
              {confusion_windows.map((win: any, i: number) => {
                const winObjects = win.obj_ids.map((id: string) => objects.find(o => o.id === id)).filter(Boolean)
                return (
                  <div key={i} className="rounded-xl border border-[#2a2a2a] bg-[#111] p-3 flex items-start gap-3">
                    <div className="w-24 flex-shrink-0">
                      <div className="h-2 bg-[#2a2a2a] rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{
                          marginLeft: `${rangeLen > 0 ? ((win.start - startFrame) / rangeLen) * 100 : 0}%`,
                          width: `${rangeLen > 0 ? ((win.end - win.start) / rangeLen) * 100 : 2}%`,
                          background: scoreColor(win.avg_score),
                        }} />
                      </div>
                      <p className="text-[9px] text-[#555] mt-1 font-mono">{win.start}–{win.end}</p>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        {winObjects.map((obj: any) => obj && (
                          <span key={obj.id} className="flex items-center gap-1 text-xs">
                            <span className="w-2 h-2 rounded-full" style={{ background: obj.color }} />
                            {obj.name}
                          </span>
                        ))}
                      </div>
                      <p className="text-[10px] text-[#666] mt-1">
                        Avg confusion: <span style={{ color: scoreColor(win.avg_score) }}>{win.avg_score.toFixed(3)}</span>
                        {' · '}{win.end - win.start + 1} frames
                      </p>
                    </div>
                    <button onClick={() => setCurrentFrame(win.start)} className="text-xs text-blue-400 hover:text-blue-300 flex-shrink-0 mt-0.5">Jump</button>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {(confusion_windows ?? []).length === 0 && (
          <div className="rounded-xl border border-[#2a2a2a] bg-[#111] p-6 text-center">
            <p className="text-sm text-emerald-400 font-medium mb-1">No confusion windows detected</p>
            <p className="text-xs text-[#666]">Identity confusion did not exceed the threshold during propagation.</p>
          </div>
        )}

      </div>
    </div>
  )
}
