import React, { useEffect, useRef, useState, useCallback } from 'react'
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

function ConfusionHeatmap({ perFrame, total, startFrame, currentFrame, onJump }: HeatmapProps) {
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
}

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

function FrameRow({ fidx, data, objects, pid, vid, isExpanded, onToggleExpand, onJump }: FrameRowProps) {
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
    <div className="border-b border-[#1e1e1e] last:border-b-0">
      {/* Row header */}
      <div
        className="flex items-center gap-2 px-3 py-2 hover:bg-[#1a1a1a] cursor-pointer select-none"
        onClick={onToggleExpand}
      >
        <span className="text-[#444] flex-shrink-0">
          {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </span>
        <span className="font-mono text-xs text-[#888] w-14 flex-shrink-0">{fidx}</span>
        <div className="flex-1 flex items-center gap-2">
          <div
            className="h-1.5 flex-1 rounded-full bg-[#1a1a1a] overflow-hidden"
          >
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${Math.min(100, score * 100)}%`,
                background: scoreColor(score),
                opacity: 0.7,
              }}
            />
          </div>
          <span
            className="text-[10px] font-mono flex-shrink-0"
            style={{ color: scoreColor(score) }}
          >
            {score.toFixed(3)}
          </span>
          <span
            className="text-[9px] px-1 py-0.5 rounded flex-shrink-0"
            style={{ background: `${scoreColor(score)}22`, color: scoreColor(score) }}
          >
            {scoreLabel(score)}
          </span>
        </div>
        {data.confused_objects && data.confused_objects.length > 0 && (
          <span className="text-[9px] text-amber-400 flex-shrink-0">confused</span>
        )}
        {data.temporal_rejections && Object.keys(data.temporal_rejections).length > 0 && (
          <span className="text-[9px] text-blue-400 flex-shrink-0">rejected</span>
        )}
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-4 pb-3 pt-1 bg-[#0a0a0a] space-y-3">
          {/* Jump button */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-[#555]">Frame {fidx}</span>
            <button
              onClick={() => onJump(fidx)}
              className="text-[10px] text-blue-400 hover:text-blue-300"
            >
              Jump to frame
            </button>
          </div>

          {/* Per-object anomaly scores */}
          {data.per_object && Object.keys(data.per_object).length > 0 && (
            <div>
              <p className="text-[9px] text-[#555] uppercase tracking-wide mb-1.5">Per-object anomaly</p>
              <div className="space-y-1">
                {Object.entries(data.per_object).map(([objId, scores]) => {
                  const obj = objects.find(o => o.id === objId)
                  const anomaly = scores.anomaly_score ?? 0
                  return (
                    <div key={objId} className="flex items-center gap-2">
                      <span
                        className="w-2 h-2 rounded-full flex-shrink-0"
                        style={{ background: obj?.color ?? '#888' }}
                      />
                      <span className="text-xs text-[#aaa] flex-1 truncate min-w-0">
                        {obj?.name ?? `Obj ${objId}`}
                      </span>
                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        <div className="w-16 h-1 bg-[#2a2a2a] rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.min(100, (anomaly / 5) * 100)}%`,
                              background: anomaly > 2.5 ? '#ef4444' : anomaly > 1.5 ? '#f59e0b' : '#22c55e',
                            }}
                          />
                        </div>
                        <span className="text-[10px] font-mono text-[#666] w-8 text-right">
                          {anomaly.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Temporal rejections */}
          {data.temporal_rejections && Object.keys(data.temporal_rejections).length > 0 && (
            <div>
              <p className="text-[9px] text-[#555] uppercase tracking-wide mb-1.5">Temporal rejections</p>
              <div className="space-y-1">
                {Object.entries(data.temporal_rejections).map(([objId, reason]) => {
                  const obj = objects.find(o => o.id === objId)
                  return (
                    <div key={objId} className="flex items-center gap-2 text-[10px]">
                      <span
                        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ background: obj?.color ?? '#888' }}
                      />
                      <span className="text-[#888]">{obj?.name ?? `Obj ${objId}`}:</span>
                      <span className="text-blue-300 font-mono">{reason}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Confused objects */}
          {data.confused_objects && data.confused_objects.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-[9px] text-[#555] uppercase tracking-wide">Confused:</span>
              <div className="flex gap-1.5">
                {data.confused_objects.map(objId => {
                  const obj = objects.find(o => o.id === objId)
                  return (
                    <span key={objId} className="flex items-center gap-1 text-[10px] text-amber-300">
                      <span className="w-1.5 h-1.5 rounded-full" style={{ background: obj?.color ?? '#f59e0b' }} />
                      {obj?.name ?? `Obj ${objId}`}
                    </span>
                  )
                })}
              </div>
            </div>
          )}

          {/* Frame preview toggle */}
          <div>
            <button
              onClick={togglePreview}
              className="flex items-center gap-1.5 text-[10px] text-[#666] hover:text-[#aaa] transition-colors"
            >
              {showPreview ? <EyeOff size={11} /> : <Eye size={11} />}
              {showPreview ? 'Hide frame preview' : 'Show frame preview'}
            </button>

            {showPreview && (
              <div className="mt-2">
                {loading ? (
                  <div className="h-16 flex items-center justify-center gap-2 text-[#555] text-xs">
                    <Loader size={12} className="animate-spin" />
                    Loading...
                  </div>
                ) : detail ? (
                  <FramePreviewCanvas detail={detail} />
                ) : (
                  <div className="h-16 flex items-center justify-center text-[#555] text-xs">
                    Failed to load
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function UncertaintyPanel() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { uncertaintyData, currentFrame, setCurrentFrame, propagationStartFrame, project } = store

  const [expandedFrames, setExpandedFrames] = useState<Set<number>>(new Set())

  const pid = project?.id ?? ''
  const vid = video?.id ?? ''

  const toggleExpand = useCallback((fidx: number) => {
    setExpandedFrames(prev => {
      const next = new Set(prev)
      if (next.has(fidx)) next.delete(fidx)
      else next.add(fidx)
      return next
    })
  }, [])

  if (!uncertaintyData || !uncertaintyData.per_frame) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0d0d0d] text-[#555] text-sm">
        No uncertainty data available. Run propagation to generate uncertainty analysis.
      </div>
    )
  }

  const { per_frame, confusion_windows, similarity_matrix } = uncertaintyData

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

  const frameEntries = Object.entries(per_frame)
    .map(([k, v]) => [parseInt(k), v] as [number, typeof v])
    .sort(([a], [b]) => a - b)

  const simPairs: { a: string; b: string; score: number }[] = []
  for (const [key, score] of Object.entries(similarity_matrix ?? {})) {
    const [a, b] = key.split('_')
    simPairs.push({ a, b, score })
  }
  simPairs.sort((x, y) => y.score - x.score)

  const rangeLen = total - 1 - startFrame

  return (
    <div className="flex-1 overflow-y-auto bg-[#0d0d0d] text-[#ccc]">
      <div className="max-w-2xl mx-auto p-6 space-y-8">

        {/* Heatmap */}
        <div>
          <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-2">
            Confusion Heatmap
          </h2>
          <p className="text-xs text-[#666] mb-3">
            Color intensity shows identity confusion per frame. Click to jump.
          </p>
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

        {/* All frames list */}
        {frameEntries.length > 0 && (
          <div>
            <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-1">
              All Frames ({frameEntries.length})
            </h2>
            <p className="text-xs text-[#666] mb-3">
              Click any frame to view its uncertainty metrics. Toggle the eye icon to load the frame image and masks.
            </p>
            <div className="rounded-xl border border-[#2a2a2a] bg-[#111] overflow-hidden">
              {/* Header */}
              <div className="flex items-center gap-2 px-3 py-1.5 bg-[#151515] border-b border-[#2a2a2a]">
                <span className="w-4 flex-shrink-0" />
                <span className="text-[10px] text-[#555] w-14 flex-shrink-0">Frame</span>
                <span className="text-[10px] text-[#555] flex-1">Confusion score</span>
              </div>
              <div className="max-h-[50vh] overflow-y-auto">
                {frameEntries.map(([fidx, data]) => (
                  <FrameRow
                    key={fidx}
                    fidx={fidx}
                    data={data}
                    objects={objects}
                    pid={pid}
                    vid={vid}
                    isExpanded={expandedFrames.has(fidx)}
                    onToggleExpand={() => toggleExpand(fidx)}
                    onJump={setCurrentFrame}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Similarity matrix */}
        {simPairs.length > 0 && (
          <div>
            <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-2">
              Object Similarity Matrix
            </h2>
            <p className="text-xs text-[#666] mb-3">
              Objects with similarity &gt; 0.3 receive identity protection during tracking.
            </p>
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
                      <span className="text-xs font-mono font-semibold" style={{ color: scoreColor(score) }}>
                        {score.toFixed(2)}
                      </span>
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                        style={{ background: `${scoreColor(score)}22`, color: scoreColor(score) }}
                      >
                        {scoreLabel(score)}
                      </span>
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
              {confusion_windows.map((win, i) => {
                const winObjects = win.obj_ids.map(id => objects.find(o => o.id === id)).filter(Boolean)
                return (
                  <div
                    key={i}
                    className="rounded-xl border border-[#2a2a2a] bg-[#111] p-3 flex items-start gap-3"
                  >
                    <div className="w-24 flex-shrink-0">
                      <div className="h-2 bg-[#2a2a2a] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            marginLeft: `${rangeLen > 0 ? ((win.start - startFrame) / rangeLen) * 100 : 0}%`,
                            width: `${rangeLen > 0 ? ((win.end - win.start) / rangeLen) * 100 : 2}%`,
                            background: scoreColor(win.avg_score),
                          }}
                        />
                      </div>
                      <p className="text-[9px] text-[#555] mt-1 font-mono">{win.start}–{win.end}</p>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        {winObjects.map(obj => obj && (
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
                    <button
                      onClick={() => setCurrentFrame(win.start)}
                      className="text-xs text-blue-400 hover:text-blue-300 flex-shrink-0 mt-0.5"
                    >
                      Jump
                    </button>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {(confusion_windows ?? []).length === 0 && frameEntries.length > 0 && (
          <div className="rounded-xl border border-[#2a2a2a] bg-[#111] p-6 text-center">
            <p className="text-sm text-emerald-400 font-medium mb-1">No confusion windows detected</p>
            <p className="text-xs text-[#666]">
              Identity confusion did not exceed the threshold during propagation.
            </p>
          </div>
        )}

      </div>
    </div>
  )
}
