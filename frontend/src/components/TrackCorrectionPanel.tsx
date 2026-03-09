import React, { useEffect, useRef, useState } from 'react'
import { Check, X, Loader, ArrowRightLeft, ChevronLeft, ChevronRight } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import { applyCorrection, rejectCorrection, getRawMasks, getSavedMask } from '../api/client'
import type { CorrectionRecord, MaskData } from '../types'
import { drawMasks } from '../utils/maskUtils'

// ── Status badge ───────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: CorrectionRecord['status'] }) {
  const styles: Record<string, string> = {
    pending: 'bg-amber-400/15 text-amber-400',
    applied: 'bg-emerald-400/15 text-emerald-400',
    rejected: 'bg-[#333] text-[#666]',
  }
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium capitalize ${styles[status]}`}>
      {status}
    </span>
  )
}

// ── Before/After viewer ────────────────────────────────────────────────────────

interface BeforeAfterProps {
  correction: CorrectionRecord
  objects: Record<string, { name: string; color: string }>
  opacity: number
}

function BeforeAfterViewer({ correction, objects, opacity }: BeforeAfterProps) {
  const { savedMaskCache, setSavedMask } = useStore()
  const [frame, setFrame] = useState(correction.swap_onset)
  const [rawMasks, setRawMasks] = useState<Record<number, MaskData>>({})
  const [loading, setLoading] = useState(false)
  const beforeRef = useRef<HTMLCanvasElement>(null)
  const afterRef = useRef<HTMLCanvasElement>(null)
  const store = useStore()
  const video = selectCurrentVideo(store)
  const pid = store.project?.id ?? ''
  const vid = store.currentVideoId ?? ''
  const W = 480
  const H = video ? Math.round(W * video.height / video.width) : 270

  const minF = correction.window_start
  const maxF = correction.window_end
  const rangeLen = maxF - minF

  // Load raw + saved masks for current frame
  useEffect(() => {
    if (!pid || !vid) return
    setLoading(true)
    const currentRaw = rawMasks[frame]
    const currentSaved = savedMaskCache[frame]
    const fetchRaw = currentRaw ? Promise.resolve(currentRaw) :
      getRawMasks(pid, vid, frame).then(d => {
        setRawMasks(prev => ({ ...prev, [frame]: d.masks }))
        return d.masks
      })
    const fetchCurrent = currentSaved ? Promise.resolve(currentSaved) :
      getSavedMask(pid, vid, frame).then(d => {
        if (d.masks) setSavedMask(frame, d.masks)
        return d.masks
      })

    Promise.all([fetchRaw, fetchCurrent])
      .catch(() => {})
      .finally(() => setLoading(false))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame, pid, vid])

  // Draw before canvas
  useEffect(() => {
    const canvas = beforeRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    canvas.width = W; canvas.height = H
    ctx.clearRect(0, 0, W, H)
    const masks = rawMasks[frame]
    if (masks) drawMasks(ctx, masks, W, H, opacity)
  }, [rawMasks, frame, W, H, opacity])

  // Draw after canvas
  useEffect(() => {
    const canvas = afterRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    canvas.width = W; canvas.height = H
    ctx.clearRect(0, 0, W, H)
    const masks = savedMaskCache[frame]
    if (masks) drawMasks(ctx, masks, W, H, opacity)
  }, [savedMaskCache, frame, W, H, opacity])

  return (
    <div className="mt-3 space-y-2">
      {/* Scrubber */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setFrame(f => Math.max(minF, f - 1))}
          className="p-0.5 text-[#555] hover:text-[#aaa]"
        >
          <ChevronLeft size={14} />
        </button>
        <div
          className="flex-1 h-1.5 bg-[#2a2a2a] rounded-full relative cursor-pointer"
          onClick={e => {
            const rect = e.currentTarget.getBoundingClientRect()
            const pct = (e.clientX - rect.left) / rect.width
            setFrame(Math.max(minF, Math.min(maxF, Math.round(pct * rangeLen + minF))))
          }}
        >
          <div
            className="absolute top-0 left-0 h-full bg-white/60 rounded-full pointer-events-none"
            style={{ width: `${rangeLen > 0 ? ((frame - minF) / rangeLen) * 100 : 0}%` }}
          />
          {/* Swap onset marker */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-amber-400 -translate-x-1/2 pointer-events-none"
            style={{ left: `${rangeLen > 0 ? ((correction.swap_onset - minF) / rangeLen) * 100 : 0}%` }}
            title={`Swap onset: frame ${correction.swap_onset}`}
          />
        </div>
        <button
          onClick={() => setFrame(f => Math.min(maxF, f + 1))}
          className="p-0.5 text-[#555] hover:text-[#aaa]"
        >
          <ChevronRight size={14} />
        </button>
        <span className="text-[10px] text-[#666] font-mono w-20 text-right">
          frame {frame} {frame === correction.swap_onset && <span className="text-amber-400">← onset</span>}
        </span>
      </div>

      {/* Side-by-side canvases */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <p className="text-[10px] text-[#666] mb-1 uppercase tracking-wide">Before (original)</p>
          <div className="relative bg-black rounded overflow-hidden" style={{ aspectRatio: `${W}/${H}` }}>
            <canvas ref={beforeRef} className="w-full h-full" />
            {loading && <div className="absolute inset-0 flex items-center justify-center"><Loader size={14} className="animate-spin text-[#555]" /></div>}
          </div>
        </div>
        <div>
          <p className="text-[10px] text-[#666] mb-1 uppercase tracking-wide">After (corrected)</p>
          <div className="relative bg-black rounded overflow-hidden" style={{ aspectRatio: `${W}/${H}` }}>
            <canvas ref={afterRef} className="w-full h-full" />
            {loading && <div className="absolute inset-0 flex items-center justify-center"><Loader size={14} className="animate-spin text-[#555]" /></div>}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Correction card ────────────────────────────────────────────────────────────

interface CardProps {
  record: CorrectionRecord
  objects: Record<string, { name: string; color: string }>
  correctionMethod: 'swap' | 'repropagate'
  maskOpacity: number
  onUpdate: (id: string, status: CorrectionRecord['status']) => void
}

function CorrectionCard({ record, objects, correctionMethod, maskOpacity, onUpdate }: CardProps) {
  const [busy, setBusy] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const store = useStore()
  const pid = store.project?.id ?? ''
  const vid = store.currentVideoId ?? ''

  const objA = objects[record.obj_id_a]
  const objB = objects[record.obj_id_b]

  async function handleApply() {
    setBusy(true)
    try {
      await applyCorrection(pid, vid, record.id, correctionMethod)
      onUpdate(record.id, 'applied')
    } catch (e: unknown) {
      console.error('Apply correction failed:', e)
    } finally {
      setBusy(false)
    }
  }

  async function handleReject() {
    setBusy(true)
    try {
      await rejectCorrection(pid, vid, record.id)
      onUpdate(record.id, 'rejected')
    } catch (e: unknown) {
      console.error('Reject correction failed:', e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rounded-xl border border-[#2a2a2a] bg-[#111] p-3">
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0 space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            {objA && (
              <span className="flex items-center gap-1 text-xs">
                <span className="w-2 h-2 rounded-full" style={{ background: objA.color }} />
                {objA.name}
              </span>
            )}
            <ArrowRightLeft size={10} className="text-[#555]" />
            {objB && (
              <span className="flex items-center gap-1 text-xs">
                <span className="w-2 h-2 rounded-full" style={{ background: objB.color }} />
                {objB.name}
              </span>
            )}
            <StatusBadge status={record.status} />
          </div>
          <p className="text-[10px] text-[#666]">
            Frames {record.window_start}–{record.window_end}
            {' · '}onset {record.swap_onset}
            {' · '}score {record.avg_confusion_score.toFixed(3)}
          </p>
        </div>

        {record.status === 'pending' && (
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={handleApply}
              disabled={busy}
              className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40"
            >
              {busy ? <Loader size={10} className="animate-spin" /> : <Check size={10} />}
              Apply
            </button>
            <button
              onClick={handleReject}
              disabled={busy}
              className="p-1 text-[#555] hover:text-red-400 disabled:opacity-40 rounded"
              title="Reject"
            >
              <X size={13} />
            </button>
          </div>
        )}

        <button
          onClick={() => setExpanded(v => !v)}
          className="text-[10px] text-blue-400 hover:text-blue-300 flex-shrink-0"
        >
          {expanded ? 'Hide' : 'Before/After'}
        </button>
      </div>

      {expanded && (
        <BeforeAfterViewer
          correction={record}
          objects={objects}
          opacity={maskOpacity}
        />
      )}
    </div>
  )
}

// ── Main panel ─────────────────────────────────────────────────────────────────

export default function TrackCorrectionPanel() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { corrections, setCorrections, config, setConfig } = store
  const pid = store.project?.id ?? ''
  const vid = store.currentVideoId ?? ''
  const [applyingAll, setApplyingAll] = useState(false)

  const objects: Record<string, { name: string; color: string }> = {}
  if (video) {
    for (const obj of Object.values(video.objects)) {
      objects[obj.id] = { name: obj.name, color: obj.color }
    }
  }

  const pending = corrections.filter(c => c.status === 'pending')

  function handleUpdate(id: string, status: CorrectionRecord['status']) {
    setCorrections(corrections.map(c => c.id === id ? { ...c, status } : c))
  }

  async function handleApplyAll() {
    if (!pending.length) return
    if (!confirm(`Apply all ${pending.length} pending corrections?`)) return
    setApplyingAll(true)
    for (const rec of pending) {
      try {
        await applyCorrection(pid, vid, rec.id, config.correctionMethod)
        handleUpdate(rec.id, 'applied')
      } catch { /* continue */ }
    }
    setApplyingAll(false)
  }

  if (corrections.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0d0d0d] text-[#555] text-sm">
        No correction records. Run propagation to detect identity swaps.
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto bg-[#0d0d0d] text-[#ccc]">
      <div className="max-w-2xl mx-auto p-6 space-y-6">

        {/* Header controls */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#666]">Method:</span>
            {(['swap', 'repropagate'] as const).map(m => (
              <button
                key={m}
                onClick={() => setConfig({ correctionMethod: m })}
                className={`text-xs px-2.5 py-1 rounded-md border transition-colors ${
                  config.correctionMethod === m
                    ? 'bg-blue-600/20 border-blue-500/60 text-blue-400'
                    : 'border-[#333] text-[#666] hover:border-[#555] hover:text-[#aaa]'
                }`}
              >
                {m === 'swap' ? 'Deterministic swap' : 'Re-propagate'}
              </button>
            ))}
          </div>
          {pending.length > 0 && (
            <button
              onClick={handleApplyAll}
              disabled={applyingAll}
              className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40"
            >
              {applyingAll ? <Loader size={11} className="animate-spin" /> : <Check size={11} />}
              Apply all ({pending.length})
            </button>
          )}
        </div>

        {/* Sorted: pending first, then by window_start */}
        <div className="space-y-2">
          {[...corrections]
            .sort((a, b) => {
              if (a.status === 'pending' && b.status !== 'pending') return -1
              if (a.status !== 'pending' && b.status === 'pending') return 1
              return a.window_start - b.window_start
            })
            .map(rec => (
              <CorrectionCard
                key={rec.id}
                record={rec}
                objects={objects}
                correctionMethod={config.correctionMethod}
                maskOpacity={config.maskOpacity}
                onUpdate={handleUpdate}
              />
            ))}
        </div>

      </div>
    </div>
  )
}
