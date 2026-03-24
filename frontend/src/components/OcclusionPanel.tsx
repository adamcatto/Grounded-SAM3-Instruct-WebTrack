import React, { useEffect, useState, useCallback } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import { getOverlaps, type OcclusionWindow } from '../api/client'
import { Layers, RefreshCw, ArrowLeftRight, CheckCircle, Clock } from 'lucide-react'

// ── Helpers ──────────────────────────────────────────────────────────────────

function windowStatusLabel(w: OcclusionWindow): string {
  if (w.end_frame === -1) return 'Active (not resolved)'
  if (w.swap_detected) return 'Swap detected'
  if (w.corrected) return 'Corrected'
  return 'No swap'
}

function windowStatusColor(w: OcclusionWindow): string {
  if (w.end_frame === -1) return '#f59e0b'   // amber — still active
  if (w.swap_detected) return '#ef4444'       // red — swap
  return '#22c55e'                            // green — clean
}

// ── Timeline bar for a single window ─────────────────────────────────────────

interface WindowBarProps {
  w: OcclusionWindow
  totalFrames: number
  currentFrame: number
  onJump: (f: number) => void
}

function WindowBar({ w, totalFrames, currentFrame, onJump }: WindowBarProps) {
  const safeTotal = Math.max(1, totalFrames - 1)
  const left = (w.onset_frame / safeTotal) * 100
  const end = w.end_frame === -1 ? totalFrames - 1 : w.end_frame
  const width = Math.max(0.5, ((end - w.onset_frame) / safeTotal) * 100)
  const cursorPct = (currentFrame / safeTotal) * 100
  const color = windowStatusColor(w)

  return (
    <div
      className="relative w-full rounded overflow-hidden cursor-pointer"
      style={{ height: 12, background: '#1a1a1a' }}
      onClick={e => {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
        const pct = (e.clientX - rect.left) / rect.width
        onJump(Math.round(pct * totalFrames))
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: `${left}%`,
          width: `${width}%`,
          top: 0,
          bottom: 0,
          background: color,
          opacity: 0.7,
        }}
      />
      {/* Playhead */}
      <div
        style={{
          position: 'absolute',
          left: `${cursorPct}%`,
          top: 0,
          bottom: 0,
          width: 1,
          background: 'rgba(255,255,255,0.8)',
          pointerEvents: 'none',
        }}
      />
    </div>
  )
}

// ── Single window card ────────────────────────────────────────────────────────

interface WindowCardProps {
  w: OcclusionWindow
  index: number
  totalFrames: number
  currentFrame: number
  objNames: Record<string, string>
  onJump: (f: number) => void
}

function WindowCard({ w, index, totalFrames, currentFrame, objNames, onJump }: WindowCardProps) {
  const [expanded, setExpanded] = useState(false)
  const color = windowStatusColor(w)
  const status = windowStatusLabel(w)
  const duration = w.end_frame === -1 ? '?' : String(w.end_frame - w.onset_frame + 1)
  const nameA = objNames[String(w.pair[0])] ?? `obj ${w.pair[0]}`
  const nameB = objNames[String(w.pair[1])] ?? `obj ${w.pair[1]}`

  return (
    <div
      style={{
        border: `1px solid ${color}44`,
        borderRadius: 6,
        marginBottom: 8,
        background: '#111',
        overflow: 'hidden',
      }}
    >
      {/* Header row */}
      <div
        style={{ padding: '8px 10px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}
        onClick={() => setExpanded(x => !x)}
      >
        <Layers size={13} color={color} />
        <span style={{ color, fontWeight: 600, fontSize: 12 }}>
          Window {index + 1}
        </span>
        <span style={{ color: '#aaa', fontSize: 11 }}>
          {nameA} ↔ {nameB}
        </span>
        <div style={{ flex: 1 }} />
        {w.swap_detected && <ArrowLeftRight size={12} color="#ef4444" aria-label="Identity swap detected" />}
        {w.corrected && <CheckCircle size={12} color="#22c55e" aria-label="Corrected" />}
        {w.end_frame === -1 && <Clock size={12} color="#f59e0b" aria-label="Still active" />}
        <span style={{ color: '#666', fontSize: 10 }}>{expanded ? '▲' : '▼'}</span>
      </div>

      {/* Timeline bar */}
      <div style={{ padding: '0 10px 6px' }}>
        <WindowBar w={w} totalFrames={totalFrames} currentFrame={currentFrame} onJump={onJump} />
      </div>

      {/* Summary row */}
      <div style={{ padding: '0 10px 8px', display: 'flex', gap: 16, fontSize: 11, color: '#aaa' }}>
        <span>
          Frames{' '}
          <span
            style={{ color: '#ccc', cursor: 'pointer', textDecoration: 'underline dotted' }}
            onClick={() => onJump(w.onset_frame)}
          >
            {w.onset_frame}
          </span>
          {' – '}
          {w.end_frame === -1
            ? <span style={{ color: '#f59e0b' }}>ongoing</span>
            : <span
                style={{ color: '#ccc', cursor: 'pointer', textDecoration: 'underline dotted' }}
                onClick={() => onJump(w.end_frame)}
              >
                {w.end_frame}
              </span>
          }
          {' '}({duration} frames)
        </span>
        <span style={{ color: windowStatusColor(w) }}>{status}</span>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div style={{ padding: '0 10px 10px', borderTop: '1px solid #222', paddingTop: 8, fontSize: 11 }}>
          {w.swap_detected && w.identity_map && (
            <div style={{ marginBottom: 6 }}>
              <span style={{ color: '#ef4444' }}>Swap map: </span>
              {Object.entries(w.identity_map).map(([obs, true_]) => (
                <span key={obs} style={{ color: '#fca5a5', marginRight: 8 }}>
                  {objNames[obs] ?? `obj ${obs}`} → {objNames[true_] ?? `obj ${true_}`}
                </span>
              ))}
            </div>
          )}
          <div style={{ color: '#666', marginBottom: 4 }}>
            Overlapping frames ({w.frames.length}):
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {w.frames.slice(0, 60).map(f => (
              <span
                key={f}
                style={{
                  padding: '1px 5px',
                  borderRadius: 3,
                  background: f === currentFrame ? '#3b82f6' : '#222',
                  color: f === currentFrame ? '#fff' : '#aaa',
                  cursor: 'pointer',
                  fontSize: 10,
                }}
                onClick={() => onJump(f)}
              >
                {f}
              </span>
            ))}
            {w.frames.length > 60 && (
              <span style={{ color: '#666', fontSize: 10, alignSelf: 'center' }}>
                +{w.frames.length - 60} more
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function OcclusionPanel() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { project, currentVideoId, currentFrame, setCurrentFrame } = store

  const [windows, setWindows] = useState<OcclusionWindow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const totalFrames = video?.num_frames ?? 1

  // Build obj id → name map from video objects
  const objNames: Record<string, string> = {}
  if (video) {
    for (const [id, obj] of Object.entries(video.objects)) {
      objNames[id] = (obj as { name?: string }).name ?? `obj ${id}`
    }
  }

  const load = useCallback(async () => {
    if (!pid || !vid) return
    setLoading(true)
    setError('')
    try {
      const data = await getOverlaps(pid, vid)
      setWindows(data.windows ?? [])
    } catch {
      setError('Failed to load overlap data.')
    } finally {
      setLoading(false)
    }
  }, [pid, vid])

  useEffect(() => { load() }, [load])

  const handleJump = (f: number) => {
    setCurrentFrame(Math.max(0, Math.min(f, totalFrames - 1)))
  }

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: 16, fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
        <Layers size={15} color="#a78bfa" />
        <span style={{ color: '#e2e8f0', fontWeight: 600, fontSize: 13 }}>
          Detected Overlaps
        </span>
        <div style={{ flex: 1 }} />
        <button
          onClick={load}
          disabled={loading}
          style={{
            background: 'none', border: '1px solid #333', borderRadius: 4,
            color: '#aaa', cursor: 'pointer', padding: '3px 8px', fontSize: 11,
            display: 'flex', alignItems: 'center', gap: 4,
          }}
        >
          <RefreshCw size={11} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          Refresh
        </button>
      </div>

      {/* Summary */}
      {!loading && !error && (
        <div style={{ fontSize: 11, color: '#666', marginBottom: 12 }}>
          {windows.length === 0
            ? 'No overlap windows detected in this video.'
            : `${windows.length} overlap window${windows.length !== 1 ? 's' : ''} detected.`
          }
          {windows.some(w => w.swap_detected) && (
            <span style={{ color: '#ef4444', marginLeft: 8 }}>
              {windows.filter(w => w.swap_detected).length} with identity swap.
            </span>
          )}
        </div>
      )}

      {loading && (
        <div style={{ color: '#666', fontSize: 12 }}>Loading…</div>
      )}

      {error && (
        <div style={{ color: '#ef4444', fontSize: 12 }}>{error}</div>
      )}

      {/* Window cards */}
      {!loading && windows.map((w, i) => (
        <WindowCard
          key={`${w.pair[0]}-${w.pair[1]}-${w.onset_frame}`}
          w={w}
          index={i}
          totalFrames={totalFrames}
          currentFrame={currentFrame}
          objNames={objNames}
          onJump={handleJump}
        />
      ))}
    </div>
  )
}
