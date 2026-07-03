import React, { useEffect, useState } from 'react'
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import ObjectTrackRow from './ObjectTrackRow'

function JumpToFrame({ currentFrame, min, max, onJump }: {
  currentFrame: number
  min: number
  max: number
  onJump: (f: number) => void
}) {
  const [value, setValue] = useState<string | null>(null)

  const commit = (raw: string) => {
    const n = parseInt(raw, 10)
    if (!isNaN(n)) onJump(Math.max(min, Math.min(max, n)))
    setValue(null)
  }

  return (
    <input
      type="number"
      min={min}
      max={max}
      value={value ?? currentFrame}
      onChange={e => setValue(e.target.value)}
      onKeyDown={e => {
        if (e.key === 'Enter') { commit((e.target as HTMLInputElement).value); (e.target as HTMLInputElement).blur() }
        if (e.key === 'Escape') { setValue(null); (e.target as HTMLInputElement).blur() }
      }}
      onBlur={e => commit(e.target.value)}
      onFocus={e => { setValue(String(currentFrame)); e.target.select() }}
      title="Jump to frame"
      className="w-16 bg-[#1a1a1a] border border-[#333] rounded px-1.5 py-0.5 text-xs font-mono text-[#ccc] text-center focus:outline-none focus:border-[#555]"
      style={{ MozAppearance: 'textfield' } as React.CSSProperties}
    />
  )
}

export default function Timeline() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    currentFrame, setCurrentFrame,
    isPlaying, setPlaying,
    propagationStartFrame,
    frameJump, setFrameJump,
  } = store

  // ── Time formatting ───────────────────────────────────────────────────────

  function formatTime(frame: number, fps: number): string {
    const secs = Math.floor(frame / fps)
    const m = Math.floor(secs / 60)
    const s = secs % 60
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  if (!video) return null

  const total = video.num_frames
  const fps = video.fps || 30
  const startFrame = propagationStartFrame
  const rangeEnd = total - 1
  const rangeLen = rangeEnd - startFrame
  const progress = rangeLen > 0 ? ((currentFrame - startFrame) / rangeLen) * 100 : 0

  return (
    <div className="flex-shrink-0 bg-[#0d0d0d]">
      {/* Controls row */}
      <div className="flex items-center gap-3 px-3 py-2 border-b border-[#1a1a1a]">
        {/* Play/Pause */}
        <button
          onClick={() => setPlaying(!isPlaying)}
          className="w-8 h-8 flex items-center justify-center rounded-full bg-white text-black hover:bg-[#e0e0e0] transition-colors flex-shrink-0"
        >
          {isPlaying ? <Pause size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" />}
        </button>

        {/* Timestamp + frame index */}
        <span className="text-xs font-mono text-[#888] flex-shrink-0">
          {formatTime(currentFrame, fps)}
          <span className="text-[#555] ml-1">#{currentFrame}</span>
        </span>

        {/* Progress scrubber */}
        <div
          className="flex-1 h-1 bg-[#2a2a2a] rounded-full cursor-pointer relative group"
          onClick={e => {
            const rect = e.currentTarget.getBoundingClientRect()
            const pct = (e.clientX - rect.left) / rect.width
            setCurrentFrame(Math.max(startFrame, Math.min(rangeEnd, Math.round(pct * rangeLen + startFrame))))
          }}
        >
          <div
            className="absolute top-0 left-0 h-full bg-white/60 rounded-full pointer-events-none"
            style={{ width: `${progress}%` }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-md pointer-events-none -translate-x-1/2"
            style={{ left: `${progress}%` }}
          />
        </div>

        {/* Total time */}
        <span className="text-xs font-mono text-[#555] flex-shrink-0">
          {formatTime(rangeEnd, fps)}
        </span>

        {/* Frame skip buttons */}
        <button
          onClick={() => setCurrentFrame(Math.max(startFrame, currentFrame - frameJump))}
          className="btn btn-ghost p-1"
          title={`Previous ${frameJump} frame${frameJump !== 1 ? 's' : ''}`}
        >
          <SkipBack size={13} />
        </button>
        <button
          onClick={() => setCurrentFrame(Math.min(total - 1, currentFrame + frameJump))}
          className="btn btn-ghost p-1"
          title={`Next ${frameJump} frame${frameJump !== 1 ? 's' : ''}`}
        >
          <SkipForward size={13} />
        </button>

        {/* Jump amount input */}
        <div className="flex items-center gap-1" title="Frame jump amount (arrow keys / skip buttons)">
          <span className="text-[10px] text-[#444]">×</span>
          <input
            type="number"
            min={1}
            max={9999}
            value={frameJump}
            onChange={e => { const n = parseInt(e.target.value, 10); if (!isNaN(n) && n >= 1) setFrameJump(n) }}
            onFocus={e => e.target.select()}
            className="w-10 bg-[#1a1a1a] border border-[#2a2a2a] rounded px-1 py-0.5 text-[10px] font-mono text-[#888] text-center focus:outline-none focus:border-[#444]"
            style={{ MozAppearance: 'textfield' } as React.CSSProperties}
          />
        </div>

        {/* Jump to frame */}
        <JumpToFrame
          currentFrame={currentFrame}
          min={startFrame}
          max={total - 1}
          onJump={setCurrentFrame}
        />
      </div>

      {/* Object track rows */}
      <ObjectTrackRow />
    </div>
  )
}
