import React, { useEffect } from 'react'
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { getSavedMask } from '../../api/client'
import ObjectTrackRow from './ObjectTrackRow'

export default function Timeline() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project, currentVideoId,
    currentFrame, setCurrentFrame,
    isPlaying, setPlaying,
    propagationStatus, savedMaskCache, setSavedMask,
    propagationStartFrame,
  } = store

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  // ── Prefetch saved masks when frame changes (post-propagation) ────────────

  useEffect(() => {
    if (
      (propagationStatus === 'done' || propagationStatus === 'running') &&
      !savedMaskCache[currentFrame] &&
      video?.propagated_frames?.includes(currentFrame)
    ) {
      getSavedMask(pid, vid, currentFrame)
        .then(data => {
          if (data.masks) setSavedMask(currentFrame, data.masks)
        })
        .catch(() => {})
    }
  }, [currentFrame, propagationStatus])

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
          onClick={() => setCurrentFrame(Math.max(startFrame, currentFrame - 1))}
          className="btn btn-ghost p-1"
          title="Previous frame"
        >
          <SkipBack size={13} />
        </button>
        <button
          onClick={() => setCurrentFrame(Math.min(total - 1, currentFrame + 1))}
          className="btn btn-ghost p-1"
          title="Next frame"
        >
          <SkipForward size={13} />
        </button>
      </div>

      {/* Object track rows */}
      <ObjectTrackRow />
    </div>
  )
}
