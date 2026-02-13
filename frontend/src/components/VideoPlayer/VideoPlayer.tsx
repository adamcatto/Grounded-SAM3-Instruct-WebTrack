import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Play, Pause, Volume2, VolumeX, Maximize, SkipBack, SkipForward } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { videoSourceUrl } from '../../api/client'

/**
 * VideoPlayer — a standalone MP4 video player tab.
 *
 * Uses a plain HTML5 <video> element pointed at the backend's video source
 * endpoint (bypasses Vite proxy so HTTP Range requests work).
 *
 * Has its own transport controls: play/pause, scrubber, skip +/-10 frames,
 * speed control, volume, fullscreen.  Syncs the current frame number
 * back to the Zustand store so the timeline stays in step.
 */
export default function VideoPlayer() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { project, currentVideoId, currentFrame, setCurrentFrame } = store

  const videoElRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(true)
  const [volume, setVolume] = useState(0.8)
  const [played, setPlayed] = useState(0)       // 0-1 fraction
  const [duration, setDuration] = useState(0)   // seconds
  const [seeking, setSeeking] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [hoveredTime, setHoveredTime] = useState<number | null>(null)
  const [hoverX, setHoverX] = useState(0)
  const [ready, setReady] = useState(false)

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const fps = video?.fps || 30

  // Sync store.currentFrame -> player seek (only when paused & user scrubs)
  const lastSyncedFrame = useRef(currentFrame)
  useEffect(() => {
    const el = videoElRef.current
    if (playing || !el || currentFrame === lastSyncedFrame.current) return
    lastSyncedFrame.current = currentFrame
    const target = currentFrame / fps
    if (Math.abs(el.currentTime - target) > 0.5 / fps) {
      el.currentTime = target
    }
  }, [currentFrame, fps, playing])

  // Play / Pause
  useEffect(() => {
    const el = videoElRef.current
    if (!el) return
    if (playing) {
      el.playbackRate = playbackRate
      const p = el.play()
      if (p) p.catch(() => setPlaying(false))
    } else {
      el.pause()
    }
  }, [playing, playbackRate])

  // Update playback rate live
  useEffect(() => {
    const el = videoElRef.current
    if (el && playing) el.playbackRate = playbackRate
  }, [playbackRate, playing])

  // Volume / mute
  useEffect(() => {
    const el = videoElRef.current
    if (!el) return
    el.muted = muted
    el.volume = volume
  }, [muted, volume])

  // Scrubber handlers
  const handleSeekChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setPlayed(parseFloat(e.target.value))
  }, [])

  const handleSeekMouseDown = useCallback(() => setSeeking(true), [])

  const handleSeekMouseUp = useCallback((e: React.MouseEvent<HTMLInputElement>) => {
    setSeeking(false)
    const val = parseFloat((e.target as HTMLInputElement).value)
    const el = videoElRef.current
    if (el && el.duration) {
      el.currentTime = val * el.duration
    }
  }, [])

  const skipFrames = useCallback((n: number) => {
    const el = videoElRef.current
    if (!el) return
    el.currentTime = Math.max(0, el.currentTime + n / fps)
  }, [fps])

  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      el.requestFullscreen()
    }
  }, [])

  const cycleSpeed = useCallback(() => {
    const speeds = [0.25, 0.5, 1, 1.5, 2, 4]
    const idx = speeds.indexOf(playbackRate)
    setPlaybackRate(speeds[(idx + 1) % speeds.length])
  }, [playbackRate])

  // Progress bar hover
  const onBarHover = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const frac = (e.clientX - rect.left) / rect.width
    setHoveredTime(frac * duration)
    setHoverX(e.clientX - rect.left)
  }, [duration])

  const onBarLeave = useCallback(() => setHoveredTime(null), [])

  function fmtTime(secs: number) {
    const m = Math.floor(secs / 60)
    const s = Math.floor(secs % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  // Empty state
  if (!video) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0d0d0d]">
        <p className="text-sm text-[#666]">No video selected</p>
      </div>
    )
  }

  const videoSrc = videoSourceUrl(pid, vid)
  const currentTime = played * duration
  const currentFrameNum = Math.round(currentTime * fps)

  return (
    <div ref={containerRef} className="flex-1 flex flex-col bg-black overflow-hidden">
      {/* Player area */}
      <div className="flex-1 relative flex items-center justify-center min-h-0">
        <video
          ref={videoElRef}
          src={videoSrc}
          muted={muted}
          preload="auto"
          playsInline
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            background: '#000',
          }}
          onLoadedMetadata={() => {
            const el = videoElRef.current
            if (el) {
              setDuration(el.duration)
              setReady(true)
            }
          }}
          onTimeUpdate={() => {
            const el = videoElRef.current
            if (!el || seeking) return
            const frac = el.duration ? el.currentTime / el.duration : 0
            setPlayed(frac)
            const frame = Math.round(el.currentTime * fps)
            lastSyncedFrame.current = frame
            setCurrentFrame(frame)
          }}
          onEnded={() => setPlaying(false)}
          onDurationChange={() => {
            const el = videoElRef.current
            if (el) setDuration(el.duration)
          }}
          onError={(e) => {
            const el = e.currentTarget
            const err = el.error
            console.error(
              '[VideoPlayer] error:',
              err ? `code=${err.code} message="${err.message}"` : 'unknown',
              'src=', el.currentSrc,
              'networkState=', el.networkState,
              'readyState=', el.readyState,
            )
          }}
        />
        {/* Loading overlay */}
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <div className="text-sm text-[#999] animate-pulse">Loading video...</div>
          </div>
        )}
      </div>

      {/* Controls bar */}
      <div className="flex-shrink-0 bg-[#1a1a1a] border-t border-[#333] px-4 py-2 flex flex-col gap-2">
        {/* Progress bar */}
        <div
          className="relative group cursor-pointer"
          onMouseMove={onBarHover}
          onMouseLeave={onBarLeave}
        >
          {hoveredTime !== null && (
            <div
              className="absolute -top-8 bg-black/80 text-[#ccc] text-[11px] font-mono px-1.5 py-0.5 rounded pointer-events-none"
              style={{ left: hoverX, transform: 'translateX(-50%)' }}
            >
              {fmtTime(hoveredTime)} &middot; #{Math.round(hoveredTime * fps)}
            </div>
          )}
          <input
            type="range"
            min={0}
            max={0.999999}
            step="any"
            value={played}
            onChange={handleSeekChange}
            onMouseDown={handleSeekMouseDown}
            onMouseUp={handleSeekMouseUp}
            className="w-full h-1.5 appearance-none bg-[#333] rounded-full cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:cursor-pointer
              [&::-webkit-slider-thumb]:shadow-[0_0_4px_rgba(59,130,246,0.5)]
              [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:rounded-full
              [&::-moz-range-thumb]:bg-blue-500 [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:cursor-pointer"
          />
          <div
            className="absolute top-[7px] left-0 h-1.5 bg-blue-500/40 rounded-full pointer-events-none"
            style={{ width: `${played * 100}%` }}
          />
        </div>

        {/* Button row */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => skipFrames(-10)}
            className="text-[#999] hover:text-white transition-colors"
            title="Back 10 frames"
          >
            <SkipBack size={16} />
          </button>

          <button
            onClick={() => setPlaying(!playing)}
            className="text-white hover:text-blue-400 transition-colors"
            title={playing ? 'Pause' : 'Play'}
          >
            {playing ? <Pause size={20} /> : <Play size={20} />}
          </button>

          <button
            onClick={() => skipFrames(10)}
            className="text-[#999] hover:text-white transition-colors"
            title="Forward 10 frames"
          >
            <SkipForward size={16} />
          </button>

          <span className="text-xs font-mono text-[#999] select-none min-w-[140px]">
            {fmtTime(currentTime)} / {fmtTime(duration)} &middot; #{currentFrameNum}
          </span>

          <div className="flex-1" />

          <button
            onClick={cycleSpeed}
            className="text-xs font-mono text-[#999] hover:text-white px-2 py-0.5 rounded border border-[#444] transition-colors"
            title="Playback speed"
          >
            {playbackRate}&times;
          </button>

          <button
            onClick={() => setMuted(!muted)}
            className="text-[#999] hover:text-white transition-colors"
            title={muted ? 'Unmute' : 'Mute'}
          >
            {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>

          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={muted ? 0 : volume}
            onChange={(e) => {
              setVolume(parseFloat(e.target.value))
              if (muted) setMuted(false)
            }}
            className="w-16 h-1 appearance-none bg-[#444] rounded cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:h-2.5
              [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#ccc] [&::-webkit-slider-thumb]:cursor-pointer
              [&::-moz-range-thumb]:w-2.5 [&::-moz-range-thumb]:h-2.5 [&::-moz-range-thumb]:rounded-full
              [&::-moz-range-thumb]:bg-[#ccc] [&::-moz-range-thumb]:border-none"
          />

          <button
            onClick={toggleFullscreen}
            className="text-[#999] hover:text-white transition-colors"
            title="Fullscreen"
          >
            <Maximize size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}
