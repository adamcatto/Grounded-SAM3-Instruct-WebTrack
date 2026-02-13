import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Info } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { videoSourceUrl, frameUrl } from '../../api/client'
import AnnotationCanvas from './AnnotationCanvas'

export default function FrameViewer() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { project, currentVideoId, currentFrame, setCurrentFrame, isPlaying, setPlaying, pointMode } = store

  const containerRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [videoReady, setVideoReady] = useState(false)
  const [showTip, setShowTip] = useState(true)

  // Suppress store→video seek while the video element itself is driving frames
  const seekLock = useRef(false)

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const fps = video?.fps || 30

  // ── Compute displayed dimensions ──────────────────────────────────────────

  useEffect(() => {
    if (!video || !containerRef.current) return

    const updateDims = () => {
      const container = containerRef.current
      if (!container || !video) return
      const cW = container.clientWidth
      const cH = container.clientHeight
      const aspect = video.width / video.height
      let w = cW
      let h = cW / aspect
      if (h > cH) { h = cH; w = h * aspect }
      setDimensions({ width: Math.floor(w), height: Math.floor(h) })
    }

    updateDims()
    const ro = new ResizeObserver(updateDims)
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [video])

  // ── Reset video element when video changes ────────────────────────────────

  useEffect(() => {
    setVideoReady(false)
  }, [vid])

  // ── Sync video element play/pause with store ──────────────────────────────

  useEffect(() => {
    const el = videoRef.current
    if (!el || !videoReady) return
    if (isPlaying) {
      el.play().catch(() => {})
    } else {
      el.pause()
    }
  }, [isPlaying, videoReady])

  // ── When store.currentFrame changes (e.g. scrubber), seek video ───────────

  useEffect(() => {
    const el = videoRef.current
    if (!el || !videoReady || seekLock.current) return
    const targetTime = currentFrame / fps
    // Only seek if the difference is significant (avoids feedback loops)
    if (Math.abs(el.currentTime - targetTime) > 0.5 / fps) {
      el.currentTime = targetTime
    }
  }, [currentFrame, videoReady, fps])

  // ── Video event handlers ──────────────────────────────────────────────────

  const handleTimeUpdate = useCallback(() => {
    const el = videoRef.current
    if (!el) return
    const frame = Math.round(el.currentTime * fps)
    seekLock.current = true
    setCurrentFrame(frame)
    requestAnimationFrame(() => { seekLock.current = false })
  }, [fps, setCurrentFrame])

  const handleLoadedData = useCallback(() => {
    setVideoReady(true)
    const el = videoRef.current
    if (el) {
      el.currentTime = currentFrame / fps
    }
  }, [currentFrame, fps])

  const handleEnded = useCallback(() => {
    setPlaying(false)
  }, [setPlaying])

  // ── Time formatting ───────────────────────────────────────────────────────

  function formatTime(frame: number, fps: number): string {
    const secs = Math.floor(frame / fps)
    const m = Math.floor(secs / 60)
    const s = secs % 60
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  if (!video) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0d0d0d]">
        <div className="text-center max-w-sm px-6">
          <p className="text-xl font-semibold text-white mb-3">
            Click an object in the video to start
          </p>
          <p className="text-sm text-[#666] leading-relaxed">
            You'll be able to use SAM3 to label and track objects throughout the video.
          </p>
        </div>
      </div>
    )
  }

  const src = videoSourceUrl(pid, vid)
  const fallbackSrc = frameUrl(pid, vid, currentFrame)

  return (
    <div
      ref={containerRef}
      className="flex-1 relative bg-black overflow-hidden flex items-center justify-center"
    >
      {/* Video element + frame fallback + annotation overlay */}
      {dimensions.width > 0 && (
        <div
          style={{
            position: 'relative',
            width: dimensions.width,
            height: dimensions.height,
            flexShrink: 0,
          }}
        >
          {/* Fallback: frame image shown instantly while video buffers */}
          {!videoReady && (
            <img
              src={fallbackSrc}
              alt={`Frame ${currentFrame}`}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                userSelect: 'none',
                pointerEvents: 'none',
              }}
              draggable={false}
            />
          )}

          {/* HTML5 video element — streams via Range requests */}
          <video
            ref={videoRef}
            src={src}
            style={{
              display: videoReady ? 'block' : 'none',
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              userSelect: 'none',
              pointerEvents: 'none',
            }}
            muted
            playsInline
            preload="metadata"
            onLoadedData={handleLoadedData}
            onTimeUpdate={handleTimeUpdate}
            onEnded={handleEnded}
          />

          {/* Annotation canvas overlay — show immediately so clicks work */}
          <AnnotationCanvas width={dimensions.width} height={dimensions.height} />
        </div>
      )}

      {/* Frame timestamp (top-right) */}
      <div className="absolute top-3 right-3 bg-black/60 rounded-lg px-2 py-1 text-xs text-[#ccc] font-mono pointer-events-none">
        {formatTime(currentFrame, fps)}
      </div>

      {/* Tip tooltip (top-right area) */}
      {showTip && video && Object.keys(video.objects).length === 1 && (
        <div className="absolute top-12 right-3 bg-[#1a1a1a] border border-[#333] rounded-xl p-3 max-w-[220px] shadow-xl">
          <div className="flex items-start gap-2">
            <Info size={14} className="text-blue-400 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-[#ccc] leading-relaxed">
              Tip: Add a new object by clicking on it in the video.
            </p>
          </div>
          <div className="flex justify-between mt-2">
            <a href="#" className="text-xs text-blue-400 hover:underline">Learn more</a>
            <button
              onClick={() => setShowTip(false)}
              className="text-xs text-[#888] hover:text-white px-2 py-0.5 border border-[#444] rounded"
            >
              Ok
            </button>
          </div>
        </div>
      )}

      {/* Cursor hint */}
      {pointMode && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/70 rounded-full px-4 py-1.5 text-xs text-[#ccc] pointer-events-none">
          {pointMode === 'add' ? 'Click to add (+) point' : 'Click to add (−) exclusion point'}
        </div>
      )}
    </div>
  )
}
