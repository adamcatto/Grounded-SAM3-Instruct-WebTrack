import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Info } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { videoSourceUrl, frameUrl } from '../../api/client'
import AnnotationCanvas from './AnnotationCanvas'

/**
 * FrameViewer — video-based annotation viewer.
 *
 * Stack (bottom to top):
 *   1. <video>           — always visible, shows the MP4 video
 *   2. AnnotationCanvas  — transparent click overlay for point prompts + mask rendering
 *
 * When the user is annotating (paused), the video stays paused on the current
 * frame.  Clicking on the canvas extracts that frame on the backend and runs
 * single-frame segmentation.
 */
export default function FrameViewer() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { project, currentVideoId, currentFrame, setCurrentFrame, isPlaying, setPlaying, pointMode } = store

  const containerRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [showTip, setShowTip] = useState(true)

  // Prevent feedback loop: video timeupdate -> setCurrentFrame -> seek effect
  const videoIsDriving = useRef(false)
  const fpsRef = useRef(video?.fps || 30)

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const fps = video?.fps || 30

  useEffect(() => { fpsRef.current = fps }, [fps])

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

  // ── Reload video element when video changes ───────────────────────────────

  useEffect(() => {
    const el = videoRef.current
    if (el) el.load()
  }, [vid])

  // ── Play / Pause ──────────────────────────────────────────────────────────

  useEffect(() => {
    const el = videoRef.current
    if (!el) return
    if (isPlaying) {
      const p = el.play()
      if (p) p.catch(err => {
        console.error('[FrameViewer] play() rejected:', err)
        setPlaying(false)
      })
    } else {
      el.pause()
    }
  }, [isPlaying, setPlaying])

  // ── Seek video when store.currentFrame changes (scrubber / skip) ──────────

  useEffect(() => {
    const el = videoRef.current
    // Only seek when paused — while playing, the video drives the frame
    if (!el || videoIsDriving.current || isPlaying) return
    const target = currentFrame / fps
    if (Math.abs(el.currentTime - target) > 0.5 / fps) {
      el.currentTime = target
    }
  }, [currentFrame, fps, isPlaying])

  // ── Video event handlers ──────────────────────────────────────────────────

  const onTimeUpdate = useCallback(() => {
    const el = videoRef.current
    // Only let the video drive currentFrame during actual playback.
    // While paused (including during seeks), onSeeked handles the update.
    if (!el || el.paused) return
    const frame = Math.floor(el.currentTime * fpsRef.current)
    videoIsDriving.current = true
    setCurrentFrame(frame)
    requestAnimationFrame(() => { videoIsDriving.current = false })
  }, [setCurrentFrame])

  const onEnded = useCallback(() => setPlaying(false), [setPlaying])

  // onSeeked intentionally omitted: the video element has no native controls
  // (pointerEvents:none) so there is no way for a user-initiated seek to
  // diverge from the store.  Reading back el.currentTime after a programmatic
  // seek causes I-frame snapping to silently move currentFrame by ±1.
  const onSeeked = undefined

  const onError = useCallback((e: React.SyntheticEvent<HTMLVideoElement>) => {
    const el = e.currentTarget
    const err = el.error
    console.error(
      '[FrameViewer] video error:',
      err ? `code=${err.code} message="${err.message}"` : 'unknown',
      'src=', el.currentSrc,
      'networkState=', el.networkState,
      'readyState=', el.readyState,
    )
  }, [])

  // ── Arrow key frame navigation ────────────────────────────────────────────

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      e.preventDefault()
      const delta = e.key === 'ArrowRight' ? store.frameJump : -store.frameJump
      const minFrame = store.propagationStartFrame
      const maxFrame = (video?.num_frames ?? 1) - 1
      const next = Math.max(minFrame, Math.min(maxFrame, currentFrame + delta))
      setCurrentFrame(next)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentFrame, video, setCurrentFrame, store.propagationStartFrame])

  // ── Helpers ───────────────────────────────────────────────────────────────

  function fmt(frame: number, fps: number) {
    const s = Math.floor(frame / fps)
    return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`
  }

  // ── Empty state ───────────────────────────────────────────────────────────

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

  const videoSrc = videoSourceUrl(pid, vid)

  return (
    <div
      ref={containerRef}
      className="flex-1 relative bg-black overflow-hidden flex items-center justify-center"
    >
      {dimensions.width > 0 && (
        <div
          style={{
            position: 'relative',
            width: dimensions.width,
            height: dimensions.height,
            flexShrink: 0,
          }}
        >
          {/* Layer 1: HTML5 video — always visible so it acts as an instant
              placeholder while the JPEG frame (layer 1b) loads from the backend.
              During playback it IS the primary display. When paused, the JPEG
              overlays it once it arrives; until then the video shows the last
              seeked frame without any network round-trip. */}
          <video
            ref={videoRef}
            style={{
              position: 'absolute', top: 0, left: 0,
              width: '100%', height: '100%',
              objectFit: 'contain',
              userSelect: 'none', pointerEvents: 'none',
            }}
            muted
            playsInline
            preload="auto"
            onTimeUpdate={onTimeUpdate}
            onSeeked={onSeeked}
            onEnded={onEnded}
            onError={onError}
          >
            <source src={videoSrc} type="video/mp4" />
          </video>

          {/* Layer 1b: JPEG frame (shown when paused — instant vs. video seek latency) */}
          {!isPlaying && pid && vid && (
            <img
              src={frameUrl(pid, vid, currentFrame)}
              style={{
                position: 'absolute', top: 0, left: 0,
                width: '100%', height: '100%',
                objectFit: 'contain',
                userSelect: 'none', pointerEvents: 'none',
              }}
              draggable={false}
              alt=""
            />
          )}

          {/* Layer 2: Annotation canvas */}
          <AnnotationCanvas width={dimensions.width} height={dimensions.height} videoRef={videoRef} />
        </div>
      )}

      {/* Frame info (top-right) */}
      <div className="absolute top-3 right-3 bg-black/60 rounded-lg px-2 py-1 text-xs text-[#ccc] font-mono pointer-events-none flex items-center gap-1.5">
        <span>{fmt(currentFrame, fps)}</span>
        <span className="text-[#666]">&middot;</span>
        <span>#{currentFrame}</span>
      </div>

      {/* Tip tooltip */}
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
