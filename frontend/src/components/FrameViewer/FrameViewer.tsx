import React, { useEffect, useRef, useState, useCallback } from 'react'
import { GripVertical, Info } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { videoSourceUrl, frameUrl } from '../../api/client'
import AnnotationCanvas from './AnnotationCanvas'

function targetIsTypingContext(target: EventTarget | null): boolean {
  const el = target instanceof HTMLElement ? target : null
  if (!el) return false
  if (el.isContentEditable) return true
  const tag = el.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
  return false
}

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
  const {
    project,
    currentVideoId,
    currentFrame,
    setCurrentFrame,
    isPlaying,
    setPlaying,
    pointMode,
    anchorPhase,
    anchorFrames,
    currentAnchorIndex,
    anchorRemainderInferencing,
    anchorRemainderAwaitingCommit,
    currentObjectId,
  } = store

  const redoModGlyph = /^Mac|^iPod|^iPhone/i.test(
    typeof navigator !== 'undefined' ? navigator.platform : '',
  )
    ? '⌘'
    : 'Ctrl'

  const containerRef = useRef<HTMLDivElement>(null)
  const frameStackRef = useRef<HTMLDivElement>(null)
  const anchorReturnPanelRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const pendingShiftToggleRef = useRef(false)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [anchorReturnOffsets, setAnchorReturnOffsets] = useState({ right: 12, bottom: 12 })
  const [showTip, setShowTip] = useState(true)
  // Keep showing the last loaded JPEG while stepping frames (avoids black flash).
  const [jpegFrame, setJpegFrame] = useState(currentFrame)

  // Prevent feedback loop: video timeupdate -> setCurrentFrame -> seek effect
  const videoIsDriving = useRef(false)
  const fpsRef = useRef(video?.fps || 30)

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const fps = video?.fps || 30

  useEffect(() => { fpsRef.current = fps }, [fps])

  // Debounce JPEG src updates during rapid arrow-key scrubbing.
  useEffect(() => {
    const t = setTimeout(() => setJpegFrame(currentFrame), 150)
    return () => clearTimeout(t)
  }, [currentFrame])

  useEffect(() => {
    setJpegFrame(currentFrame)
  }, [vid])

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

  useEffect(() => {
    setAnchorReturnOffsets({ right: 12, bottom: 12 })
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
    const cancelShiftToggleArm = () => {
      pendingShiftToggleRef.current = false
    }

    const handleMouseDown = (e: MouseEvent) => {
      if (e.shiftKey) cancelShiftToggleArm()
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (targetIsTypingContext(e.target)) return

      const s = useStore.getState()
      const vidMeta = video
      const framesMax = Math.max((vidMeta?.num_frames ?? 1) - 1, 0)

      const mod = e.metaKey || e.ctrlKey

      if (mod && !e.repeat && e.key.toLowerCase() === 'z') {
        const hasVideo =
          !!(s.project && s.currentVideoId && s.project.videos[s.currentVideoId])
        const tracking = s.propagationStatus === 'running'
        if (
          !hasVideo
          || tracking
          || s.historyBusy
        ) {
          cancelShiftToggleArm()
          return
        }
        cancelShiftToggleArm()
        const wantsRedo = e.shiftKey
        if (wantsRedo) {
          if (s.redoStack.length > 0) {
            e.preventDefault()
            void s.redoLast()
          }
        } else if (s.undoStack.length > 0) {
          e.preventDefault()
          void s.undoLast()
        }
        return
      }

      const objectsMap = vidMeta?.objects ?? {}
      const objectIds = Object.keys(objectsMap).sort((a, b) => Number(a) - Number(b))

      if (e.key === 'Tab' && objectIds.length > 0) {
        cancelShiftToggleArm()
        e.preventDefault()
        const curId = s.currentObjectId
        const idx = curId ? objectIds.indexOf(curId) : -1
        const delta = e.shiftKey ? -1 : 1
        const nextIdx = idx < 0
          ? 0
          : (idx + delta + objectIds.length) % objectIds.length
        const nextId = objectIds[nextIdx]!
        const pm = s.pointMode
        s.setCurrentObject(nextId)
        if (pm === 'remove') s.setPointMode('remove')
        return
      }

      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        if (e.shiftKey) cancelShiftToggleArm()
        e.preventDefault()
        const delta = e.key === 'ArrowRight' ? s.frameJump : -s.frameJump
        const minFrame = s.propagationStartFrame
        const next = Math.max(minFrame, Math.min(framesMax, currentFrame + delta))
        setCurrentFrame(next)
        return
      }

      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') {
        if (!mod && !e.altKey && !e.repeat) pendingShiftToggleRef.current = true
        return
      }

      if (e.shiftKey) cancelShiftToggleArm()
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      if (targetIsTypingContext(e.target)) return
      if (!(e.code === 'ShiftLeft' || e.code === 'ShiftRight')) return
      if (!pendingShiftToggleRef.current) return

      pendingShiftToggleRef.current = false
      const s = useStore.getState()
      if (!s.currentObjectId) return

      const m = s.pointMode
      const next = m === 'add' ? 'remove' : 'add'
      s.setPointMode(next)
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    window.addEventListener('mousedown', handleMouseDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('mousedown', handleMouseDown)
    }
  }, [currentFrame, video, setCurrentFrame])

  const targetAnchorFrame =
    anchorFrames.length > 0 && currentAnchorIndex >= 0 && currentAnchorIndex < anchorFrames.length
      ? anchorFrames[currentAnchorIndex]
      : null

  const anchorLabelingActive =
    anchorPhase &&
    targetAnchorFrame != null &&
    !(anchorRemainderInferencing && !anchorRemainderAwaitingCommit)

  const showAnchorReturnChip =
    anchorLabelingActive &&
    !isPlaying &&
    dimensions.width > 0 &&
    currentFrame !== targetAnchorFrame

  const handleAnchorReturnDragStart = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    const panel = anchorReturnPanelRef.current
    const stack = frameStackRef.current
    if (!panel || !stack) return
    const startX = e.clientX
    const startY = e.clientY
    const startRight = anchorReturnOffsets.right
    const startBottom = anchorReturnOffsets.bottom
    const cw = stack.clientWidth
    const ch = stack.clientHeight
    const pw = panel.offsetWidth
    const ph = panel.offsetHeight
    const pad = 6
    function move(ev: MouseEvent) {
      const dx = ev.clientX - startX
      const dy = ev.clientY - startY
      let nr = startRight - dx
      let nb = startBottom - dy
      nr = Math.max(pad, Math.min(cw - pad - pw, nr))
      nb = Math.max(pad, Math.min(ch - pad - ph, nb))
      setAnchorReturnOffsets({ right: nr, bottom: nb })
    }
    function up() {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }, [anchorReturnOffsets.right, anchorReturnOffsets.bottom])

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
          ref={frameStackRef}
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

          {/* Layer 1b: high-fidelity JPEG — only when settled on this frame.
              While scrubbing, hide the overlay so a stale frame-0 JPEG does not
              cover the seeked video underneath. */}
          {!isPlaying && pid && vid && jpegFrame === currentFrame && (
            <img
              key={`${pid}/${vid}/${jpegFrame}`}
              src={frameUrl(pid, vid, jpegFrame)}
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
          <AnnotationCanvas
            width={dimensions.width}
            height={dimensions.height}
            videoRef={videoRef}
            scrubbing={jpegFrame !== currentFrame}
          />

          {showAnchorReturnChip && targetAnchorFrame != null && (
            <div
              ref={anchorReturnPanelRef}
              role="dialog"
              aria-label="Return to anchor frame"
              className="absolute z-20 w-[min(calc(100%-16px),260px)] rounded-xl border border-blue-600/35 bg-[#111]/95 shadow-xl backdrop-blur-sm overflow-hidden"
              style={{ right: anchorReturnOffsets.right, bottom: anchorReturnOffsets.bottom }}
            >
              <div
                className="flex items-center gap-2 px-2.5 py-1.5 border-b border-[#333] bg-[#161616]/95 cursor-grab active:cursor-grabbing select-none"
                onMouseDown={handleAnchorReturnDragStart}
              >
                <GripVertical size={14} className="text-[#666] flex-shrink-0" aria-hidden />
                <span className="text-[11px] font-medium text-blue-300/95 truncate">
                  Anchor {currentAnchorIndex + 1} · frame #{targetAnchorFrame}
                </span>
              </div>
              <div className="p-2.5 space-y-2">
                <p className="text-[11px] text-[#888] leading-snug">
                  You moved to frame #{currentFrame}. Go back to finish this anchor before Done, next?
                </p>
                <button
                  type="button"
                  className="btn btn-primary w-full py-1.5 text-xs font-medium"
                  onClick={() => setCurrentFrame(targetAnchorFrame)}
                >
                  Go to anchor frame #{targetAnchorFrame}
                </button>
              </div>
            </div>
          )}
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
      {Object.keys(video.objects).length > 0 && (pointMode || currentObjectId) && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/70 rounded-xl px-4 py-2 text-xs text-[#ccc] pointer-events-none text-center max-w-[min(560px,calc(100%-2rem))]">
          {pointMode !== null ? (
            <div>
              {pointMode === 'add' ? 'Click to add (+) point' : 'Click to add (−) exclusion point'}
            </div>
          ) : (
            <div className="text-[#bbb]">Tap Shift once to toggle add (+) vs remove (−) mode</div>
          )}
          <div className="text-[10px] text-[#888] mt-1.5 leading-snug">
            Tab · next object · Shift+Tab · previous · tap Shift toggles +/- · {redoModGlyph}+Z undo · {redoModGlyph}+Shift+Z redo
          </div>
        </div>
      )}
    </div>
  )
}
