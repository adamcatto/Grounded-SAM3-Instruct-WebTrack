import React, { useEffect, useRef } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { thumbUrl } from '../../api/client'

export default function FrameStrip() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { project, currentVideoId, currentFrame, setCurrentFrame, propagationStartFrame } = store

  const stripRef = useRef<HTMLDivElement>(null)
  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  // Keep current frame visible
  useEffect(() => {
    if (!stripRef.current) return
    const el = stripRef.current.querySelector(`[data-fidx="${currentFrame}"]`) as HTMLElement
    if (el) {
      el.scrollIntoView({ inline: 'nearest', behavior: 'smooth', block: 'nearest' })
    }
  }, [currentFrame])

  if (!video) return null

  // Use preview_indices if available (only a subset of frames are on disk)
  // Otherwise fall back to computing evenly-spaced indices
  const total = video.num_frames
  let indices: number[]
  if (video.preview_indices && video.preview_indices.length > 0) {
    indices = video.preview_indices
  } else {
    const step = Math.max(1, Math.floor(total / 150))
    indices = []
    for (let i = 0; i < total; i += step) {
      indices.push(i)
    }
  }

  // Hide frames before the propagation start frame
  indices = indices.filter(i => i >= propagationStartFrame)

  // Compute effective step for highlight range
  const step = indices.length > 1
    ? Math.max(1, Math.ceil((indices[indices.length - 1] - indices[0]) / indices.length))
    : 1

  return (
    <div
      ref={stripRef}
      className="flex items-center gap-0.5 overflow-x-auto overflow-y-hidden h-12 px-1 select-none"
      style={{ scrollbarWidth: 'none' }}
    >
      {indices.map(fidx => (
        <button
          key={fidx}
          data-fidx={fidx}
          onClick={() => setCurrentFrame(fidx)}
          className={`relative flex-shrink-0 h-10 rounded overflow-hidden border-2 transition-colors
            ${Math.abs(currentFrame - fidx) < step
              ? 'border-white'
              : 'border-transparent hover:border-[#555]'}`}
          style={{ aspectRatio: `${video.width}/${video.height}` }}
        >
          <img
            src={thumbUrl(pid, vid, fidx)}
            alt={`Frame ${fidx}`}
            className="h-full w-full object-cover"
            loading="lazy"
          />
        </button>
      ))}
    </div>
  )
}
