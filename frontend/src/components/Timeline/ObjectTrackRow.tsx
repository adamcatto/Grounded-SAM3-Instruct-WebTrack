import React, { useEffect, useMemo, useRef } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'

// Canvas pixel width for the propagated-frames track.
// Higher = more tick resolution. 2000 px is plenty for any monitor width.
const CANVAS_W = 2000
const CANVAS_H = 12

interface TrackCanvasProps {
  color: string
  propagatedFrames: number[]
  total: number
  startFrame: number
}

function TrackCanvas({ color, propagatedFrames, total, startFrame }: TrackCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rangeLen = total - 1 - startFrame

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H)
    if (rangeLen <= 0) return
    ctx.fillStyle = color
    ctx.globalAlpha = 0.8
    for (const f of propagatedFrames) {
      if (f < startFrame) continue
      const x = Math.round(((f - startFrame) / rangeLen) * CANVAS_W)
      ctx.fillRect(x, 0, 2, CANVAS_H)
    }
    ctx.globalAlpha = 1
  // propagatedFrames identity changes only when propagation advances
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [color, propagatedFrames, rangeLen])

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_W}
      height={CANVAS_H}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
    />
  )
}

export default function ObjectTrackRow() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { currentFrame, setCurrentFrame, propagationStartFrame } = store

  const propagatedFrames = useMemo(
    () => video?.propagated_frames ?? [],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [video?.propagated_frames]
  )

  if (!video || Object.keys(video.objects).length === 0) return null

  const total = video.num_frames
  const startFrame = propagationStartFrame
  const rangeLen = total - 1 - startFrame

  return (
    <div className="border-t border-[#222] bg-[#0d0d0d] px-2 py-1 space-y-0.5">
      {Object.values(video.objects).map(obj => (
        <div key={obj.id} className="flex items-center gap-2 h-5">
          {/* Label */}
          <span
            className="text-[10px] font-medium w-16 flex-shrink-0 truncate"
            style={{ color: obj.color }}
          >
            {obj.name}
          </span>

          {/* Track bar */}
          <div className="flex-1 relative h-3 bg-[#1a1a1a] rounded overflow-hidden">
            {/* Propagated segments — drawn on a single canvas instead of N divs */}
            <TrackCanvas
              color={obj.color}
              propagatedFrames={propagatedFrames}
              total={total}
              startFrame={startFrame}
            />

            {/* Current frame indicator */}
            <div
              className="absolute top-0 h-full w-0.5 bg-white z-10"
              style={{ left: `${rangeLen > 0 ? ((currentFrame - startFrame) / rangeLen) * 100 : 0}%` }}
            />

            {/* Click scrubber */}
            <div
              className="absolute inset-0 cursor-pointer z-20"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const pct = (e.clientX - rect.left) / rect.width
                setCurrentFrame(Math.max(startFrame, Math.min(total - 1, Math.round(pct * rangeLen + startFrame))))
              }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
