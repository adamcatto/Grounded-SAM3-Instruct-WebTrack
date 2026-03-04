import React from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'

export default function ObjectTrackRow() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const { currentFrame, setCurrentFrame, propagationStartFrame } = store

  if (!video || Object.keys(video.objects).length === 0) return null

  const total = video.num_frames
  const startFrame = propagationStartFrame
  const propagatedSet = new Set(video.propagated_frames ?? [])

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
            {/* Propagated segments */}
            {Array.from(propagatedSet).sort((a, b) => a - b).filter(f => f >= startFrame).map((f) => {
              const x = ((f - startFrame) / (total - 1 - startFrame)) * 100
              return (
                <div
                  key={f}
                  className="absolute top-0 h-full w-[2px] rounded-sm"
                  style={{
                    left: `${x}%`,
                    background: obj.color,
                    opacity: 0.8,
                  }}
                />
              )
            })}

            {/* Current frame indicator */}
            <div
              className="absolute top-0 h-full w-0.5 bg-white z-10"
              style={{ left: `${((currentFrame - startFrame) / (total - 1 - startFrame)) * 100}%` }}
            />

            {/* Click scrubber */}
            <div
              className="absolute inset-0 cursor-pointer"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const pct = (e.clientX - rect.left) / rect.width
                const rangeLen = total - 1 - startFrame
                setCurrentFrame(Math.max(startFrame, Math.min(total - 1, Math.round(pct * rangeLen + startFrame))))
              }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
