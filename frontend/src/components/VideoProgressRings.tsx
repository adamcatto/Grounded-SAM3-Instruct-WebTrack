import type { VideoMeta } from '../types'
import { anchorLabelingPercent, trackingCoveragePercent } from '../utils/videoProgress'

interface RingProps {
  pct: number
  stroke: string
  trackStroke: string
  labelTitle: string
}

function PartialRing({ pct, stroke, trackStroke, labelTitle }: RingProps) {
  const size = 26
  const strokeWidth = 2.75
  const r = (size - strokeWidth) / 2
  const c = 2 * Math.PI * r
  const p = Math.min(100, Math.max(0, pct))
  const filled = (p / 100) * c
  const gap = Math.max(c - filled, 0.001)

  return (
    <div
      className="relative shrink-0"
      style={{ width: size, height: size }}
      title={`${labelTitle}: ${Math.round(p)}%`}
    >
      <svg width={size} height={size} className="block" aria-hidden>
        <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={trackStroke}
            strokeWidth={strokeWidth}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={stroke}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${filled} ${gap}`}
          />
        </g>
      </svg>
      <span
        className="pointer-events-none absolute inset-0 flex items-center justify-center text-[7px] font-semibold tabular-nums leading-none text-[#ccc]"
      >
        {Math.round(p)}
        <span className="text-[6px] font-medium text-[#888] ml-px">%</span>
      </span>
    </div>
  )
}

interface VideoProgressRingsProps {
  video: VideoMeta
  className?: string
}

/**
 * Anchor labeling (orange) and whole-video tracking coverage (green), right-aligned pills.
 */
export default function VideoProgressRings({ video, className = '' }: VideoProgressRingsProps) {
  const anchorPct = anchorLabelingPercent(video)
  const trackPct = trackingCoveragePercent(video)

  return (
    <div
      className={`flex items-center gap-1 shrink-0 ${className}`}
      role="group"
      aria-label={`Anchors labeled ${anchorPct} percent. Frames tracked ${trackPct} percent.`}
    >
      <PartialRing
        pct={anchorPct}
        stroke="#f97316"
        trackStroke="rgba(249,115,22,0.18)"
        labelTitle="Anchor frames labeled"
      />
      <PartialRing
        pct={trackPct}
        stroke="#22c55e"
        trackStroke="rgba(34,197,94,0.18)"
        labelTitle="Whole-video frames tracked"
      />
    </div>
  )
}
