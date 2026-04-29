import type { VideoMeta } from '../types'
import { computeAnchorFrames } from './anchorFrames'

/** 0–100: how many computed anchor indices are present in annotated_anchors. */
export function anchorLabelingPercent(v: VideoMeta): number {
  const start = v.start_frame ?? 0
  const anchors = computeAnchorFrames(start, v.num_frames)
  if (anchors.length === 0) return 0
  const done = new Set(v.annotated_anchors ?? [])
  const labeledCount = anchors.filter(a => done.has(a)).length
  return Math.min(100, Math.round((100 * labeledCount) / anchors.length))
}

/** 0–100: unique propagated frames / total frames (whole-video coverage). */
export function trackingCoveragePercent(v: VideoMeta): number {
  if (v.propagation_complete) return 100
  const nf = Math.max(1, v.num_frames)
  const n = [...new Set(v.propagated_frames ?? [])].length
  return Math.min(100, Math.round((100 * n) / nf))
}
