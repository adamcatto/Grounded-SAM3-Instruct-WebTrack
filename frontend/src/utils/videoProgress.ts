import type { VideoMeta } from '../types'
import { computeAnchorFrames } from './anchorFrames'

/** 0–100: how many computed anchor indices have evidence of labeling.
 *  A frame counts as labeled if it appears in annotated_anchors OR if any
 *  object has point_prompts saved for that frame (handles commits that didn't persist). */
export function anchorLabelingPercent(v: VideoMeta): number {
  const start = v.start_frame ?? 0
  const anchors = computeAnchorFrames(start, v.num_frames)
  if (anchors.length === 0) return 0
  const done = new Set(v.annotated_anchors ?? [])
  // Also count frames with saved point prompts for any object
  for (const objPrompts of Object.values(v.point_prompts ?? {})) {
    for (const frameKey of Object.keys(objPrompts)) {
      done.add(Number(frameKey))
    }
  }
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
