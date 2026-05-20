/** Mirror backend `anchor_helpers` (batch propagation grid). */

export const STREAM_BATCH_SIZE = 1000
export const ANCHOR_BATCH_SIZE_MIN = 10
export const ANCHOR_BATCH_SIZE_MAX = 10_000

export function normalizeAnchorBatchSize(value: unknown, fallback = STREAM_BATCH_SIZE): number {
  const n = typeof value === 'number' ? value : parseInt(String(value), 10)
  if (!Number.isFinite(n)) return fallback
  return Math.max(ANCHOR_BATCH_SIZE_MIN, Math.min(ANCHOR_BATCH_SIZE_MAX, Math.round(n)))
}

export function videoAnchorBatchSize(
  video: { anchor_batch_size?: number } | null | undefined,
  fallback = STREAM_BATCH_SIZE,
): number {
  return normalizeAnchorBatchSize(video?.anchor_batch_size, fallback)
}

export function computeAnchorFrames(
  startFrame: number,
  numFrames: number,
  batchSize: number = STREAM_BATCH_SIZE,
): number[] {
  const last = numFrames - 1
  const anchors: number[] = []
  for (let f = startFrame; f < numFrames; f += batchSize) {
    anchors.push(f)
  }
  if (anchors.length === 0) anchors.push(last)
  else if (anchors[anchors.length - 1] !== last) anchors.push(last)
  return anchors
}

/** 0-based anchor indices whose frame index is in `annotatedFrames`. */
export function annotatedAnchorIndicesFromFrames(
  anchorFrames: number[],
  annotatedFrames: number[],
): number[] {
  const done = new Set(annotatedFrames)
  const idxs: number[] = []
  anchorFrames.forEach((fr, i) => {
    if (done.has(fr)) idxs.push(i)
  })
  return idxs
}

/** First anchor index not present in `labeledFrameIndices` (queue order); `anchorFrames.length` if all done. */
export function firstUnlabeledAnchorIndex(
  anchorFrames: number[],
  labeledFrameIndices: Iterable<number>,
): number {
  const done = new Set(labeledFrameIndices)
  for (let i = 0; i < anchorFrames.length; i++) {
    if (!done.has(anchorFrames[i]!)) return i
  }
  return anchorFrames.length
}

/** True once any anchor has been committed or point-labeled (interval locked in settings). */
export function hasAnchorLabelingStarted(
  video: {
    start_frame?: number
    num_frames: number
    annotated_anchors?: number[]
    anchor_labeling_complete?: boolean
    anchor_batch_size?: number
    point_prompts?: Record<string, Record<string, unknown>>
  },
  fallbackBatchSize = STREAM_BATCH_SIZE,
): boolean {
  if ((video.annotated_anchors?.length ?? 0) > 0) return true
  if (video.anchor_labeling_complete) return true
  const bs = videoAnchorBatchSize(video, fallbackBatchSize)
  const anchors = new Set(computeAnchorFrames(video.start_frame ?? 0, video.num_frames, bs))
  for (const objPrompts of Object.values(video.point_prompts ?? {})) {
    for (const frameKey of Object.keys(objPrompts)) {
      const fi = Number(frameKey)
      if (anchors.has(fi)) return true
    }
  }
  return false
}
