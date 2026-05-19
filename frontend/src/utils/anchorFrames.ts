/** Mirror backend `anchor_helpers.compute_anchor_frames` (batch propagation grid). */

export const STREAM_BATCH_SIZE = 1000

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
