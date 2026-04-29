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
