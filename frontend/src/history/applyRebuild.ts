import { useStore } from '../store/useStore'
import type { MaskData } from '../types'
import { clearMaskCache } from '../utils/maskUtils'

/** Merge rebuild_from_config PNG masks into live + saved cache; bust decoded bitmap cache. */
export function applyRebuildMasksToStore(masksByFrame: Record<string, MaskData>) {
  const s = useStore.getState()
  const cf = s.currentFrame
  for (const [fk, masks] of Object.entries(masksByFrame)) {
    const f = Number(fk)
    const prev = s.savedMaskCache[f] ?? {}
    s.setSavedMask(f, { ...prev, ...masks })
    if (f === cf) {
      const live = s.currentFrameMasksFrame === cf ? s.currentFrameMasks : {}
      s.setCurrentFrameMasks({ ...live, ...masks }, cf)
    }
  }
  clearMaskCache()
}

export function localAnnotationsToPointPrompts(
  local: Record<string, Record<string, { points: { x: number; y: number; label: 0 | 1 }[] }>>,
): Record<string, Record<string, { points: [number, number][]; labels: number[] }>> {
  const out: Record<string, Record<string, { points: [number, number][]; labels: number[] }>> = {}
  for (const [objId, frames] of Object.entries(local)) {
    out[objId] = {}
    for (const [fi, { points }] of Object.entries(frames)) {
      out[objId][fi] = {
        points: points.map(p => [p.x, p.y] as [number, number]),
        labels: points.map(p => p.label),
      }
    }
  }
  return out
}
