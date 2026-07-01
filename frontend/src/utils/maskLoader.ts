import { getSavedMask } from '../api/client'
import type { MaskData } from '../types'
import {
  evictCompositeFrame,
  getCompositeBitmap,
  setCompositeBitmap,
} from './compositeMaskCache'

const PREFETCH_RADIUS = 3

type InFlightKey = string

const inFlightPerObject = new Map<InFlightKey, Promise<MaskData | null>>()
const inFlightComposite = new Map<InFlightKey, Promise<ImageBitmap | null>>()

function perObjectKey(pid: string, vid: string, fidx: number): InFlightKey {
  return `obj:${pid}/${vid}/${fidx}`
}

function compositeKey(pid: string, vid: string, fidx: number): InFlightKey {
  return `cmp:${pid}/${vid}/${fidx}`
}

function compositeUrl(pid: string, vid: string, fidx: number): string {
  return `/api/projects/${pid}/videos/${vid}/masks/${fidx}/composite`
}

/** Fetch per-object masks (JSON base64). Dedupes concurrent requests. */
export async function loadPerObjectMasks(
  pid: string,
  vid: string,
  fidx: number,
): Promise<MaskData | null> {
  const k = perObjectKey(pid, vid, fidx)
  const existing = inFlightPerObject.get(k)
  if (existing) return existing

  const promise = getSavedMask(pid, vid, fidx)
    .then(data => {
      const masks = data.masks ?? {}
      return Object.keys(masks).length > 0 ? masks : null
    })
    .catch(() => null)
    .finally(() => { inFlightPerObject.delete(k) })

  inFlightPerObject.set(k, promise)
  return promise
}

/** Fetch composite PNG for fast scrub display. Uses browser HTTP cache when ETag matches. */
export async function loadCompositeBitmap(
  pid: string,
  vid: string,
  fidx: number,
): Promise<ImageBitmap | null> {
  const cached = getCompositeBitmap(pid, vid, fidx)
  if (cached) return cached

  const k = compositeKey(pid, vid, fidx)
  const existing = inFlightComposite.get(k)
  if (existing) return existing

  const promise = fetch(compositeUrl(pid, vid, fidx))
    .then(async resp => {
      if (resp.status === 404) return null
      if (!resp.ok) throw new Error(`composite ${resp.status}`)
      const blob = await resp.blob()
      if (blob.size === 0) return null
      const bitmap = await createImageBitmap(blob)
      setCompositeBitmap(pid, vid, fidx, bitmap)
      return bitmap
    })
    .catch(() => null)
    .finally(() => { inFlightComposite.delete(k) })

  inFlightComposite.set(k, promise)
  return promise
}

export function invalidateMaskLoaderFrame(pid: string, vid: string, fidx: number): void {
  inFlightPerObject.delete(perObjectKey(pid, vid, fidx))
  inFlightComposite.delete(compositeKey(pid, vid, fidx))
  evictCompositeFrame(pid, vid, fidx)
}

/** Prefetch masks around `center` in the direction of recent travel. */
export function prefetchMaskWindow(
  pid: string,
  vid: string,
  center: number,
  direction: -1 | 0 | 1,
  propagatedFrames: Set<number>,
  onPerObject: (fidx: number, masks: MaskData) => void,
): void {
  const offsets: number[] = [0]
  for (let d = 1; d <= PREFETCH_RADIUS; d++) {
    if (direction >= 0) offsets.push(d)
    if (direction <= 0) offsets.push(-d)
  }

  for (const delta of offsets) {
    const fidx = center + delta
    if (fidx < 0 || !propagatedFrames.has(fidx)) continue

    void loadCompositeBitmap(pid, vid, fidx)

    void loadPerObjectMasks(pid, vid, fidx).then(masks => {
      if (masks) onPerObject(fidx, masks)
    })
  }
}
