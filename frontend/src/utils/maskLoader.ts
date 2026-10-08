import { getSavedMask, displayMaskUrl } from '../api/client'
import type { MaskData } from '../types'
import {
  evictCompositeFrame,
  getCompositeBitmap,
  setCompositeBitmap,
} from './compositeMaskCache'

const PREFETCH_RADIUS = 2
const DISPLAY_PREFETCH_RADIUS = 4

type InFlightKey = string

const inFlightPerObject = new Map<InFlightKey, Promise<MaskData | null>>()
const inFlightDisplay = new Map<InFlightKey, Promise<ImageBitmap | null>>()

// Bumped whenever masks are cleared / the video is reset. Responses to requests
// issued under an older generation are dropped, so a fetch that was in flight
// during the clear cannot repopulate the caches with the cleared masks.
let generation = 0

/** Forget all in-flight mask requests and discard their results when they land. */
export function invalidateMaskLoader(): void {
  generation += 1
  inFlightPerObject.clear()
  inFlightDisplay.clear()
}

function perObjectKey(pid: string, vid: string, fidx: number): InFlightKey {
  return `obj:${pid}/${vid}/${fidx}`
}

function displayKey(pid: string, vid: string, fidx: number): InFlightKey {
  return `dsp:${pid}/${vid}/${fidx}`
}

/** Fetch per-object masks (JSON base64). Used for hover labels and annotation. */
export async function loadPerObjectMasks(
  pid: string,
  vid: string,
  fidx: number,
): Promise<MaskData | null> {
  const k = perObjectKey(pid, vid, fidx)
  const existing = inFlightPerObject.get(k)
  if (existing) return existing

  const gen = generation
  const promise = getSavedMask(pid, vid, fidx)
    .then(data => {
      if (gen !== generation) return null
      const masks = data.masks ?? {}
      return Object.keys(masks).length > 0 ? masks : null
    })
    .catch(() => null)
    .finally(() => { if (inFlightPerObject.get(k) === promise) inFlightPerObject.delete(k) })

  inFlightPerObject.set(k, promise)
  return promise
}

/** Fetch pre-materialized display WebP (Phase A scrub cache). */
export async function loadDisplayBitmap(
  pid: string,
  vid: string,
  fidx: number,
  signal?: AbortSignal,
): Promise<ImageBitmap | null> {
  const cached = getCompositeBitmap(pid, vid, fidx)
  if (cached) return cached

  const k = displayKey(pid, vid, fidx)

  const existing = inFlightDisplay.get(k)
  if (existing) return existing

  const gen = generation
  const promise = fetch(displayMaskUrl(pid, vid, fidx), { cache: 'no-store', signal })
    .then(async resp => {
      if (resp.status === 404) return null
      if (!resp.ok) throw new Error(`display ${resp.status}`)
      const blob = await resp.blob()
      if (blob.size === 0 || gen !== generation) return null
      const bitmap = await createImageBitmap(blob)
      if (gen !== generation) {
        bitmap.close()
        return null
      }
      setCompositeBitmap(pid, vid, fidx, bitmap)
      return bitmap
    })
    .catch(() => null)
    .finally(() => { if (inFlightDisplay.get(k) === promise) inFlightDisplay.delete(k) })

  inFlightDisplay.set(k, promise)
  return promise
}

/** @deprecated Use loadDisplayBitmap — kept as alias for callers. */
export const loadCompositeBitmap = loadDisplayBitmap

export function invalidateMaskLoaderFrame(pid: string, vid: string, fidx: number): void {
  inFlightPerObject.delete(perObjectKey(pid, vid, fidx))
  inFlightDisplay.delete(displayKey(pid, vid, fidx))
  evictCompositeFrame(pid, vid, fidx)
}

/** Prefetch per-object masks around `center` for hover/edit detail. */
export function prefetchMaskWindow(
  pid: string,
  vid: string,
  center: number,
  direction: -1 | 0 | 1,
  propagatedFrames: Set<number>,
  onPerObject: (fidx: number, masks: MaskData) => void,
): void {
  const offsets: number[] = []
  for (let d = 1; d <= PREFETCH_RADIUS; d++) {
    if (direction >= 0) offsets.push(d)
    if (direction <= 0) offsets.push(-d)
  }

  for (const delta of offsets) {
    const fidx = center + delta
    if (fidx < 0 || !propagatedFrames.has(fidx)) continue

    void loadPerObjectMasks(pid, vid, fidx).then(masks => {
      if (masks) onPerObject(fidx, masks)
    })
  }
}

/** Prefetch composited display masks for the current scrub direction. */
export function prefetchDisplayWindow(
  pid: string,
  vid: string,
  center: number,
  direction: -1 | 0 | 1,
  propagatedFrames: Set<number>,
): void {
  const offsets: number[] = [0]
  for (let d = 1; d <= DISPLAY_PREFETCH_RADIUS; d++) {
    if (direction >= 0) offsets.push(d)
    if (direction <= 0) offsets.push(-d)
  }

  for (const delta of offsets) {
    const fidx = center + delta
    if (fidx < 0 || !propagatedFrames.has(fidx)) continue
    void loadDisplayBitmap(pid, vid, fidx)
  }
}
