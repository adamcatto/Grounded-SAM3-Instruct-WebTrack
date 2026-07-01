import { getSavedMask, displayMaskUrl } from '../api/client'
import type { MaskData } from '../types'
import {
  evictCompositeFrame,
  getCompositeBitmap,
  setCompositeBitmap,
} from './compositeMaskCache'

const PREFETCH_RADIUS = 2

type InFlightKey = string

const inFlightPerObject = new Map<InFlightKey, Promise<MaskData | null>>()
const inFlightDisplay = new Map<InFlightKey, Promise<ImageBitmap | null>>()
const displayMissKeys = new Set<InFlightKey>()

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

/** Fetch pre-materialized display WebP (Phase A scrub cache). */
export async function loadDisplayBitmap(
  pid: string,
  vid: string,
  fidx: number,
): Promise<ImageBitmap | null> {
  const cached = getCompositeBitmap(pid, vid, fidx)
  if (cached) return cached

  const k = displayKey(pid, vid, fidx)
  if (displayMissKeys.has(k)) return null

  const existing = inFlightDisplay.get(k)
  if (existing) return existing

  const promise = fetch(displayMaskUrl(pid, vid, fidx))
    .then(async resp => {
      if (resp.status === 404) {
        displayMissKeys.add(k)
        return null
      }
      if (!resp.ok) throw new Error(`display ${resp.status}`)
      const blob = await resp.blob()
      if (blob.size === 0) return null
      const bitmap = await createImageBitmap(blob)
      setCompositeBitmap(pid, vid, fidx, bitmap)
      return bitmap
    })
    .catch(() => null)
    .finally(() => { inFlightDisplay.delete(k) })

  inFlightDisplay.set(k, promise)
  return promise
}

/** @deprecated Use loadDisplayBitmap — kept as alias for callers. */
export const loadCompositeBitmap = loadDisplayBitmap

export function invalidateMaskLoaderFrame(pid: string, vid: string, fidx: number): void {
  inFlightPerObject.delete(perObjectKey(pid, vid, fidx))
  inFlightDisplay.delete(displayKey(pid, vid, fidx))
  displayMissKeys.delete(displayKey(pid, vid, fidx))
  evictCompositeFrame(pid, vid, fidx)
}

export function clearDisplayMissCache(pid?: string, vid?: string): void {
  if (!pid || !vid) {
    displayMissKeys.clear()
    return
  }
  const prefix = `dsp:${pid}/${vid}/`
  for (const k of [...displayMissKeys]) {
    if (k.startsWith(prefix)) displayMissKeys.delete(k)
  }
}

/** Prefetch per-object masks around `center` (display WebP is current-frame only). */
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

    void loadPerObjectMasks(pid, vid, fidx).then(masks => {
      if (masks) onPerObject(fidx, masks)
    })
  }
}
