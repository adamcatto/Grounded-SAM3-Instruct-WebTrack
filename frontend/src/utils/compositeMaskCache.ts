/** LRU cache of decoded composite mask bitmaps (one per frame). */

const MAX_COMPOSITE_CACHE = 80
const cache = new Map<string, ImageBitmap>()

function key(pid: string, vid: string, fidx: number): string {
  return `${pid}/${vid}/${fidx}`
}

export function getCompositeBitmap(pid: string, vid: string, fidx: number): ImageBitmap | undefined {
  return cache.get(key(pid, vid, fidx))
}

export function setCompositeBitmap(pid: string, vid: string, fidx: number, bitmap: ImageBitmap): void {
  const k = key(pid, vid, fidx)
  if (cache.has(k)) {
    cache.delete(k)
  } else if (cache.size >= MAX_COMPOSITE_CACHE) {
    const oldest = cache.keys().next().value
    if (oldest !== undefined) {
      cache.get(oldest)?.close()
      cache.delete(oldest)
    }
  }
  cache.set(k, bitmap)
}

export function evictCompositeFrame(pid: string, vid: string, fidx: number): void {
  const k = key(pid, vid, fidx)
  cache.get(k)?.close()
  cache.delete(k)
}

export function evictCompositeVideo(pid: string, vid: string): void {
  const prefix = `${pid}/${vid}/`
  for (const k of [...cache.keys()]) {
    if (k.startsWith(prefix)) {
      cache.get(k)?.close()
      cache.delete(k)
    }
  }
}

export function clearCompositeCache(): void {
  for (const bitmap of cache.values()) bitmap.close()
  cache.clear()
}
