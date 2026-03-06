/**
 * Utilities for rendering base64 mask PNGs onto a canvas.
 *
 * Uses ImageBitmap (instead of HTMLImageElement) so that evicted entries can
 * be explicitly freed with .close(), releasing GPU/CPU memory immediately.
 * The cache is bounded to MAX_IMAGE_CACHE entries (FIFO eviction).
 */

const MAX_IMAGE_CACHE = 100  // decoded bitmaps; each full-res RGBA mask is ~8 MB
const imageCache = new Map<string, ImageBitmap>()

function b64ToBlob(b64: string): Blob {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return new Blob([bytes], { type: 'image/png' })
}

export async function loadMaskBitmap(b64: string): Promise<ImageBitmap> {
  const cached = imageCache.get(b64)
  if (cached) return cached

  const bitmap = await createImageBitmap(b64ToBlob(b64))

  // Evict the oldest entry when at capacity (Map preserves insertion order)
  if (imageCache.size >= MAX_IMAGE_CACHE) {
    const oldestKey = imageCache.keys().next().value
    if (oldestKey !== undefined) {
      imageCache.get(oldestKey)?.close()
      imageCache.delete(oldestKey)
    }
  }

  imageCache.set(b64, bitmap)
  return bitmap
}

/** Explicitly close and remove specific bitmaps by their b64 keys. */
export function evictMaskImages(b64s: string[]) {
  for (const b64 of b64s) {
    imageCache.get(b64)?.close()
    imageCache.delete(b64)
  }
}

export async function drawMasks(
  ctx: CanvasRenderingContext2D,
  masks: Record<string, string>,  // objId → base64 PNG
  width: number,
  height: number,
  opacity = 1.0,
): Promise<void> {
  const prev = ctx.globalAlpha
  ctx.globalAlpha = Math.max(0, Math.min(1, opacity))
  for (const [, b64] of Object.entries(masks)) {
    try {
      const bitmap = await loadMaskBitmap(b64)
      ctx.drawImage(bitmap, 0, 0, width, height)
    } catch (e) {
      console.warn('Failed to draw mask:', e)
    }
  }
  ctx.globalAlpha = prev
}

export function drawPoints(
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number; label: 0 | 1 }[],
  canvasWidth: number,
  canvasHeight: number,
  scale = 1.0,
) {
  for (const pt of points) {
    const px = pt.x * canvasWidth
    const py = pt.y * canvasHeight
    const isPos = pt.label === 1
    const outerR = 10 * scale
    const innerR = 7 * scale
    const sym = 3.5 * scale

    // Outer ring
    ctx.beginPath()
    ctx.arc(px, py, outerR, 0, Math.PI * 2)
    ctx.fillStyle = isPos ? 'rgba(59, 130, 246, 0.3)' : 'rgba(239, 68, 68, 0.3)'
    ctx.fill()

    // Inner circle
    ctx.beginPath()
    ctx.arc(px, py, innerR, 0, Math.PI * 2)
    ctx.fillStyle = isPos ? '#3B82F6' : '#EF4444'
    ctx.fill()

    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // + or - symbol
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2
    ctx.lineCap = 'round'
    if (isPos) {
      // Plus
      ctx.beginPath()
      ctx.moveTo(px - sym, py)
      ctx.lineTo(px + sym, py)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(px, py - sym)
      ctx.lineTo(px, py + sym)
      ctx.stroke()
    } else {
      // Minus
      ctx.beginPath()
      ctx.moveTo(px - sym, py)
      ctx.lineTo(px + sym, py)
      ctx.stroke()
    }
  }
}

export function clearMaskCache() {
  for (const bitmap of imageCache.values()) bitmap.close()
  imageCache.clear()
}
