/**
 * Utilities for rendering base64 mask PNGs onto a canvas.
 *
 * Uses ImageBitmap (instead of HTMLImageElement) so that evicted entries can
 * be explicitly freed with .close(), releasing GPU/CPU memory immediately.
 * The cache is bounded to MAX_IMAGE_CACHE entries (FIFO eviction).
 */

const MAX_IMAGE_CACHE = 40  // decoded bitmaps; each full-res RGBA mask is ~8 MB
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

export function drawCompositeMask(
  ctx: CanvasRenderingContext2D,
  bitmap: ImageBitmap,
  width: number,
  height: number,
  opacity = 1.0,
): void {
  const prev = ctx.globalAlpha
  ctx.globalAlpha = Math.max(0, Math.min(1, opacity))
  ctx.drawImage(bitmap, 0, 0, width, height)
  ctx.globalAlpha = prev
}

export async function drawMasks(
  ctx: CanvasRenderingContext2D,
  masks: Record<string, string>,  // objId → base64 PNG
  width: number,
  height: number,
  opacity = 1.0,
  objectNames?: Record<string, string>,  // objId → object name (for labels)
  showLabels = true,
): Promise<void> {
  const prev = ctx.globalAlpha
  ctx.globalAlpha = Math.max(0, Math.min(1, opacity))
  
  // Store mask centers for label placement
  const labelPositions: { objId: string; cx: number; cy: number }[] = []
  
  for (const [objId, b64] of Object.entries(masks)) {
    try {
      const bitmap = await loadMaskBitmap(b64)
      ctx.drawImage(bitmap, 0, 0, width, height)
      
      // Calculate mask center for label placement
      if (showLabels && objectNames) {
        // Create temp canvas to read mask pixels and find centroid
        const tempCanvas = document.createElement('canvas')
        tempCanvas.width = width
        tempCanvas.height = height
        const tempCtx = tempCanvas.getContext('2d')
        if (tempCtx) {
          tempCtx.drawImage(bitmap, 0, 0, width, height)
          const imageData = tempCtx.getImageData(0, 0, width, height)
          const data = imageData.data
          
          let sumX = 0, sumY = 0, count = 0
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              const alpha = data[(y * width + x) * 4 + 3]
              if (alpha > 30) {
                sumX += x
                sumY += y
                count++
              }
            }
          }
          
          if (count > 0) {
            labelPositions.push({
              objId,
              cx: sumX / count,
              cy: sumY / count,
            })
          }
        }
      }
    } catch (e) {
      console.warn('Failed to draw mask:', e)
    }
  }
  ctx.globalAlpha = prev
  
  // Draw labels after all masks are drawn
  if (showLabels && objectNames && labelPositions.length > 0) {
    ctx.font = 'bold 12px Inter, system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    
    for (const { objId, cx, cy } of labelPositions) {
      // First check if this exact objId has a name (e.g., "1_1" might have "mouse_1")
      let label: string
      if (objectNames[objId]) {
        label = objectNames[objId]
      } else if (objId.includes('_')) {
        // Fallback: for "1_2", use base name + suffix
        const baseName = objectNames[objId.split('_')[0]] || objId.split('_')[0]
        label = baseName + '_' + objId.split('_')[1]
      } else {
        label = objectNames[objId] || objId
      }
      
      // Measure text for background
      const metrics = ctx.measureText(label)
      const textWidth = metrics.width
      const textHeight = 14
      const padding = 4
      
      // Draw background pill
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.beginPath()
      ctx.roundRect(
        cx - textWidth / 2 - padding,
        cy - textHeight / 2 - padding / 2,
        textWidth + padding * 2,
        textHeight + padding,
        4
      )
      ctx.fill()
      
      // Draw text
      ctx.fillStyle = '#ffffff'
      ctx.fillText(label, cx, cy)
    }
  }
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
