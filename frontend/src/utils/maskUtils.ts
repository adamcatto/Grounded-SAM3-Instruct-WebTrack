/**
 * Utilities for rendering base64 mask PNGs onto a canvas.
 */

const imageCache = new Map<string, HTMLImageElement>()

export function loadMaskImage(b64: string): Promise<HTMLImageElement> {
  if (imageCache.has(b64)) {
    return Promise.resolve(imageCache.get(b64)!)
  }
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      imageCache.set(b64, img)
      resolve(img)
    }
    img.onerror = reject
    img.src = `data:image/png;base64,${b64}`
  })
}

export async function drawMasks(
  ctx: CanvasRenderingContext2D,
  masks: Record<string, string>,  // objId → base64 PNG
  width: number,
  height: number
): Promise<void> {
  for (const [, b64] of Object.entries(masks)) {
    try {
      const img = await loadMaskImage(b64)
      ctx.drawImage(img, 0, 0, width, height)
    } catch (e) {
      console.warn('Failed to draw mask:', e)
    }
  }
}

export function drawPoints(
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number; label: 0 | 1 }[],
  canvasWidth: number,
  canvasHeight: number
) {
  for (const pt of points) {
    const px = pt.x * canvasWidth
    const py = pt.y * canvasHeight
    const isPos = pt.label === 1

    // Outer ring
    ctx.beginPath()
    ctx.arc(px, py, 10, 0, Math.PI * 2)
    ctx.fillStyle = isPos ? 'rgba(59, 130, 246, 0.3)' : 'rgba(239, 68, 68, 0.3)'
    ctx.fill()

    // Inner circle
    ctx.beginPath()
    ctx.arc(px, py, 7, 0, Math.PI * 2)
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
      ctx.moveTo(px - 3.5, py)
      ctx.lineTo(px + 3.5, py)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(px, py - 3.5)
      ctx.lineTo(px, py + 3.5)
      ctx.stroke()
    } else {
      // Minus
      ctx.beginPath()
      ctx.moveTo(px - 3.5, py)
      ctx.lineTo(px + 3.5, py)
      ctx.stroke()
    }
  }
}

export function clearMaskCache() {
  imageCache.clear()
}
