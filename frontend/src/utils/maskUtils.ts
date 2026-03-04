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
  height: number,
  opacity = 1.0,
): Promise<void> {
  const prev = ctx.globalAlpha
  ctx.globalAlpha = Math.max(0, Math.min(1, opacity))
  for (const [, b64] of Object.entries(masks)) {
    try {
      const img = await loadMaskImage(b64)
      ctx.drawImage(img, 0, 0, width, height)
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
  imageCache.clear()
}
