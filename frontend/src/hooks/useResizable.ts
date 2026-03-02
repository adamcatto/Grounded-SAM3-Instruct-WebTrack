import { useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'

interface ResizableOptions {
  /** Minimum size in px */
  min?: number
  /** Maximum size in px */
  max?: number
  /** 'horizontal' resizes width (col-resize), 'vertical' resizes height (row-resize) */
  direction?: 'horizontal' | 'vertical'
  /**
   * Reverse delta direction. Use for panels anchored to the opposite edge:
   * - 'vertical' + reverse: drag UP → bigger (e.g. timeline at bottom)
   */
  reverse?: boolean
}

/**
 * Returns [size, onMouseDown] for a resizable panel.
 * Attach onMouseDown to a <ResizeHandle> element.
 *
 * During drag, sets document.body cursor and userSelect so the resize
 * feels smooth even if the pointer wanders outside the handle.
 */
export function useResizable(
  initialSize: number,
  {
    min = 50,
    max = 2000,
    direction = 'horizontal',
    reverse = false,
  }: ResizableOptions = {}
): [number, (e: ReactMouseEvent) => void] {
  const [size, setSize] = useState(initialSize)

  function onMouseDown(e: ReactMouseEvent) {
    e.preventDefault()
    const startPos = direction === 'horizontal' ? e.clientX : e.clientY
    const startSize = size // captured at drag start — stable for the whole drag

    function onMouseMove(ev: MouseEvent) {
      const pos = direction === 'horizontal' ? ev.clientX : ev.clientY
      const delta = reverse ? startPos - pos : pos - startPos
      setSize(Math.max(min, Math.min(max, startSize + delta)))
    }

    function onMouseUp() {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }

    document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
  }

  return [size, onMouseDown]
}
