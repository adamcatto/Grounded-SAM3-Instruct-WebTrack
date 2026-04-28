import type { MouseEvent as ReactMouseEvent } from 'react'

interface ResizeHandleProps {
  direction: 'horizontal' | 'vertical'
  onMouseDown: (e: ReactMouseEvent) => void
}

/**
 * A thin drag-handle divider between two panels.
 *
 * - 'horizontal': vertical bar between left/right panels (col-resize cursor)
 * - 'vertical':   horizontal bar between top/bottom panels (row-resize cursor)
 *
 * The grab area is 5 px; the visible indicator is a 1 px centered line that
 * highlights blue on hover so it's discoverable but unobtrusive.
 */
export default function ResizeHandle({ direction, onMouseDown }: ResizeHandleProps) {
  const isCol = direction === 'horizontal'
  return (
    <div
      onMouseDown={onMouseDown}
      className={[
        'flex-shrink-0 group flex items-center justify-center z-10 select-none',
        isCol ? 'cursor-col-resize self-stretch' : 'cursor-row-resize w-full',
      ].join(' ')}
      style={isCol ? { width: 5 } : { height: 5 }}
    >
      <div
        className={[
          'bg-[#1e1e1e] group-hover:bg-blue-500/50 transition-colors duration-150',
          isCol ? 'w-px h-full' : 'h-px w-full',
        ].join(' ')}
      />
    </div>
  )
}
