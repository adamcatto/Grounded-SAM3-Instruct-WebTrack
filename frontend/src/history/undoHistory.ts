/** One undoable user action: symmetric undo/redo callbacks + toast labels. */

export const MAX_UNDO_STACK = 50

export interface HistoryCommand {
  labelUndo: string
  labelRedo: string
  undo: () => Promise<void>
  redo: () => Promise<void>
}
