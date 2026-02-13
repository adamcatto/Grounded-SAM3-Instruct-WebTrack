import React, { useState } from 'react'
import { Pencil, Trash2, MousePointer, MinusCircle, Check } from 'lucide-react'
import { useStore } from '../../store/useStore'
import { renameObject, removeObject, clearObjectPoints } from '../../api/client'

interface Props {
  objId: string
  name: string
  color: string
  isActive: boolean
  onSelect: () => void
}

export default function ObjectCard({ objId, name, color, isActive, onSelect }: Props) {
  const {
    project, currentVideoId,
    pointMode, setPointMode, setCurrentObject,
    clearLocalPoints, setCurrentFrameMasks, currentFrameMasks,
  } = useStore()

  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState(name)
  const [removing, setRemoving] = useState(false)

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  async function handleRename() {
    if (!editName.trim() || editName === name) { setEditing(false); return }
    await renameObject(pid, vid, objId, editName.trim())
    setEditing(false)
    // Refresh project in store - quick local update
    useStore.getState().updateVideo({
      objects: {
        ...useStore.getState().project?.videos[vid]?.objects,
        [objId]: { id: objId, name: editName.trim(), color },
      },
    })
  }

  async function handleRemove() {
    if (!confirm(`Remove object "${name}"?`)) return
    setRemoving(true)
    try {
      await removeObject(pid, vid, objId)
      clearLocalPoints(objId)
      // Remove masks for this object from current frame
      const newMasks = { ...currentFrameMasks }
      delete newMasks[objId]
      setCurrentFrameMasks(newMasks)
      useStore.getState().updateVideo({
        objects: Object.fromEntries(
          Object.entries(useStore.getState().project?.videos[vid]?.objects ?? {}).filter(([k]) => k !== objId)
        ),
      })
      if (useStore.getState().currentObjectId === objId) {
        setCurrentObject(null)
      }
    } finally {
      setRemoving(false)
    }
  }

  async function handleClear() {
    clearLocalPoints(objId)
    const newMasks = { ...currentFrameMasks }
    delete newMasks[objId]
    setCurrentFrameMasks(newMasks)
    try {
      await clearObjectPoints(pid, vid, objId)
    } catch { /* ignore */ }
  }

  return (
    <div
      className={`rounded-xl border transition-all cursor-pointer
        ${isActive
          ? 'border-[#444] bg-[#1e1e1e]'
          : 'border-[#2a2a2a] bg-[#161616] hover:border-[#333] hover:bg-[#1a1a1a]'
        }`}
      onClick={onSelect}
    >
      {/* Top row: color swatch + name */}
      <div className="flex items-center gap-2.5 px-3 pt-3 pb-2">
        {/* Color thumbnail */}
        <div
          className="w-10 h-10 rounded-lg flex-shrink-0 border border-white/10"
          style={{ background: `linear-gradient(135deg, ${color}88, ${color}44)` }}
        />
        <div className="flex-1 min-w-0">
          {editing ? (
            <input
              type="text"
              value={editName}
              onChange={e => setEditName(e.target.value)}
              onBlur={handleRename}
              onKeyDown={e => { if (e.key === 'Enter') handleRename(); if (e.key === 'Escape') setEditing(false) }}
              className="w-full text-sm py-0.5"
              autoFocus
              onClick={e => e.stopPropagation()}
            />
          ) : (
            <span className="text-sm font-medium text-white block truncate">{name}</span>
          )}
        </div>
        <button
          onClick={e => { e.stopPropagation(); setEditing(!editing) }}
          className="p-1 text-[#555] hover:text-[#ccc] rounded"
          title="Rename"
        >
          {editing ? <Check size={13} /> : <Pencil size={13} />}
        </button>
        <button
          onClick={e => { e.stopPropagation(); handleRemove() }}
          className="p-1 text-[#555] hover:text-red-400 rounded"
          title="Remove object"
          disabled={removing}
        >
          <Trash2 size={13} />
        </button>
      </div>

      {/* Active controls: Add / Remove point buttons */}
      {isActive && (
        <div className="px-3 pb-3">
          <p className="text-[11px] text-[#666] mb-2 leading-relaxed">
            Select <span className="text-blue-400">+</span> to add areas and{' '}
            <span className="text-red-400">−</span> to remove areas from the object.
          </p>
          <div className="flex gap-2">
            <button
              onClick={e => {
                e.stopPropagation()
                setCurrentObject(objId)
                setPointMode(pointMode === 'add' ? null : 'add')
              }}
              className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-medium border transition-colors
                ${pointMode === 'add' && isActive
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'border-[#333] text-[#888] hover:border-blue-500/60 hover:text-blue-400'}`}
            >
              <MousePointer size={12} />
              Add
            </button>
            <button
              onClick={e => {
                e.stopPropagation()
                setCurrentObject(objId)
                setPointMode(pointMode === 'remove' ? null : 'remove')
              }}
              className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-medium border transition-colors
                ${pointMode === 'remove' && isActive
                  ? 'bg-red-600 border-red-500 text-white'
                  : 'border-[#333] text-[#888] hover:border-red-500/60 hover:text-red-400'}`}
            >
              <MinusCircle size={12} />
              Remove
            </button>
          </div>
          <button
            onClick={e => { e.stopPropagation(); handleClear() }}
            className="w-full mt-2 py-1 text-xs text-[#666] hover:text-[#999] transition-colors"
          >
            Clear selection
          </button>
        </div>
      )}

      {/* Inactive: Edit/Clear links */}
      {!isActive && (
        <div className="flex gap-3 px-3 pb-3">
          <button
            onClick={e => { e.stopPropagation(); onSelect() }}
            className="text-xs text-[#666] hover:text-[#aaa] flex items-center gap-1"
          >
            <Pencil size={11} /> Edit selection
          </button>
          <button
            onClick={e => { e.stopPropagation(); handleClear() }}
            className="text-xs text-[#666] hover:text-[#aaa] flex items-center gap-1"
          >
            <Trash2 size={11} /> Clear
          </button>
        </div>
      )}
    </div>
  )
}
