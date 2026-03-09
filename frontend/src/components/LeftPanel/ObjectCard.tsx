import React, { useState, useEffect } from 'react'
import { Pencil, Trash2, MousePointer, MinusCircle, Check, ChevronDown, ChevronUp, PlusCircle, Save, Loader } from 'lucide-react'
import { useStore } from '../../store/useStore'
import { renameObject, removeObject, clearObjectPoints, updateObject, addInstance } from '../../api/client'

interface Props {
  objId: string
  name: string
  color: string
  isActive: boolean
  onSelect: () => void
  description?: string
  minInstances?: number
  maxInstances?: number
}

export default function ObjectCard({ objId, name, color, isActive, onSelect, description, minInstances = 1, maxInstances = 1 }: Props) {
  const {
    project, currentVideoId,
    pointMode, setPointMode, setCurrentObject,
    clearLocalPoints, setCurrentFrameMasks, currentFrameMasks,
    addToast,
  } = useStore()

  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState(name)
  const [removing, setRemoving] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [editDesc, setEditDesc] = useState(description ?? '')
  const [editMin, setEditMin] = useState(minInstances)
  const [editMax, setEditMax] = useState(maxInstances)
  const [savingDetails, setSavingDetails] = useState(false)
  const [detailsDirty, setDetailsDirty] = useState(false)

  // Sync state when props change (e.g., after save updates store)
  useEffect(() => {
    setEditDesc(description ?? '')
  }, [description])
  useEffect(() => {
    setEditMin(minInstances)
  }, [minInstances])
  useEffect(() => {
    setEditMax(maxInstances)
  }, [maxInstances])

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const hasDescription = (description ?? '').trim().length > 0

  async function handleRename() {
    if (!editName.trim() || editName === name) { setEditing(false); return }
    await renameObject(pid, vid, objId, editName.trim())
    setEditing(false)
    // Refresh project in store - quick local update
    useStore.getState().updateVideo({
      objects: {
        ...useStore.getState().project?.videos[vid]?.objects,
        [objId]: { id: objId, name: editName.trim(), color, description, min_instances: minInstances, max_instances: maxInstances },
      },
    })
  }

  async function handleSaveDetails() {
    const updates: Record<string, unknown> = {}
    if (editDesc !== (description ?? '')) updates.description = editDesc
    if (editMin !== minInstances) updates.min_instances = editMin
    if (editMax !== maxInstances) updates.max_instances = editMax
    if (Object.keys(updates).length === 0) {
      setDetailsDirty(false)
      return
    }
    setSavingDetails(true)
    try {
      await updateObject(pid, vid, objId, updates)
      const currentObjs = useStore.getState().project?.videos[vid]?.objects ?? {}
      useStore.getState().updateVideo({
        objects: {
          ...currentObjs,
          [objId]: { ...currentObjs[objId], ...updates },
        },
      })
      setDetailsDirty(false)
      addToast('Object details saved', 'success')
    } catch (e) {
      console.error('Failed to save object details:', e)
      addToast('Failed to save object details', 'error')
    } finally {
      setSavingDetails(false)
    }
  }

  async function handleAddInstance() {
    try {
      await addInstance(pid, vid, objId)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      console.warn('addInstance failed:', msg)
    }
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
          className="w-10 h-10 rounded-lg flex-shrink-0 border border-white/10 relative"
          style={{ background: `linear-gradient(135deg, ${color}88, ${color}44)` }}
        >
          {hasDescription && (
            <span
              className="absolute -top-1 -right-1 w-3.5 h-3.5 rounded-full bg-blue-500 text-white flex items-center justify-center text-[8px] font-bold"
              title="Text description set"
            >T</span>
          )}
          {maxInstances > 1 && (
            <span
              className="absolute -bottom-1 -right-1 w-3.5 h-3.5 rounded-full bg-purple-500 text-white flex items-center justify-center text-[8px] font-bold"
              title={`Multi-instance: ${minInstances}–${maxInstances}`}
            >{maxInstances}</span>
          )}
        </div>
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
          {hasDescription && !editing && (
            <span className="text-[10px] text-[#666] block truncate">{description}</span>
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
          onClick={e => { e.stopPropagation(); setShowDetails(!showDetails) }}
          className="p-1 text-[#555] hover:text-[#ccc] rounded"
          title="Edit description and instances"
        >
          {showDetails ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
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

      {/* Description and instances detail panel */}
      {showDetails && (
        <div className="px-3 pb-3 border-t border-[#222] pt-2.5" onClick={e => e.stopPropagation()}>
          <label className="block text-[10px] text-[#666] mb-1 uppercase tracking-wide">Text description</label>
          <textarea
            value={editDesc}
            onChange={e => { setEditDesc(e.target.value); setDetailsDirty(true) }}
            placeholder={`e.g. "${name}"`}
            rows={2}
            className="w-full text-xs bg-[#111] border border-[#333] rounded px-2 py-1.5 text-[#ccc] placeholder-[#444] resize-none focus:border-blue-500/60 focus:outline-none"
          />
          <div className="flex gap-3 mt-2">
            <div className="flex-1">
              <label className="block text-[10px] text-[#666] mb-1 uppercase tracking-wide">Min instances</label>
              <input
                type="number"
                min={1}
                max={10}
                value={editMin}
                onChange={e => { setEditMin(Number(e.target.value)); setDetailsDirty(true) }}
                className="w-full text-xs bg-[#111] border border-[#333] rounded px-2 py-1 text-[#ccc] focus:border-blue-500/60 focus:outline-none"
              />
            </div>
            <div className="flex-1">
              <label className="block text-[10px] text-[#666] mb-1 uppercase tracking-wide">Max instances</label>
              <input
                type="number"
                min={1}
                max={10}
                value={editMax}
                onChange={e => { setEditMax(Number(e.target.value)); setDetailsDirty(true) }}
                className="w-full text-xs bg-[#111] border border-[#333] rounded px-2 py-1 text-[#ccc] focus:border-blue-500/60 focus:outline-none"
              />
            </div>
          </div>
          <button
            onClick={handleSaveDetails}
            disabled={!detailsDirty || savingDetails}
            className={`mt-3 w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-medium border transition-colors
              ${detailsDirty
                ? 'bg-blue-600 border-blue-500 text-white hover:bg-blue-500'
                : 'border-[#333] text-[#555] cursor-not-allowed'}`}
          >
            {savingDetails ? <Loader size={12} className="animate-spin" /> : <Save size={12} />}
            {savingDetails ? 'Saving...' : 'Save Details'}
          </button>
          {maxInstances > 1 && isActive && (
            <button
              onClick={handleAddInstance}
              className="mt-2 flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300"
            >
              <PlusCircle size={11} /> Add instance slot
            </button>
          )}
        </div>
      )}

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
