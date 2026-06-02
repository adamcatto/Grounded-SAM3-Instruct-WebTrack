import React, { useState, useEffect } from 'react'
import { Pencil, Trash2, MousePointer, MinusCircle, Check, ChevronDown, ChevronUp, Save, Loader, Eye, EyeOff, ZoomIn, ZoomOut, Plus } from 'lucide-react'
import { useStore } from '../../store/useStore'
import { renameObject, removeObject, clearObjectFramePoints, updateObject, rebuildFromConfig, replaceFramePromptsData, restoreObjectSnapshot, getSavedMask } from '../../api/client'
import { applyRebuildMasksToStore, localAnnotationsToPointPrompts } from '../../history/applyRebuild'
import type { ObjectKind } from '../../types'

interface Props {
  objId: string
  name: string
  color: string
  isActive: boolean
  onSelect: () => void
  description?: string
  /** Nesting depth (0 = top-level); used for indentation. */
  depth?: number
  /** 'point' sub-objects render a small keypoint badge. */
  kind?: ObjectKind
  /** Opens the "add sub-object" form for this object in the parent panel. */
  onAddSub?: () => void
}

export default function ObjectCard({ objId, name, color, isActive, onSelect, description, depth = 0, kind = 'segmentation', onAddSub }: Props) {
  const {
    project, currentVideoId, currentFrame,
    pointMode, setPointMode, setCurrentObject,
    clearLocalPoints, clearLocalPointsForFrame, setCurrentFrameMasks, currentFrameMasks,
    savedMaskCache, setSavedMask,
    addToast,
    toggleObjectVisibility, isObjectVisible, setZoomToObject, zoomToObjectId,
  } = useStore()

  const visible = isObjectVisible(objId)
  const isZoomTarget = zoomToObjectId === objId

  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState(name)
  const [removing, setRemoving] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [editDesc, setEditDesc] = useState(description ?? '')
  const [savingDetails, setSavingDetails] = useState(false)
  const [detailsDirty, setDetailsDirty] = useState(false)

  // Sync state when props change (e.g., after save updates store)
  useEffect(() => {
    setEditDesc(description ?? '')
  }, [description])

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const hasDescription = (description ?? '').trim().length > 0

  async function handleRename() {
    if (!editName.trim() || editName === name) { setEditing(false); return }
    const prevName = name
    const nextName = editName.trim()
    await renameObject(pid, vid, objId, nextName)
    setEditing(false)
    const o = useStore.getState().project?.videos[vid]?.objects
    useStore.getState().updateVideo({
      objects: {
        ...o,
        // Preserve hierarchy fields (parent_id/kind/...) — only the name changes.
        [objId]: { ...(o?.[objId] ?? { id: objId, name, color, description }), name: nextName },
      },
    })
    useStore.getState().pushHistory({
      labelUndo: 'Rename object',
      labelRedo: 'Rename object',
      undo: async () => {
        await renameObject(pid, vid, objId, prevName)
        const cur = useStore.getState().project?.videos[vid]?.objects
        const base = cur?.[objId]
        if (base) {
          useStore.getState().updateVideo({
            objects: { ...cur, [objId]: { ...base, name: prevName } },
          })
        }
      },
      redo: async () => {
        await renameObject(pid, vid, objId, nextName)
        const cur = useStore.getState().project?.videos[vid]?.objects
        const base = cur?.[objId]
        if (base) {
          useStore.getState().updateVideo({
            objects: { ...cur, [objId]: { ...base, name: nextName } },
          })
        }
      },
    })
  }

  async function handleSaveDetails() {
    const updates: Record<string, unknown> = {}
    if (editDesc !== (description ?? '')) updates.description = editDesc
    if (Object.keys(updates).length === 0) {
      setDetailsDirty(false)
      return
    }
    const prevDesc = description ?? ''
    const nextDesc = editDesc
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
      useStore.getState().pushHistory({
        labelUndo: 'Object description',
        labelRedo: 'Object description',
        undo: async () => {
          await updateObject(pid, vid, objId, { description: prevDesc })
          const cur = useStore.getState().project?.videos[vid]?.objects ?? {}
          const base = cur[objId]
          if (base) {
            useStore.getState().updateVideo({
              objects: { ...cur, [objId]: { ...base, description: prevDesc } },
            })
          }
        },
        redo: async () => {
          await updateObject(pid, vid, objId, { description: nextDesc })
          const cur = useStore.getState().project?.videos[vid]?.objects ?? {}
          const base = cur[objId]
          if (base) {
            useStore.getState().updateVideo({
              objects: { ...cur, [objId]: { ...base, description: nextDesc } },
            })
          }
        },
      })
    } catch (e) {
      console.error('Failed to save object details:', e)
      addToast('Failed to save object details', 'error')
    } finally {
      setSavingDetails(false)
    }
  }

  async function handleRemove() {
    if (!confirm(`Remove object "${name}"?`)) return
    const v = useStore.getState().project?.videos[vid]
    const objSnap = v?.objects[objId]
    if (!objSnap) return
    const pointSnap: Record<string, { points: [number, number][]; labels: number[] }> = v?.point_prompts?.[objId]
      ? JSON.parse(JSON.stringify(v.point_prompts[objId]))
      : {}
    const ig = (v as { instance_groups?: Record<string, number[]> }).instance_groups?.[objId]

    // Backend removal cascades to descendants — mirror that in the store.
    const allObjsNow = useStore.getState().project?.videos[vid]?.objects ?? {}
    const removedIds = new Set<string>([objId])
    {
      const stack = [objId]
      while (stack.length) {
        const cur = stack.pop() as string
        for (const [k, o] of Object.entries(allObjsNow)) {
          if (o.parent_id != null && String(o.parent_id) === cur && !removedIds.has(k)) {
            removedIds.add(k)
            stack.push(k)
          }
        }
      }
    }

    setRemoving(true)
    try {
      await removeObject(pid, vid, objId)
      for (const rid of removedIds) clearLocalPoints(rid)
      const newMasks = { ...useStore.getState().currentFrameMasks }
      for (const rid of removedIds) delete newMasks[rid]
      useStore.getState().setCurrentFrameMasks(newMasks, useStore.getState().currentFrameMasksFrame ?? useStore.getState().currentFrame)
      useStore.getState().updateVideo({
        objects: Object.fromEntries(
          Object.entries(useStore.getState().project?.videos[vid]?.objects ?? {}).filter(([k]) => !removedIds.has(k))
        ),
        point_prompts: (() => {
          const pp = { ...useStore.getState().project?.videos[vid]?.point_prompts }
          for (const rid of removedIds) delete pp[rid]
          return pp
        })(),
      })
      if (removedIds.has(useStore.getState().currentObjectId ?? '')) {
        setCurrentObject(null)
      }

      const promptFrames = Object.keys(pointSnap).map(Number)
      useStore.getState().pushHistory({
        labelUndo: 'Remove object',
        labelRedo: 'Remove object',
        undo: async () => {
          await restoreObjectSnapshot(pid, vid, {
            id: objSnap.id,
            name: objSnap.name,
            color: objSnap.color,
            description: objSnap.description,
            min_instances: (objSnap as { min_instances?: number }).min_instances,
            max_instances: (objSnap as { max_instances?: number }).max_instances,
          }, pointSnap, ig ?? null)
          const la = JSON.parse(JSON.stringify(useStore.getState().localAnnotations))
          for (const [fk, pr] of Object.entries(pointSnap)) {
            la[objId] = la[objId] ?? {}
            la[objId][fk] = {
              points: pr.points.map(([x, y], i) => ({ x, y, label: pr.labels[i] as 0 | 1 })),
            }
          }
          useStore.setState({ localAnnotations: la })
          const st = useStore.getState()
          const fm = st.project?.videos[vid]
          if (fm) {
            st.updateVideo({
              objects: { ...fm.objects, [objId]: objSnap },
              point_prompts: { ...fm.point_prompts, [objId]: pointSnap },
            })
          }
          const frames = promptFrames.length > 0 ? promptFrames : [st.currentFrame]
          const anchorNow = st.anchorPhase
          const uniq = [...new Set(frames)]
          const rb = await rebuildFromConfig(pid, vid, uniq, anchorNow, anchorNow && uniq.length === 1 ? uniq[0] : null)
          applyRebuildMasksToStore(rb.masks_by_frame)
        },
        redo: async () => {
          await removeObject(pid, vid, objId)
          clearLocalPoints(objId)
          const nm = { ...useStore.getState().currentFrameMasks }
          delete nm[objId]
          useStore.getState().setCurrentFrameMasks(nm, useStore.getState().currentFrameMasksFrame ?? useStore.getState().currentFrame)
          const pp = { ...useStore.getState().project?.videos[vid]?.point_prompts ?? {} }
          delete pp[objId]
          useStore.getState().updateVideo({
            objects: Object.fromEntries(
              Object.entries(useStore.getState().project?.videos[vid]?.objects ?? {}).filter(([k]) => k !== objId)
            ),
            point_prompts: pp,
          })
          if (useStore.getState().currentObjectId === objId) setCurrentObject(null)
        },
      })
    } finally {
      setRemoving(false)
    }
  }

  async function handleClear() {
    const st0 = useStore.getState()
    const key = String(currentFrame)
    const prevAnn = st0.localAnnotations[objId]?.[key]
    const prevLiveMask = st0.currentFrameMasks[objId]
    let prevSavedMaskB64: string | undefined = st0.savedMaskCache[currentFrame]?.[objId]
    if (prevSavedMaskB64 === undefined && prevAnn) {
      try {
        const gm = await getSavedMask(pid, vid, currentFrame)
        prevSavedMaskB64 = gm.masks?.[objId]
      } catch { /* ignore */ }
    }

    clearLocalPointsForFrame(objId, currentFrame)
    const newLive = { ...useStore.getState().currentFrameMasks }
    delete newLive[objId]
    const cf = useStore.getState().currentFrameMasksFrame
    setCurrentFrameMasks(newLive, cf ?? currentFrame)
    const frameSaved = useStore.getState().savedMaskCache[currentFrame]
    if (frameSaved && frameSaved[objId]) {
      const updated = { ...frameSaved }
      delete updated[objId]
      setSavedMask(currentFrame, updated)
    }
    try {
      await clearObjectFramePoints(pid, vid, objId, currentFrame)
    } catch { /* ignore */ }

    const fIdx = currentFrame
    useStore.getState().pushHistory({
      labelUndo: 'Clear selection',
      labelRedo: 'Clear selection',
      undo: async () => {
        const pts = prevAnn ? prevAnn.points.map(p => [p.x, p.y] as [number, number]) : []
        const labs = prevAnn ? prevAnn.points.map(p => p.label as number) : []
        await replaceFramePromptsData(pid, vid, objId, fIdx, pts, labs)
        const la = JSON.parse(JSON.stringify(useStore.getState().localAnnotations)) as typeof st0.localAnnotations
        if (prevAnn) {
          la[objId] = { ...la[objId], [key]: prevAnn }
        } else {
          const of = la[objId]
          if (of) {
            delete of[key]
            if (Object.keys(of).length === 0) delete la[objId]
          }
        }
        useStore.setState({ localAnnotations: la })
        useStore.getState().updateVideo({ point_prompts: localAnnotationsToPointPrompts(la) })
        const anchorNow = useStore.getState().anchorPhase
        const rb = await rebuildFromConfig(pid, vid, [fIdx], anchorNow, anchorNow ? fIdx : null)
        applyRebuildMasksToStore(rb.masks_by_frame)
        if (prevSavedMaskB64 !== undefined) {
          const merged = { ...(useStore.getState().savedMaskCache[fIdx] ?? {}) }
          merged[objId] = prevSavedMaskB64
          useStore.getState().setSavedMask(fIdx, merged)
        }
        if (prevLiveMask !== undefined && useStore.getState().currentFrame === fIdx) {
          const lm = { ...useStore.getState().currentFrameMasks, [objId]: prevLiveMask }
          useStore.getState().setCurrentFrameMasks(lm, fIdx)
        }
      },
      redo: async () => {
        useStore.getState().clearLocalPointsForFrame(objId, fIdx)
        const nl = { ...useStore.getState().currentFrameMasks }
        delete nl[objId]
        useStore.getState().setCurrentFrameMasks(nl, useStore.getState().currentFrameMasksFrame ?? fIdx)
        const fs = useStore.getState().savedMaskCache[fIdx]
        if (fs && fs[objId]) {
          const u = { ...fs }
          delete u[objId]
          useStore.getState().setSavedMask(fIdx, u)
        }
        try {
          await clearObjectFramePoints(pid, vid, objId, fIdx)
        } catch { /* ignore */ }
      },
    })
  }

  return (
    <div
      className={`rounded-xl border transition-all cursor-pointer
        ${isActive
          ? 'border-[#444] bg-[#1e1e1e]'
          : 'border-[#2a2a2a] bg-[#161616] hover:border-[#333] hover:bg-[#1a1a1a]'
        } ${!visible ? 'opacity-50' : ''}`}
      style={depth > 0 ? { marginLeft: depth * 14, borderLeft: `2px solid ${color}66` } : undefined}
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
            <span className="text-sm font-medium text-white truncate flex items-center gap-1.5">
              {name}
              {kind === 'point' && (
                <span className="text-[8px] uppercase tracking-wide px-1 py-0.5 rounded bg-[#333] text-[#aaa]" title="Keypoint sub-object">pt</span>
              )}
            </span>
          )}
          {hasDescription && !editing && (
            <span className="text-[10px] text-[#666] block truncate">{description}</span>
          )}
        </div>
        {/* Visibility toggle (cascade-aware) */}
        <button
          onClick={e => { e.stopPropagation(); toggleObjectVisibility(objId) }}
          className="p-1 text-[#555] hover:text-[#ccc] rounded"
          title={visible ? 'Hide mask' : 'Show mask'}
        >
          {visible ? <Eye size={13} /> : <EyeOff size={13} />}
        </button>
        {/* Zoom-to-bbox toggle */}
        <button
          onClick={e => { e.stopPropagation(); setZoomToObject(isZoomTarget ? null : objId) }}
          className={`p-1 rounded ${isZoomTarget ? 'text-blue-400' : 'text-[#555] hover:text-[#ccc]'}`}
          title={isZoomTarget ? 'Zoom out' : 'Zoom to this object'}
        >
          {isZoomTarget ? <ZoomOut size={13} /> : <ZoomIn size={13} />}
        </button>
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
          title="Edit description"
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

      {/* Description detail panel */}
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
          {onAddSub && (
            <button
              onClick={e => { e.stopPropagation(); onAddSub() }}
              className="w-full mt-1 py-1 text-xs text-[#666] hover:text-[#aaa] flex items-center justify-center gap-1 transition-colors"
            >
              <Plus size={11} /> Add sub-object
            </button>
          )}
        </div>
      )}

      {/* Inactive: Edit/Clear links */}
      {!isActive && (
        <div className="flex gap-3 px-3 pb-3 flex-wrap">
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
          {onAddSub && (
            <button
              onClick={e => { e.stopPropagation(); onAddSub() }}
              className="text-xs text-[#666] hover:text-[#aaa] flex items-center gap-1"
            >
              <Plus size={11} /> Sub-object
            </button>
          )}
        </div>
      )}
    </div>
  )
}
