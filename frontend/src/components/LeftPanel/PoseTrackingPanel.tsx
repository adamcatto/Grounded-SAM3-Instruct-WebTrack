import React, { useEffect, useMemo, useState } from 'react'
import { ChevronDown, ChevronLeft, ChevronRight, EyeOff, Loader, Plus, Redo2, SkipForward, Sparkles, Trash2, Undo2 } from 'lucide-react'
import {
  addPoseObject, addPosePart, cachePoseMemory, clearPoseAnnotationsForFrame,
  deletePoseAnnotation, getProject, predictPose, setPoseAnnotation, setPoseOccluded,
  skipPoseAnchor, unskipPoseAnchor, updateVideoMeta,
} from '../../api/client'
import type { PoseAnnotation } from '../../types'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { OBJECT_COLORS } from '../../utils/colors'
import { computeAnchorFrames, normalizeAnchorBatchSize } from '../../utils/anchorFrames'
import ValidatedIntegerInput from '../ValidatedIntegerInput'

export default function PoseTrackingPanel() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project, currentVideoId, currentFrame, currentPoseObjectId, currentPosePartId,
    setCurrentPosePart, setProject, addToast, setCurrentFrame,
    setPropagationStartFrame, updateVideo, pushHistory,
    undoStack, redoStack, historyBusy, undoLast, redoLast,
  } = store
  const [newObject, setNewObject] = useState('')
  const [partDrafts, setPartDrafts] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [nextFrames, setNextFrames] = useState(30)
  const [startFrame, setStartFrame] = useState(video?.start_frame ?? 0)
  const [predicting, setPredicting] = useState(false)
  const [anchorLabeling, setAnchorLabeling] = useState(false)
  const [anchorSpacingInput, setAnchorSpacingInput] = useState(String(video?.anchor_batch_size ?? 1000))
  const anchorSpacingValid = /^\d+$/.test(anchorSpacingInput)
  const anchorSpacing = anchorSpacingValid
    ? normalizeAnchorBatchSize(Number(anchorSpacingInput))
    : normalizeAnchorBatchSize(video?.anchor_batch_size ?? 1000)
  const objects = useMemo(() => Object.values(video?.pose_objects ?? {}), [video?.pose_objects])
  const anchorFrames = useMemo(
    () => computeAnchorFrames(video?.start_frame ?? 0, video?.num_frames ?? 1, normalizeAnchorBatchSize(anchorSpacing)),
    [video?.start_frame, video?.num_frames, anchorSpacing],
  )
  const currentAnchorIndex = anchorFrames.indexOf(currentFrame)
  const poseParts = useMemo(
    () => objects.flatMap(object => Object.values(object.parts ?? {}).map(part => [object.id, part.id] as const)),
    [objects],
  )
  const anchorComplete = (frame: number) => poseParts.length > 0 && poseParts.every(
    ([oid, partId]) => Boolean(video?.pose_annotations?.[oid]?.[partId]?.[String(frame)]),
  )
  const anchorCached = (frame: number) => (video?.pose_memory_frames ?? []).includes(frame)
  const anchorReady = (frame: number) => anchorCached(frame) && anchorComplete(frame)
  const anchorSkipped = (frame: number) => (video?.pose_skipped_anchor_frames ?? []).includes(frame)
  const anchorHandled = (frame: number) => anchorReady(frame) || anchorSkipped(frame)
  const nextUnhandledAnchor = (afterIndex: number) => (
    [...anchorFrames.slice(afterIndex + 1), ...anchorFrames.slice(0, afterIndex)]
      .find(frame => !anchorHandled(frame))
  )

  useEffect(() => {
    setStartFrame(video?.start_frame ?? 0)
    setAnchorSpacingInput(String(video?.anchor_batch_size ?? 1000))
  }, [video?.id, video?.start_frame, video?.anchor_batch_size])

  if (!project || !video || !currentVideoId) return null

  async function commitStartFrame(value = startFrame) {
    const frame = Math.max(0, Math.min(video!.num_frames - 2, Math.round(value)))
    setStartFrame(frame)
    await updateVideoMeta(project!.id, currentVideoId!, { start_frame: frame })
    updateVideo({ start_frame: frame })
    setPropagationStartFrame(frame)
    setCurrentFrame(frame)
  }

  async function refresh() {
    setProject(await getProject(project!.id))
  }

  async function createObject() {
    const name = newObject.trim()
    if (!name) return
    await addPoseObject(project!.id, currentVideoId!, name, OBJECT_COLORS[objects.length % OBJECT_COLORS.length])
    setNewObject('')
    await refresh()
  }

  async function createPart(oid: string, partCount: number) {
    const name = (partDrafts[oid] ?? '').trim()
    if (!name) return
    const color = OBJECT_COLORS[(partCount + objects.findIndex(object => object.id === oid) * 3) % OBJECT_COLORS.length]
    await addPosePart(project!.id, currentVideoId!, oid, name, color)
    setPartDrafts(drafts => ({ ...drafts, [oid]: '' }))
    await refresh()
  }

  async function restoreAnnotation(oid: string, partId: string, frame: number, annotation: PoseAnnotation) {
    if (annotation.visible === false) {
      await setPoseOccluded(project!.id, currentVideoId!, oid, partId, frame)
    } else if (annotation.x !== undefined && annotation.y !== undefined) {
      await setPoseAnnotation(project!.id, currentVideoId!, oid, partId, frame, annotation.x, annotation.y)
    }
  }

  async function deletePoint(oid: string, partId: string, point: PoseAnnotation) {
    const frame = currentFrame
    try {
      await deletePoseAnnotation(project!.id, currentVideoId!, oid, partId, frame)
      await refresh()
      pushHistory({
        labelUndo: 'Delete pose point',
        labelRedo: 'Delete pose point',
        undo: async () => {
          await restoreAnnotation(oid, partId, frame, point)
          await refresh()
        },
        redo: async () => {
          await deletePoseAnnotation(project!.id, currentVideoId!, oid, partId, frame)
          await refresh()
        },
      })
    } catch {
      addToast('Could not clear pose point', 'error')
    }
  }

  async function clearFramePoints() {
    const frame = currentFrame
    const snapshot: { oid: string; partId: string; point: PoseAnnotation }[] = []
    for (const object of objects) {
      for (const part of Object.values(object.parts ?? {})) {
        const point = video!.pose_annotations?.[object.id]?.[part.id]?.[String(frame)]
        if (point) snapshot.push({ oid: object.id, partId: part.id, point })
      }
    }
    if (snapshot.length === 0) return
    try {
      await clearPoseAnnotationsForFrame(project!.id, currentVideoId!, frame)
      await refresh()
      pushHistory({
        labelUndo: 'Clear frame pose points',
        labelRedo: 'Clear frame pose points',
        undo: async () => {
          for (const item of snapshot) {
            await restoreAnnotation(item.oid, item.partId, frame, item.point)
          }
          await refresh()
        },
        redo: async () => {
          await clearPoseAnnotationsForFrame(project!.id, currentVideoId!, frame)
          await refresh()
        },
      })
    } catch {
      addToast('Could not clear pose points on this frame', 'error')
    }
  }

  async function markSelectedOccluded() {
    if (!currentPoseObjectId || !currentPosePartId) return
    const oid = currentPoseObjectId
    const partId = currentPosePartId
    const frame = currentFrame
    const before = video!.pose_annotations?.[oid]?.[partId]?.[String(frame)]
    try {
      await setPoseOccluded(project!.id, currentVideoId!, oid, partId, frame)
      await refresh()
      pushHistory({
        labelUndo: 'Mark pose part occluded',
        labelRedo: 'Mark pose part occluded',
        undo: async () => {
          if (before) await restoreAnnotation(oid, partId, frame, before)
          else await deletePoseAnnotation(project!.id, currentVideoId!, oid, partId, frame)
          await refresh()
        },
        redo: async () => {
          await setPoseOccluded(project!.id, currentVideoId!, oid, partId, frame)
          await refresh()
        },
      })
    } catch {
      addToast('Could not mark pose part occluded', 'error')
    }
  }

  async function skipCurrentAnchor() {
    if (currentAnchorIndex < 0) return
    const frame = currentFrame
    try {
      await skipPoseAnchor(project!.id, currentVideoId!, frame)
      updateVideo({
        pose_skipped_anchor_frames: [...new Set([
          ...(video!.pose_skipped_anchor_frames ?? []), frame,
        ])].sort((a, b) => a - b),
      })
      const next = nextUnhandledAnchor(currentAnchorIndex)
      if (next === undefined) {
        setAnchorLabeling(false)
        addToast('All pose anchors are ready or skipped', 'success')
      } else {
        setCurrentFrame(next)
      }
      pushHistory({
        labelUndo: 'Skip pose anchor',
        labelRedo: 'Skip pose anchor',
        undo: async () => {
          await unskipPoseAnchor(project!.id, currentVideoId!, frame)
          await refresh()
        },
        redo: async () => {
          await skipPoseAnchor(project!.id, currentVideoId!, frame)
          await refresh()
        },
      })
    } catch {
      addToast('Could not skip this pose anchor', 'error')
    }
  }

  async function runPrediction() {
    setPredicting(true)
    try {
      const result = await predictPose(project!.id, currentVideoId!, startFrame, nextFrames)
      let status = result.status
      let latest = await getProject(project!.id)
      setProject(latest)
      while (status === 'running') {
        await new Promise(resolve => window.setTimeout(resolve, 750))
        latest = await getProject(project!.id)
        setProject(latest)
        status = latest.videos[currentVideoId!]?.pose_tracking?.status ?? 'failed'
      }
      const tracking = latest.videos[currentVideoId!]?.pose_tracking
      if (status === 'failed') throw new Error(tracking?.error ?? 'CoTracker3 pose prediction failed')
      addToast(`Tracked ${result.points} pose parts through frame ${tracking?.end_frame}`, 'success')
    } catch (error: unknown) {
      const detail = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail ?? (error as Error)?.message ?? 'CoTracker3 pose prediction failed', 'error')
    } finally {
      setPredicting(false)
    }
  }

  async function beginAnchorLabeling() {
    if (!anchorSpacingValid) return
    const spacing = normalizeAnchorBatchSize(Number(anchorSpacingInput))
    setAnchorSpacingInput(String(spacing))
    await updateVideoMeta(project!.id, currentVideoId!, { anchor_batch_size: spacing })
    updateVideo({ anchor_batch_size: spacing })
    setAnchorLabeling(true)
    setCurrentFrame(anchorFrames.find(frame => !anchorHandled(frame)) ?? anchorFrames[0])
  }

  async function commitAnchorAndAdvance() {
    if (currentAnchorIndex < 0 || !anchorComplete(currentFrame)) {
      addToast('Label every pose part on this anchor frame first', 'info')
      return
    }
    try {
      await cachePoseMemory(project!.id, currentVideoId!, currentFrame)
      updateVideo({
        pose_memory_frames: [...new Set([...(video!.pose_memory_frames ?? []), currentFrame])].sort((a, b) => a - b),
        pose_skipped_anchor_frames: (video!.pose_skipped_anchor_frames ?? []).filter(frame => frame !== currentFrame),
      })
      const next = nextUnhandledAnchor(currentAnchorIndex)
      if (next === undefined) {
        setAnchorLabeling(false)
        addToast('All pose anchor features are cached', 'success')
      } else {
        setCurrentFrame(next)
      }
    } catch (error: unknown) {
      const detail = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail ?? 'Could not cache this pose anchor', 'error')
    }
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto p-3 text-sm">
      <div className="mb-3">
        <h2 className="font-semibold text-white">Pose tracking</h2>
        <p className="mt-1 text-xs text-[#777]">
          Label pose anchors, then choose any query frame and predict forward.
        </p>
      </div>

      <div className="mb-3 rounded-lg border border-[#303030] bg-[#181818] p-2.5">
        <div className="mb-2 text-xs font-medium text-white">Pose anchor frames</div>
        <label className="block text-[11px] text-[#aaa]">
          Anchor spacing
          <input
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            value={anchorSpacingInput}
            disabled={anchorLabeling || predicting}
            onChange={event => setAnchorSpacingInput(event.target.value)}
            className="mt-1 w-full rounded border border-[#444] bg-[#e5e7eb] px-2 py-1.5 text-[#111827]"
          />
          {!anchorSpacingValid && (
            <span className="mt-1 block text-[11px] text-red-400">Only numbers are allowed</span>
          )}
        </label>
        {!anchorLabeling ? (
          <button
            type="button"
            disabled={predicting || poseParts.length === 0 || !anchorSpacingValid}
            onClick={() => void beginAnchorLabeling()}
            className="btn btn-secondary mt-2 w-full text-xs"
          >
            Label pose anchors ({anchorFrames.filter(anchorHandled).length}/{anchorFrames.length} handled)
          </button>
        ) : (
          <div className="mt-2 space-y-2">
            <div className="text-[11px] text-sky-200">
              Anchor {Math.max(1, currentAnchorIndex + 1)}/{anchorFrames.length} · frame {currentFrame}
            </div>
            <div className="flex gap-1">
              <button
                type="button"
                disabled={currentAnchorIndex <= 0}
                onClick={() => setCurrentFrame(anchorFrames[currentAnchorIndex - 1])}
                className="btn btn-ghost px-2"
              >
                <ChevronLeft size={14} />
              </button>
              <button
                type="button"
                onClick={() => void commitAnchorAndAdvance()}
                className="btn btn-secondary min-w-0 flex-1 text-xs"
              >
                Cache features & next
              </button>
              <button
                type="button"
                onClick={() => void skipCurrentAnchor()}
                className="btn btn-ghost flex items-center gap-1 px-2 text-xs"
                title="Skip this anchor without using it as memory"
              >
                <SkipForward size={13} /> Skip
              </button>
              <button
                type="button"
                disabled={currentAnchorIndex < 0 || currentAnchorIndex >= anchorFrames.length - 1}
                onClick={() => setCurrentFrame(anchorFrames[currentAnchorIndex + 1])}
                className="btn btn-ghost px-2"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>

      <label className="mb-3 block text-xs text-[#aaa]">
        Start frame
        <div className="mt-1 flex gap-2">
          <ValidatedIntegerInput
            value={startFrame}
            min={0}
            max={Math.max(0, video.num_frames - 2)}
            onCommit={value => {
              setStartFrame(value)
              void commitStartFrame(value)
            }}
            disabled={predicting}
            wrapperClassName="min-w-0 flex-1"
            className="w-full rounded border border-[#333] bg-[#1a1a1a] px-2 py-1.5 text-xs text-[#ddd]"
          />
        </div>
      </label>

      <div className="mb-3 flex gap-2">
        <input
          value={newObject}
          onChange={event => setNewObject(event.target.value)}
          onKeyDown={event => { if (event.key === 'Enter') void createObject() }}
          placeholder="Top-level object (e.g. mouse)"
          className="min-w-0 flex-1 rounded border border-[#3a3a3a] bg-[#1b1b1b] px-2 py-1.5 text-xs text-white"
        />
        <button onClick={() => void createObject()} className="btn btn-secondary px-2" title="Add pose object">
          <Plus size={14} />
        </button>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-1">
        <button
          type="button"
          disabled={!currentPoseObjectId || !currentPosePartId}
          onClick={() => void markSelectedOccluded()}
          className="btn btn-ghost flex items-center justify-center gap-1 text-[11px]"
          title="Record the selected pose part as occluded on this frame"
        >
          <EyeOff size={12} /> Mark occluded
        </button>
        <button
          type="button"
          disabled={!poseParts.some(([oid, partId]) => Boolean(video.pose_annotations?.[oid]?.[partId]?.[String(currentFrame)]))}
          onClick={() => void clearFramePoints()}
          className="btn btn-ghost flex items-center justify-center gap-1 text-[11px]"
          title={`Clear all labeled pose points on frame ${currentFrame}`}
        >
          <Trash2 size={12} /> Clear frame
        </button>
        <button
          type="button"
          disabled={undoStack.length === 0 || historyBusy}
          onClick={() => void undoLast()}
          className="btn btn-ghost flex items-center justify-center gap-1 text-[11px]"
        >
          <Undo2 size={12} /> Undo
        </button>
        <button
          type="button"
          disabled={redoStack.length === 0 || historyBusy}
          onClick={() => void redoLast()}
          className="btn btn-ghost flex items-center justify-center gap-1 text-[11px]"
        >
          <Redo2 size={12} /> Redo
        </button>
      </div>

      <div className="space-y-2">
        {objects.map(object => {
          const parts = Object.values(object.parts ?? {})
          const isOpen = expanded[object.id] ?? true
          return (
            <div key={object.id} className="rounded-lg border border-[#303030] bg-[#181818]">
              <button
                type="button"
                onClick={() => setExpanded(state => ({ ...state, [object.id]: !isOpen }))}
                className="flex w-full items-center gap-2 px-2.5 py-2 text-left"
              >
                {isOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: object.color }} />
                <span className="font-medium text-white">{object.name}</span>
                <span className="ml-auto text-[10px] text-[#666]">{parts.length} parts</span>
              </button>
              {isOpen && (
                <div className="space-y-1 border-t border-[#292929] p-2">
                  {parts.map(part => {
                    const annotation = video.pose_annotations?.[object.id]?.[part.id]?.[String(currentFrame)]
                    const selected = currentPoseObjectId === object.id && currentPosePartId === part.id
                    return (
                      <div
                        key={part.id}
                        className={`flex w-full items-center gap-2 rounded px-2 py-1.5 text-xs ${
                          selected ? 'bg-sky-500/20 text-sky-100 ring-1 ring-sky-500/50' : 'text-[#bbb] hover:bg-white/5'
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => setCurrentPosePart(object.id, part.id)}
                          className="flex min-w-0 flex-1 items-center gap-2 text-left"
                        >
                          <span className="h-3 w-3 flex-shrink-0 rounded-full" style={{ backgroundColor: part.color }} />
                          <span>{part.name}</span>
                        </button>
                        <span className="ml-auto text-[10px] text-[#666]">
                          {annotation?.visible === false ? 'occluded' : annotation ? 'labeled' : 'click frame'}
                        </span>
                        {annotation && (
                          <button
                            type="button"
                            onClick={() => void deletePoint(object.id, part.id, annotation)}
                            className="rounded p-0.5 text-[#777] hover:bg-red-500/15 hover:text-red-300"
                            title={`Clear ${part.name} on frame ${currentFrame}`}
                          >
                            <Trash2 size={12} />
                          </button>
                        )}
                      </div>
                    )
                  })}
                  <div className="flex gap-1 pt-1">
                    <input
                      value={partDrafts[object.id] ?? ''}
                      onChange={event => setPartDrafts(drafts => ({ ...drafts, [object.id]: event.target.value }))}
                      onKeyDown={event => { if (event.key === 'Enter') void createPart(object.id, parts.length) }}
                      placeholder="Part (snout, midback…)"
                      className="min-w-0 flex-1 rounded border border-[#333] bg-[#222] px-2 py-1 text-[11px] text-white"
                    />
                    <button onClick={() => void createPart(object.id, parts.length)} className="btn btn-ghost p-1">
                      <Plus size={13} />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="mt-auto space-y-2 border-t border-[#303030] pt-3">
        <label className="block text-xs text-[#aaa]">
          Predict next N frames
          <ValidatedIntegerInput
            value={nextFrames}
            min={1}
            max={Math.max(1, video.num_frames - startFrame - 1)}
            onCommit={setNextFrames}
            wrapperClassName="mt-1"
            className="w-full rounded border border-[#444] bg-[#e5e7eb] px-2 py-1.5 text-[#111827]"
          />
        </label>
        <button
          type="button"
          disabled={predicting || objects.length === 0}
          onClick={() => void runPrediction()}
          className="flex w-full items-center justify-center gap-2 rounded bg-violet-600 py-2 font-medium text-white hover:bg-violet-500 disabled:bg-[#333] disabled:text-[#666]"
        >
          {predicting ? <Loader size={14} className="animate-spin" /> : <Sparkles size={14} />}
          {predicting ? 'Running CoTracker3…' : `Predict from frame ${startFrame}`}
        </button>
      </div>
    </div>
  )
}
