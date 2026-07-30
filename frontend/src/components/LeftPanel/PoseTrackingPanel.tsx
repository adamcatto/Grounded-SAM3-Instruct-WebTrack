import React, { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight, Loader, Plus, Sparkles } from 'lucide-react'
import { addPoseObject, addPosePart, getProject, predictPose } from '../../api/client'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import { OBJECT_COLORS } from '../../utils/colors'

export default function PoseTrackingPanel() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project, currentVideoId, currentFrame, currentPoseObjectId, currentPosePartId,
    setCurrentPosePart, setProject, addToast,
  } = store
  const [newObject, setNewObject] = useState('')
  const [partDrafts, setPartDrafts] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [nextFrames, setNextFrames] = useState(30)
  const [predicting, setPredicting] = useState(false)
  const objects = useMemo(() => Object.values(video?.pose_objects ?? {}), [video?.pose_objects])
  if (!project || !video || !currentVideoId) return null

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

  async function runPrediction() {
    setPredicting(true)
    try {
      const result = await predictPose(project!.id, currentVideoId!, currentFrame, nextFrames)
      await refresh()
      addToast(`Tracked ${result.points} pose parts through frame ${result.end_frame}`, 'success')
    } catch (error: unknown) {
      const detail = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail ?? 'CoTracker3 pose prediction failed', 'error')
    } finally {
      setPredicting(false)
    }
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto p-3 text-sm">
      <div className="mb-3">
        <h2 className="font-semibold text-white">Pose tracking</h2>
        <p className="mt-1 text-xs text-[#777]">
          Frame {currentFrame} is the query frame. Select a part, then click its location once.
        </p>
      </div>

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
                      <button
                        key={part.id}
                        type="button"
                        onClick={() => setCurrentPosePart(object.id, part.id)}
                        className={`flex w-full items-center gap-2 rounded px-2 py-1.5 text-xs ${
                          selected ? 'bg-sky-500/20 text-sky-100 ring-1 ring-sky-500/50' : 'text-[#bbb] hover:bg-white/5'
                        }`}
                      >
                        <span className="h-3 w-3 rounded-full" style={{ backgroundColor: part.color }} />
                        <span>{part.name}</span>
                        <span className="ml-auto text-[10px] text-[#666]">
                          {annotation ? 'labeled' : 'click frame'}
                        </span>
                      </button>
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
          <input
            type="number"
            min={1}
            max={Math.max(1, video.num_frames - currentFrame - 1)}
            value={nextFrames}
            onChange={event => setNextFrames(Math.max(1, Number(event.target.value)))}
            className="mt-1 w-full rounded border border-[#444] bg-[#e5e7eb] px-2 py-1.5 text-[#111827]"
          />
        </label>
        <button
          type="button"
          disabled={predicting || objects.length === 0}
          onClick={() => void runPrediction()}
          className="flex w-full items-center justify-center gap-2 rounded bg-violet-600 py-2 font-medium text-white hover:bg-violet-500 disabled:bg-[#333] disabled:text-[#666]"
        >
          {predicting ? <Loader size={14} className="animate-spin" /> : <Sparkles size={14} />}
          {predicting ? 'Running CoTracker3…' : `Predict from frame ${currentFrame}`}
        </button>
      </div>
    </div>
  )
}
