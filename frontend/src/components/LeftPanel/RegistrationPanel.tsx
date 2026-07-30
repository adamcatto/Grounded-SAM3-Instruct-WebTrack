import React, { useEffect, useMemo, useState } from 'react'
import { CheckCircle2, ChevronLeft, ChevronRight, ExternalLink, Pentagon, Redo2, ScanLine, Trash2, Undo2, WandSparkles } from 'lucide-react'
import {
  applyRegistrationMorphology,
  clearRegistrationVideo,
  computeRegistration,
  fitRegistrationEdges,
  getProject,
  getRegistrationMask,
  redoRegistrationMorphology,
  setRegistrationPolygon,
  undoRegistrationMorphology,
} from '../../api/client'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import type { VideoRegistration } from '../../types'
import RegisteredPreviewModal from '../RegisteredPreviewModal'

export default function RegistrationPanel() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project,
    currentVideoId,
    pointMode,
    setPointMode,
    enterRegistrationVideo,
    setProject,
    setDrawerOpen,
    addToast,
    setCurrentFrameMasks,
    setSavedMask,
    clearLocalPoints,
    registrationTool,
    setRegistrationTool,
    registrationPolygonPoints,
    clearRegistrationPolygonPoints,
    registrationActiveEdge,
    setRegistrationActiveEdge,
    registrationEdgePoints,
    clearRegistrationEdgePoints,
  } = store
  const [computing, setComputing] = useState(false)
  const [morphologyBusy, setMorphologyBusy] = useState(false)
  const [morphologyOperation, setMorphologyOperation] = useState<'opening' | 'closing'>('opening')
  const [kernelSize, setKernelSize] = useState(5)
  const [previewOpen, setPreviewOpen] = useState(false)

  const videos = useMemo(
    () => project ? Object.values(project.videos) : [],
    [project],
  )
  const index = videos.findIndex(v => v.id === currentVideoId)
  const registration = project?.registration
  const entry = currentVideoId ? registration?.videos[currentVideoId] : undefined
  const labeled = videos.filter(v => {
    const item = registration?.videos[v.id]
    return item?.labeled || item?.registered
  }).length
  const allLabeled = videos.length > 0 && labeled === videos.length
  const registeredVideos = videos.filter(v => registration?.videos[v.id]?.registered)
  const maxKernelSize = Math.max(1, Math.min(video?.width ?? 1, video?.height ?? 1))
  const history = entry?.morphology_history ?? []
  const historyCursor = Math.max(0, Math.min(entry?.morphology_cursor ?? 0, history.length))

  function applyRegistrationResult(result: { mask: string | null; registration: VideoRegistration }) {
    const currentProject = useStore.getState().project
    if (!currentProject?.registration || !currentVideoId) return
    setProject({
      ...currentProject,
      registration: {
        ...currentProject.registration,
        status: 'labeling',
        videos: {
          ...currentProject.registration.videos,
          [currentVideoId]: result.registration,
        },
      },
    })
    if (result.mask) {
      const masks = { __registration_floor__: result.mask }
      setCurrentFrameMasks(masks, 0)
      setSavedMask(0, masks)
    } else {
      setCurrentFrameMasks({}, 0)
      setSavedMask(0, {})
    }
  }

  useEffect(() => {
    if (!project?.id || !currentVideoId || !entry?.labeled) return
    let cancelled = false
    void getRegistrationMask(project.id, currentVideoId).then(result => {
      if (cancelled || !result.mask) return
      const masks = { __registration_floor__: result.mask }
      setCurrentFrameMasks(masks, 0)
      setSavedMask(0, masks)
    })
    return () => { cancelled = true }
  }, [project?.id, currentVideoId, entry?.labeled, setCurrentFrameMasks, setSavedMask])

  if (!project || !video || !registration) return null

  async function generate() {
    setComputing(true)
    try {
      await computeRegistration(project!.id)
      const fresh = await getProject(project!.id)
      setProject(fresh)
      addToast('Registration homographies generated', 'success')
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail || 'Could not generate registration parameters', 'error')
    } finally {
      setComputing(false)
    }
  }

  async function clearMaskAndPoints() {
    if (!currentVideoId || !window.confirm('Clear all registration points, the floor mask, and morphology history for this camera?')) return
    setMorphologyBusy(true)
    try {
      const result = await clearRegistrationVideo(project!.id, currentVideoId)
      clearLocalPoints('__registration_floor__')
      clearRegistrationPolygonPoints()
      clearRegistrationEdgePoints()
      applyRegistrationResult(result)
      addToast('Registration points and mask cleared', 'success')
    } catch {
      addToast('Could not clear registration mask', 'error')
    } finally {
      setMorphologyBusy(false)
    }
  }

  async function fitEdges() {
    if (!currentVideoId) return
    const ready = (['top', 'right', 'bottom', 'left'] as const).every(
      edge => registrationEdgePoints[edge].length >= 2,
    )
    if (!ready) {
      addToast('Add at least two points to each of the four edges', 'error')
      return
    }
    setMorphologyBusy(true)
    try {
      const result = await fitRegistrationEdges(project!.id, currentVideoId, registrationEdgePoints)
      const fresh = await getProject(project!.id)
      setProject(fresh)
      addToast(
        `Lens and edge calibration fitted (straightness RMS ${result.straightness_rms_pixels.toFixed(2)} px)`,
        'success',
      )
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail || 'Could not fit edge calibration', 'error')
    } finally {
      setMorphologyBusy(false)
    }
  }

  async function usePolygonHull() {
    if (!currentVideoId || registrationPolygonPoints.length < 3) return
    setMorphologyBusy(true)
    try {
      const result = await setRegistrationPolygon(
        project!.id,
        currentVideoId,
        registrationPolygonPoints,
      )
      clearLocalPoints('__registration_floor__')
      applyRegistrationResult(result)
      addToast('Convex polygon hull applied as floor mask', 'success')
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail || 'Could not create polygon mask', 'error')
    } finally {
      setMorphologyBusy(false)
    }
  }

  async function applyMorphology() {
    if (!currentVideoId) return
    if (!Number.isInteger(kernelSize) || kernelSize < 1 || kernelSize > maxKernelSize) {
      addToast(`Filter size must be an integer from 1 to ${maxKernelSize}`, 'error')
      return
    }
    setMorphologyBusy(true)
    try {
      const result = await applyRegistrationMorphology(
        project!.id,
        currentVideoId,
        morphologyOperation,
        kernelSize,
      )
      applyRegistrationResult(result)
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail || 'Could not apply morphology', 'error')
    } finally {
      setMorphologyBusy(false)
    }
  }

  async function stepMorphology(direction: 'undo' | 'redo') {
    if (!currentVideoId) return
    setMorphologyBusy(true)
    try {
      const result = direction === 'undo'
        ? await undoRegistrationMorphology(project!.id, currentVideoId)
        : await redoRegistrationMorphology(project!.id, currentVideoId)
      applyRegistrationResult(result)
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail || `Could not ${direction} morphology`, 'error')
    } finally {
      setMorphologyBusy(false)
    }
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="px-3 py-3 border-b border-[#252525]">
        <div className="flex items-center gap-2 text-sky-300">
          <ScanLine size={16} />
          <span className="text-sm font-semibold">Register videos</span>
        </div>
        <p className="text-xs text-[#777] mt-1 leading-relaxed">
          Segment the visible square floor on frame 0. Positive points include floor;
          negative points remove walls, animals, or surrounding areas.
        </p>
      </div>

      <div className="p-3 space-y-3 overflow-y-auto flex-1">
        <div className="rounded-lg border border-sky-700/40 bg-sky-500/10 p-3">
          <div className="flex items-center gap-2">
            {(entry?.labeled || entry?.registered) && <CheckCircle2 size={14} className="text-emerald-400" />}
            <p className="text-sm text-sky-100 font-medium truncate">{video.name}</p>
          </div>
          <p className="text-xs text-[#777] mt-1">
            Camera {index + 1} of {videos.length} · first frame (frame 0)
          </p>
        </div>

        <div className="rounded-lg border border-[#333] bg-[#181818] p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="h-3 w-3 rounded-sm bg-sky-400" />
            <span className="text-sm text-[#ddd] font-medium">Common floor</span>
          </div>
          <div className="grid grid-cols-3 gap-2 mb-2">
            <button
              type="button"
              onClick={() => setRegistrationTool('points')}
              className={`px-2 py-2 rounded text-xs ${
                registrationTool === 'points' ? 'bg-sky-600 text-white' : 'bg-[#252525] text-[#aaa]'
              }`}
            >
              Point prompts
            </button>
            <button
              type="button"
              onClick={() => setRegistrationTool('polygon')}
              className={`px-2 py-2 rounded text-xs ${
                registrationTool === 'polygon' ? 'bg-amber-500 text-black' : 'bg-[#252525] text-[#aaa]'
              }`}
            >
              <Pentagon size={12} className="inline mr-1" /> Polygon
            </button>
            <button
              type="button"
              onClick={() => setRegistrationTool('edges')}
              className={`px-2 py-2 rounded text-xs ${
                registrationTool === 'edges' ? 'bg-violet-500 text-white' : 'bg-[#252525] text-[#aaa]'
              }`}
            >
              Curved edges
            </button>
          </div>
          {registrationTool === 'points' ? (
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setPointMode('add')}
              className={`px-2 py-2 rounded text-xs ${
                pointMode === 'add' ? 'bg-emerald-600 text-white' : 'bg-[#252525] text-[#aaa]'
              }`}
            >
              Positive point
            </button>
            <button
              type="button"
              onClick={() => setPointMode('remove')}
              className={`px-2 py-2 rounded text-xs ${
                pointMode === 'remove' ? 'bg-red-600 text-white' : 'bg-[#252525] text-[#aaa]'
              }`}
            >
              Negative point
            </button>
          </div>
          ) : registrationTool === 'polygon' ? (
            <div className="space-y-2">
              <p className="text-[10px] text-[#777]">
                Click at least three boundary points. The filled mask will be their convex hull.
              </p>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={registrationPolygonPoints.length === 0 || morphologyBusy}
                  onClick={clearRegistrationPolygonPoints}
                  className="px-2 py-2 rounded text-xs bg-[#252525] text-[#aaa] disabled:opacity-40"
                >
                  Clear vertices ({registrationPolygonPoints.length})
                </button>
                <button
                  type="button"
                  disabled={registrationPolygonPoints.length < 3 || morphologyBusy}
                  onClick={() => void usePolygonHull()}
                  className="px-2 py-2 rounded text-xs bg-amber-500 hover:bg-amber-400 text-black font-medium disabled:bg-[#292929] disabled:text-[#666]"
                >
                  Use convex hull
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <p className="text-[10px] text-[#777]">
                Select an edge, then add points along its visible curved boundary. Use 3+ points
                where possible; every edge requires at least 2.
              </p>
              <div className="grid grid-cols-2 gap-1.5">
                {([
                  ['top', 'Top', 'bg-yellow-400'],
                  ['right', 'Right', 'bg-cyan-400'],
                  ['bottom', 'Bottom', 'bg-pink-400'],
                  ['left', 'Left', 'bg-lime-400'],
                ] as const).map(([edge, label, color]) => (
                  <button
                    key={edge}
                    type="button"
                    onClick={() => setRegistrationActiveEdge(edge)}
                    className={`flex items-center gap-1.5 px-2 py-1.5 rounded text-xs ${
                      registrationActiveEdge === edge
                        ? 'ring-1 ring-white/60 bg-[#333] text-white'
                        : 'bg-[#252525] text-[#aaa]'
                    }`}
                  >
                    <span className={`h-2.5 w-2.5 rounded-full ${color}`} />
                    <span>{label} ({registrationEdgePoints[edge].length})</span>
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={registrationEdgePoints[registrationActiveEdge].length === 0}
                  onClick={() => clearRegistrationEdgePoints(registrationActiveEdge)}
                  className="px-2 py-2 rounded text-xs bg-[#252525] text-[#aaa] disabled:opacity-40"
                >
                  Clear active edge
                </button>
                <button
                  type="button"
                  disabled={morphologyBusy}
                  onClick={() => clearRegistrationEdgePoints()}
                  className="px-2 py-2 rounded text-xs bg-[#252525] text-[#aaa] disabled:opacity-40"
                >
                  Clear all edges
                </button>
              </div>
              <button
                type="button"
                disabled={
                  morphologyBusy ||
                  !(['top', 'right', 'bottom', 'left'] as const).every(
                    edge => registrationEdgePoints[edge].length >= 2,
                  )
                }
                onClick={() => void fitEdges()}
                className="w-full px-2 py-2 rounded text-xs font-medium bg-violet-600 hover:bg-violet-500 text-white disabled:bg-[#292929] disabled:text-[#666]"
              >
                Fit lens + overhead square
              </button>
              {entry?.calibration_source === 'partial_edges_radial_distortion' && (
                <p className="text-[10px] text-emerald-400">
                  Fitted · straightness RMS {(entry.straightness_rms_pixels ?? 0).toFixed(2)} px
                </p>
              )}
            </div>
          )}
          <button
            type="button"
            disabled={morphologyBusy}
            onClick={() => void clearMaskAndPoints()}
            className="mt-2 w-full flex items-center justify-center gap-1.5 px-2 py-2 rounded text-xs text-red-300 bg-red-500/10 hover:bg-red-500/20 border border-red-500/25 disabled:opacity-40"
          >
            <Trash2 size={13} /> Clear points and mask
          </button>
        </div>

        <div className="rounded-lg border border-[#333] bg-[#181818] p-3 space-y-2">
          <div>
            <p className="text-sm text-[#ddd] font-medium">Mask morphology</p>
            <p className="text-[10px] text-[#666] mt-0.5">
              Opening removes small islands; closing fills small gaps and holes.
            </p>
          </div>
          <div className="grid grid-cols-[1fr_72px] gap-2">
            <select
              value={morphologyOperation}
              onChange={e => setMorphologyOperation(e.target.value as 'opening' | 'closing')}
              className="text-xs bg-[#e5e7eb] text-[#111827] border-[#9ca3af]"
            >
              <option value="opening">Opening</option>
              <option value="closing">Closing</option>
            </select>
            <input
              type="number"
              min={1}
              max={maxKernelSize}
              step={1}
              value={kernelSize}
              onChange={e => setKernelSize(Number(e.target.value))}
              className="text-xs bg-[#e5e7eb] text-[#111827] border-[#9ca3af]"
              title={`Filter size, 1–${maxKernelSize} pixels`}
            />
          </div>
          <button
            type="button"
            disabled={!entry?.labeled || morphologyBusy}
            onClick={() => void applyMorphology()}
            className="w-full px-2 py-2 rounded text-xs font-medium bg-sky-600 hover:bg-sky-500 text-white disabled:bg-[#292929] disabled:text-[#666]"
          >
            Apply {morphologyOperation} ({kernelSize} px)
          </button>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              disabled={morphologyBusy || historyCursor === 0}
              onClick={() => void stepMorphology('undo')}
              className="btn btn-secondary text-xs py-1.5 disabled:opacity-40"
            >
              <Undo2 size={12} className="inline mr-1" /> Undo
            </button>
            <button
              type="button"
              disabled={morphologyBusy || historyCursor >= history.length}
              onClick={() => void stepMorphology('redo')}
              className="btn btn-secondary text-xs py-1.5 disabled:opacity-40"
            >
              <Redo2 size={12} className="inline mr-1" /> Redo
            </button>
          </div>
          <div className="border-t border-[#2a2a2a] pt-2">
            <div className="flex justify-between text-[10px] text-[#777] mb-1">
              <span>Pixel edit queue</span>
              <span>{historyCursor}/{history.length} · max 50</span>
            </div>
            <div className="max-h-32 overflow-y-auto space-y-0.5">
              {history.length === 0 ? (
                <p className="text-[10px] text-[#555] py-1">No morphology edits yet.</p>
              ) : history.map((edit, i) => (
                <div
                  key={edit.id}
                  className={`grid grid-cols-[1fr_auto] gap-2 px-1.5 py-1 rounded text-[10px] ${
                    i < historyCursor ? 'text-[#aaa] bg-white/[0.03]' : 'text-[#555]'
                  }`}
                >
                  <span>{i + 1}. {edit.operation} · {edit.kernel_size}px</span>
                  <span className="font-mono">
                    <span className="text-emerald-500/90">+{edit.pixels_added}</span>
                    {' / '}
                    <span className="text-red-400/90">−{edit.pixels_removed}</span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs text-[#888]">
          <span>{labeled}/{videos.length} floor masks labeled</span>
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            className="text-sky-400 hover:text-sky-300"
          >
            View all cameras
          </button>
        </div>

        <button
          type="button"
          disabled={!entry?.registered || !currentVideoId}
          onClick={() => setPreviewOpen(true)}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-md text-xs font-medium bg-[#252525] hover:bg-[#303030] text-sky-300 border border-sky-500/25 disabled:text-[#555] disabled:border-[#333]"
        >
          <ExternalLink size={13} />
          Browse registered frames
        </button>

        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            disabled={index <= 0}
            onClick={() => enterRegistrationVideo(videos[index - 1].id)}
            className="btn btn-secondary text-xs py-2 disabled:opacity-40"
          >
            <ChevronLeft size={13} className="inline mr-1" /> Previous
          </button>
          <button
            type="button"
            disabled={index < 0 || index >= videos.length - 1}
            onClick={() => enterRegistrationVideo(videos[index + 1].id)}
            className="btn btn-secondary text-xs py-2 disabled:opacity-40"
          >
            Next <ChevronRight size={13} className="inline ml-1" />
          </button>
        </div>

        <button
          type="button"
          disabled={!allLabeled || computing}
          onClick={() => void generate()}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-md text-sm font-medium bg-emerald-600 hover:bg-emerald-500 text-white disabled:bg-[#292929] disabled:text-[#666]"
        >
          <WandSparkles size={14} />
          {registration.status === 'complete' ? 'Recompute parameters' : 'Generate registration parameters'}
        </button>
      </div>
      <RegisteredPreviewModal
        open={previewOpen}
        onClose={() => setPreviewOpen(false)}
        projectId={project.id}
        videos={registeredVideos}
        initialVideoId={video.id}
      />
    </div>
  )
}
