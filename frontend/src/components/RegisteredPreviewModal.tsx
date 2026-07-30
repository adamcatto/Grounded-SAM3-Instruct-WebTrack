import React, { useEffect, useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight, Grid2X2, Images, X } from 'lucide-react'
import { registrationFramePreviewUrl } from '../api/client'
import type { VideoMeta } from '../types'

interface Props {
  open: boolean
  onClose: () => void
  projectId: string
  videos: VideoMeta[]
  initialVideoId: string
}

export default function RegisteredPreviewModal({
  open,
  onClose,
  projectId,
  videos,
  initialVideoId,
}: Props) {
  const [mode, setMode] = useState<'comparison' | 'grid'>('comparison')
  const [selectedIds, setSelectedIds] = useState<string[]>([initialVideoId])
  const [draftFrame, setDraftFrame] = useState(0)
  const [renderedFrame, setRenderedFrame] = useState(0)
  const [showOriginal, setShowOriginal] = useState(true)
  const [loading, setLoading] = useState<Record<string, boolean>>({})
  const [errors, setErrors] = useState<Record<string, boolean>>({})
  const [revision, setRevision] = useState('')

  const videoById = useMemo(
    () => Object.fromEntries(videos.map(video => [video.id, video])),
    [videos],
  )
  const selectedVideos = useMemo(
    () => selectedIds.map(id => videoById[id]).filter(Boolean),
    [selectedIds, videoById],
  )
  const primaryVideo = videoById[initialVideoId] ?? videos[0]
  const commonLastFrame = useMemo(() => {
    const relevant = mode === 'grid' ? selectedVideos : primaryVideo ? [primaryVideo] : []
    if (relevant.length === 0) return 0
    return Math.max(0, Math.min(...relevant.map(video => video.num_frames - 1)))
  }, [mode, primaryVideo, selectedVideos])

  useEffect(() => {
    if (!open) return
    setMode('comparison')
    setSelectedIds([initialVideoId])
    setDraftFrame(0)
    setRenderedFrame(0)
    setRevision(String(Date.now()))
    setErrors({})
  }, [initialVideoId, open])

  useEffect(() => {
    setDraftFrame(frame => Math.min(frame, commonLastFrame))
  }, [commonLastFrame])

  useEffect(() => {
    if (!open) return
    const timer = window.setTimeout(() => {
      const frame = Math.max(0, Math.min(commonLastFrame, Math.round(draftFrame)))
      setRenderedFrame(frame)
      const ids = mode === 'grid' ? selectedIds : primaryVideo ? [primaryVideo.id] : []
      setLoading(Object.fromEntries(ids.map(id => [id, true])))
      setErrors({})
    }, 140)
    return () => window.clearTimeout(timer)
  }, [commonLastFrame, draftFrame, mode, open, primaryVideo, selectedIds])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        setDraftFrame(frame => Math.max(0, frame - 1))
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        setDraftFrame(frame => Math.min(commonLastFrame, frame + 1))
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [commonLastFrame, onClose, open])

  if (!open || !primaryVideo) return null

  function toggleVideo(id: string) {
    setSelectedIds(current =>
      current.includes(id) ? current.filter(item => item !== id) : [...current, id],
    )
  }

  function registeredUrl(videoId: string) {
    return registrationFramePreviewUrl(
      projectId,
      videoId,
      renderedFrame,
      'registered',
      revision,
    )
  }

  const originalUrl = registrationFramePreviewUrl(
    projectId,
    primaryVideo.id,
    renderedFrame,
    'original',
    revision,
  )

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 flex flex-col p-4" role="dialog" aria-modal>
      <div className="mx-auto w-full max-w-[1600px] flex-1 min-h-0 rounded-xl border border-[#333] bg-[#111] shadow-2xl flex flex-col overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[#292929]">
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold text-white">Registered-frame viewer</h2>
            <p className="text-xs text-[#777]">
              Shared frame {renderedFrame} · {mode === 'grid' ? `${selectedVideos.length} selected videos` : primaryVideo.name}
            </p>
          </div>
          <div className="flex rounded-md border border-[#333] overflow-hidden">
            <button
              type="button"
              onClick={() => setMode('comparison')}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs ${
                mode === 'comparison' ? 'bg-sky-600 text-white' : 'bg-[#1d1d1d] text-[#888]'
              }`}
            >
              <Images size={13} /> Original comparison
            </button>
            <button
              type="button"
              onClick={() => setMode('grid')}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs ${
                mode === 'grid' ? 'bg-sky-600 text-white' : 'bg-[#1d1d1d] text-[#888]'
              }`}
            >
              <Grid2X2 size={13} /> Multi-video grid
            </button>
          </div>
          {mode === 'comparison' && (
            <label className="flex items-center gap-2 text-xs text-[#aaa]">
              <input
                type="checkbox"
                checked={showOriginal}
                onChange={event => setShowOriginal(event.target.checked)}
              />
              Show original
            </label>
          )}
          <button type="button" onClick={onClose} className="btn btn-ghost p-1.5" aria-label="Close">
            <X size={17} />
          </button>
        </div>

        {mode === 'grid' && (
          <div className="px-4 py-2 border-b border-[#292929] bg-[#151515]">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] uppercase tracking-wide text-[#666]">Registered videos</span>
              <div className="flex gap-2 text-[10px]">
                <button type="button" onClick={() => setSelectedIds(videos.map(video => video.id))} className="text-sky-400">
                  Select all
                </button>
                <button type="button" onClick={() => setSelectedIds([])} className="text-[#777]">
                  Clear
                </button>
              </div>
            </div>
            <div className="flex gap-1.5 overflow-x-auto pb-1">
              {videos.map(video => (
                <label
                  key={video.id}
                  className={`shrink-0 flex items-center gap-1.5 px-2 py-1 rounded border text-[10px] cursor-pointer ${
                    selectedIds.includes(video.id)
                      ? 'border-sky-500/50 bg-sky-500/15 text-sky-100'
                      : 'border-[#333] bg-[#1b1b1b] text-[#777]'
                  }`}
                  title={video.name}
                >
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(video.id)}
                    onChange={() => toggleVideo(video.id)}
                  />
                  <span className="max-w-40 truncate">{video.name}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        {mode === 'comparison' ? (
          <div className={`flex-1 min-h-0 grid gap-3 p-3 ${showOriginal ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
            {showOriginal && (
              <figure className="min-h-0 flex flex-col">
                <figcaption className="text-xs text-[#888] mb-1">Original · frame {renderedFrame}</figcaption>
                <div className="flex-1 min-h-0 bg-black rounded-lg border border-[#292929] flex items-center justify-center overflow-hidden">
                  <img src={originalUrl} alt={`Original frame ${renderedFrame}`} className="max-w-full max-h-full object-contain" />
                </div>
              </figure>
            )}
            <PreviewTile
              video={primaryVideo}
              frame={renderedFrame}
              src={registeredUrl(primaryVideo.id)}
              loading={Boolean(loading[primaryVideo.id])}
              error={Boolean(errors[primaryVideo.id])}
              onLoad={() => setLoading(state => ({ ...state, [primaryVideo.id]: false }))}
              onError={() => {
                setLoading(state => ({ ...state, [primaryVideo.id]: false }))
                setErrors(state => ({ ...state, [primaryVideo.id]: true }))
              }}
              large
            />
          </div>
        ) : (
          <div className="flex-1 min-h-0 overflow-y-auto p-3">
            {selectedVideos.length === 0 ? (
              <div className="h-full flex items-center justify-center text-sm text-[#666]">
                Select one or more registered videos above.
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3 auto-rows-[minmax(260px,1fr)]">
                {selectedVideos.map(video => (
                  <PreviewTile
                    key={video.id}
                    video={video}
                    frame={renderedFrame}
                    src={registeredUrl(video.id)}
                    loading={Boolean(loading[video.id])}
                    error={Boolean(errors[video.id])}
                    onLoad={() => setLoading(state => ({ ...state, [video.id]: false }))}
                    onError={() => {
                      setLoading(state => ({ ...state, [video.id]: false }))
                      setErrors(state => ({ ...state, [video.id]: true }))
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        <div className="px-4 py-3 border-t border-[#292929] space-y-2">
          <input
            type="range"
            min={0}
            max={commonLastFrame}
            step={1}
            value={Math.min(draftFrame, commonLastFrame)}
            onChange={event => setDraftFrame(Number(event.target.value))}
            className="w-full"
            aria-label="Shared preview frame"
          />
          <div className="flex items-center justify-center gap-2">
            <button
              type="button"
              disabled={draftFrame <= 0}
              onClick={() => setDraftFrame(frame => Math.max(0, frame - 1))}
              className="btn btn-secondary px-3 py-1.5 disabled:opacity-40"
            >
              <ChevronLeft size={14} />
            </button>
            <label className="flex items-center gap-2 text-xs text-[#888]">
              Shared frame
              <input
                type="number"
                min={0}
                max={commonLastFrame}
                value={draftFrame}
                onChange={event => setDraftFrame(Math.max(0, Math.min(commonLastFrame, Number(event.target.value))))}
                className="w-24 bg-[#e5e7eb] text-[#111827]"
              />
              <span>/ {commonLastFrame}</span>
            </label>
            <button
              type="button"
              disabled={draftFrame >= commonLastFrame}
              onClick={() => setDraftFrame(frame => Math.min(commonLastFrame, frame + 1))}
              className="btn btn-secondary px-3 py-1.5 disabled:opacity-40"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function PreviewTile({
  video,
  frame,
  src,
  loading,
  error,
  onLoad,
  onError,
  large = false,
}: {
  video: VideoMeta
  frame: number
  src: string
  loading: boolean
  error: boolean
  onLoad: () => void
  onError: () => void
  large?: boolean
}) {
  return (
    <figure className="min-h-0 flex flex-col">
      <figcaption className="text-xs text-sky-300 mb-1 truncate" title={video.name}>
        {video.name} · frame {frame}
      </figcaption>
      <div className={`relative flex-1 min-h-0 bg-black rounded-lg border border-sky-500/25 flex items-center justify-center overflow-hidden ${large ? '' : 'aspect-square'}`}>
        {loading && <span className="absolute text-xs text-[#666]">Rendering…</span>}
        {error && <span className="absolute text-xs text-red-400 px-4 text-center">Could not render frame</span>}
        <img
          src={src}
          alt={`Registered ${video.name} frame ${frame}`}
          className="max-w-full max-h-full object-contain"
          onLoad={onLoad}
          onError={onError}
        />
      </div>
    </figure>
  )
}
