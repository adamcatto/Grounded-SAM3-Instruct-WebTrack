import React, { useEffect, useMemo, useState } from 'react'
import { ArrowDownAZ, ArrowDownWideNarrow, ArrowUpAZ, ArrowUpWideNarrow, X, Plus, Film, FolderOpen, Trash2, ScanLine, CheckCircle2 } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useResizable } from '../hooks/useResizable'
import ProjectsFolderBrowserModal from './ProjectsFolderBrowserModal'
import { listProjects, createProject, mergeProjects, deleteProject, getProject, getProjectsRoot, removeVideo, initializeRegistration, computeRegistration } from '../api/client'
import type { ProjectsRootInfo } from '../api/client'
import type { Project } from '../types'
import VideoProgressRings from './VideoProgressRings'

export default function ProjectDrawer() {
  const {
    drawerOpen, setDrawerOpen,
    project, setProject, setCurrentVideo,
    currentVideoId,
    setUploadModalOpen,
    addToast,
    clearHistory,
    registrationMode,
    enterRegistrationVideo,
  } = useStore()

  const [projects, setProjects] = useState<Project[]>([])
  const [projectsRoot, setProjectsRootState] = useState<ProjectsRootInfo | null>(null)
  const [rootInfoError, setRootInfoError] = useState<string>('')
  const [folderBrowserOpen, setFolderBrowserOpen] = useState(false)
  const [newName, setNewName] = useState('')
  const [newTrackingMode, setNewTrackingMode] = useState<'segmentation_tracking' | 'pose_tracking'>('segmentation_tracking')
  const [creating, setCreating] = useState(false)
  const [mergeMode, setMergeMode] = useState(false)
  const [mergeLeftId, setMergeLeftId] = useState('')
  const [mergeRightId, setMergeRightId] = useState('')
  const [drawerWidthPx, onDrawerResizeMouseDown] = useResizable(320, {
    min: 220,
    max: 560,
    direction: 'horizontal',
  })
  const [registrationBusy, setRegistrationBusy] = useState(false)
  const [projectSort, setProjectSort] = useState<'name' | 'created'>('name')
  const [projectSortAscending, setProjectSortAscending] = useState(true)

  useEffect(() => {
    if (!drawerOpen) return
    loadProjects()
    setRootInfoError('')
    getProjectsRoot()
      .then(info => {
        setProjectsRootState(info)
        setRootInfoError('')
      })
      .catch((e: unknown) => {
        setProjectsRootState(null)
        const ax = e as { response?: { status?: number; data?: { detail?: string } } }
        const status = ax?.response?.status
        const detail = ax?.response?.data?.detail
        if (status === 404) {
          setRootInfoError('Backend is out of date — restart it to use the projects folder browser.')
        } else if (typeof detail === 'string') {
          setRootInfoError(detail)
        } else {
          setRootInfoError('Could not reach backend')
        }
      })
  }, [drawerOpen])

  useEffect(() => {
    if (!drawerOpen || !project?.id) return
    let cancelled = false
    getProject(project.id)
      .then(p => {
        if (!cancelled) setProject(p)
      })
      .catch(() => {
        /* keep cached project */
      })
    return () => {
      cancelled = true
    }
  }, [drawerOpen, project?.id, setProject])

  async function loadProjects() {
    try {
      const plist = await listProjects()
      setProjects(plist)
      const pid = useStore.getState().project?.id
      if (pid && !plist.some(p => p.id === pid)) {
        setProject(null)
        setCurrentVideo(null)
      }
    } catch { /* ignore */ }
  }

  function sortableProjectName(value: string): string {
    return value.trim().replace(/^[0-9a-f]{8}(?:[-_\s]+|$)/i, '').trim()
  }

  const sortedProjects = useMemo(() => {
    const direction = projectSortAscending ? 1 : -1
    return [...projects].sort((a, b) => {
      let comparison: number
      if (projectSort === 'created') {
        const aTime = Number.isFinite(Date.parse(a.created_at)) ? Date.parse(a.created_at) : 0
        const bTime = Number.isFinite(Date.parse(b.created_at)) ? Date.parse(b.created_at) : 0
        comparison = aTime - bTime
      } else {
        comparison = sortableProjectName(a.name).localeCompare(
          sortableProjectName(b.name),
          undefined,
          { sensitivity: 'base', numeric: true },
        )
      }
      return comparison === 0 ? a.id.localeCompare(b.id) : comparison * direction
    })
  }, [projectSort, projectSortAscending, projects])

  const mergeableProjects = useMemo(
    () => [...projects].sort((a, b) =>
      sortableProjectName(a.name).localeCompare(
        sortableProjectName(b.name),
        undefined,
        { sensitivity: 'base', numeric: true },
      ),
    ),
    [projects],
  )

  const canCreate = mergeMode
    ? Boolean(newName.trim() && mergeLeftId && mergeRightId && mergeLeftId !== mergeRightId)
    : Boolean(newName.trim())

  async function handleCreate() {
    if (!canCreate) return
    setCreating(true)
    try {
      const p = mergeMode
        ? await mergeProjects(newName.trim(), mergeLeftId, mergeRightId)
        : await createProject(newName.trim(), newTrackingMode)
      setProjects(prev => [...prev, p])
      setProject(p)
      const vids = Object.keys(p.videos)
      setCurrentVideo(vids.length > 0 ? vids[vids.length - 1] : null)
      setNewName('')
      setMergeLeftId('')
      setMergeRightId('')
      addToast(
        mergeMode
          ? `Merged projects into "${p.name}" (${vids.length} videos)`
          : `Created project "${p.name}"`,
        'success',
      )
    } catch {
      addToast(mergeMode ? 'Failed to merge projects' : 'Failed to create project', 'error')
    } finally {
      setCreating(false)
    }
  }

  async function handleDeleteProject(pid: string, e: React.MouseEvent) {
    e.stopPropagation()
    if (!confirm('Delete this project and all its data?')) return
    await deleteProject(pid)
    setProjects(prev => prev.filter(p => p.id !== pid))
    if (project?.id === pid) {
      setProject(null)
      setCurrentVideo(null)
    }
  }

  function handleSelectProject(p: Project) {
    setProject(p)
    const vids = Object.keys(p.videos)
    setCurrentVideo(vids.length > 0 ? vids[vids.length - 1] : null)
    setDrawerOpen(false)
  }

  function handleSelectVideo(vid: string) {
    setCurrentVideo(vid)
    setDrawerOpen(false)
  }

  async function handleStartRegistration() {
    if (!project || Object.keys(project.videos).length === 0) return
    setRegistrationBusy(true)
    try {
      await initializeRegistration(project.id)
      const fresh = await getProject(project.id)
      setProject(fresh)
      const firstUnlabeled = Object.keys(fresh.videos).find(
        vid => !fresh.registration?.videos?.[vid]?.labeled,
      ) ?? Object.keys(fresh.videos)[0]
      enterRegistrationVideo(firstUnlabeled)
      addToast('Registration created. Segment the visible floor in each first frame.', 'success')
      setDrawerOpen(false)
    } catch {
      addToast('Failed to initialize video registration', 'error')
    } finally {
      setRegistrationBusy(false)
    }
  }

  async function handleComputeRegistration() {
    if (!project) return
    setRegistrationBusy(true)
    try {
      await computeRegistration(
        project.id,
        project.registration?.warp_mode === 'affine_full_frame'
          ? 'affine_full_frame'
          : 'bounded_full_frame_mesh',
      )
      const fresh = await getProject(project.id)
      setProject(fresh)
      addToast('Registration homographies generated', 'success')
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      addToast(detail || 'Failed to generate registration', 'error')
    } finally {
      setRegistrationBusy(false)
    }
  }

  async function handleDeleteVideo(vid: string, e: React.MouseEvent) {
    e.stopPropagation()
    if (!project) return
    const meta = project.videos[vid]
    const label = meta?.name ?? 'this video'
    if (
      !confirm(
        `Delete "${label}" from this project?\n\n` +
          'The video file and all annotations, masks, and tracking data on disk will be removed. This cannot be undone.',
      )
    ) {
      return
    }
    try {
      await removeVideo(project.id, vid)
      const nextVideos = { ...project.videos }
      delete nextVideos[vid]
      const nextProject: Project = { ...project, videos: nextVideos }
      setProjects(prev => prev.map(p => (p.id === project.id ? nextProject : p)))
      setProject(nextProject)
      clearHistory()
      if (currentVideoId === vid) {
        const sorted = Object.values(nextVideos).sort((a, b) =>
          a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }),
        )
        setCurrentVideo(sorted.length > 0 ? sorted[0].id : null)
      }
      addToast(`Removed "${label}"`, 'success')
    } catch {
      addToast('Failed to delete video', 'error')
    }
  }

  const currentVideos = useMemo(() => {
    if (!project) return []
    return Object.values(project.videos).sort((a, b) =>
      a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }),
    )
  }, [project])

  return (
    <>
      {/* Backdrop */}
      {drawerOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50"
          onClick={() => setDrawerOpen(false)}
        />
      )}

      {/* Drawer — width is user-resizable (drag right edge) */}
      <div
        className={`fixed top-0 left-0 h-full z-50 bg-[#111111] border-r border-[#2a2a2a] flex flex-col shadow-xl
          transition-transform duration-200 ease-out
          ${drawerOpen ? 'translate-x-0' : '-translate-x-full'}`}
        style={{ width: drawerWidthPx }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a2a2a] shrink-0">
          <span className="text-sm font-semibold text-white">Projects</span>
          <button onClick={() => setDrawerOpen(false)} className="btn btn-ghost p-1">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-4 min-h-0">
          <div className="rounded-lg border border-[#2a2a2a] bg-[#161616] p-3 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <label className="text-xs text-[#666] uppercase tracking-wider">Projects folder</label>
              <button
                type="button"
                disabled={!projectsRoot}
                onClick={() => setFolderBrowserOpen(true)}
                className="text-xs text-blue-400 hover:text-blue-300 font-medium disabled:text-[#444] disabled:cursor-not-allowed"
                title={projectsRoot ? 'Browse for a projects folder' : 'Waiting for backend…'}
              >
                Browse…
              </button>
            </div>
            <p
              className="text-[10px] text-[#a3a3a3] font-mono break-all leading-relaxed"
              title={projectsRoot?.active_root}
            >
              {projectsRoot?.active_root ?? (rootInfoError ? '—' : 'Loading…')}
            </p>
            {rootInfoError && (
              <p className="text-[10px] text-red-400 leading-snug">{rootInfoError}</p>
            )}
            {projectsRoot?.env_var ? (
              <p className="text-[10px] text-[#555] leading-snug">
                Env default via{' '}
                <span className="text-[#777] font-mono">{projectsRoot.env_var}</span>
                {projectsRoot.env_default_root !== projectsRoot.active_root ? (
                  <>
                    {' '}
                    <span className="text-[#444]">·</span>{' '}
                    <span className="text-[#666] font-mono truncate block mt-0.5" title={projectsRoot.env_default_root}>
                      {projectsRoot.env_default_root}
                    </span>
                  </>
                ) : null}
              </p>
            ) : null}
          </div>

          {/* New project */}
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <label className="text-xs text-[#666] uppercase tracking-wider">New Project</label>
              <label className="flex items-center gap-1.5 text-[10px] text-[#888] cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={mergeMode}
                  onChange={e => setMergeMode(e.target.checked)}
                  className="rounded border-[#444]"
                />
                Merge two projects
              </label>
            </div>
            {mergeMode && (
              <p className="text-[10px] text-[#666] leading-snug">
                Copies all videos and annotations into a new project. Original projects are kept unchanged.
              </p>
            )}
            {mergeMode && (
              <div className="space-y-2">
                <select
                  value={mergeLeftId}
                  onChange={e => setMergeLeftId(e.target.value)}
                  className="w-full text-sm"
                  disabled={mergeableProjects.length === 0}
                >
                  <option value="">First project…</option>
                  {mergeableProjects.map(p => (
                    <option key={p.id} value={p.id}>
                      {p.name} ({Object.keys(p.videos).length} videos)
                    </option>
                  ))}
                </select>
                <select
                  value={mergeRightId}
                  onChange={e => setMergeRightId(e.target.value)}
                  className="w-full text-sm"
                  disabled={mergeableProjects.length === 0}
                >
                  <option value="">Second project…</option>
                  {mergeableProjects.map(p => (
                    <option key={p.id} value={p.id}>
                      {p.name} ({Object.keys(p.videos).length} videos)
                    </option>
                  ))}
                </select>
                {mergeLeftId && mergeRightId && mergeLeftId === mergeRightId && (
                  <p className="text-[10px] text-red-400">Choose two different projects.</p>
                )}
                {mergeableProjects.length < 2 && (
                  <p className="text-[10px] text-amber-500/90">Need at least two projects to merge.</p>
                )}
              </div>
            )}
            <div className="flex gap-2">
              <input
                type="text"
                value={newName}
                onChange={e => setNewName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleCreate()}
                placeholder={mergeMode ? 'Merged project name…' : 'Project name…'}
                className="flex-1 text-sm"
              />
              <button
                onClick={handleCreate}
                disabled={creating || !canCreate}
                className="btn btn-primary px-2"
                title={mergeMode ? 'Merge projects' : 'Create project'}
              >
                <Plus size={14} />
              </button>
            </div>
            {!mergeMode && (
              <select
                value={newTrackingMode}
                onChange={event => setNewTrackingMode(event.target.value as typeof newTrackingMode)}
                className="w-full bg-[#e5e7eb] text-xs text-[#111827]"
              >
                <option value="segmentation_tracking">Segmentation tracking (SAM3)</option>
                <option value="pose_tracking">Pose tracking (CoTracker3)</option>
              </select>
            )}
          </div>

          {/* Projects list */}
          {projects.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center gap-1.5">
                <label className="mr-auto text-xs text-[#666] uppercase tracking-wider">Projects</label>
                <select
                  value={projectSort}
                  onChange={event => {
                    const next = event.target.value as typeof projectSort
                    setProjectSort(next)
                    setProjectSortAscending(next === 'name')
                  }}
                  className="max-w-28 rounded border border-[#444] bg-[#e5e7eb] px-1.5 py-1 text-[10px] text-[#111827]"
                  aria-label="Sort projects by"
                >
                  <option value="name">Name</option>
                  <option value="created">Creation date</option>
                </select>
                <button
                  type="button"
                  onClick={() => setProjectSortAscending(value => !value)}
                  className="btn btn-ghost p-1 text-[#888]"
                  title={projectSortAscending ? 'Ascending; click for descending' : 'Descending; click for ascending'}
                  aria-label={projectSortAscending ? 'Sort ascending' : 'Sort descending'}
                >
                  {projectSort === 'name'
                    ? projectSortAscending ? <ArrowDownAZ size={14} /> : <ArrowUpAZ size={14} />
                    : projectSortAscending ? <ArrowDownWideNarrow size={14} /> : <ArrowUpWideNarrow size={14} />}
                </button>
              </div>
              {sortedProjects.map(p => (
                <button
                  key={p.id}
                  onClick={() => handleSelectProject(p)}
                  className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm transition-colors
                    ${project?.id === p.id ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30' : 'hover:bg-[#1a1a1a] text-[#ccc]'}`}
                >
                  <FolderOpen size={14} className="flex-shrink-0" />
                  <span className="flex-1 truncate">{p.name}</span>
                  <span className="text-[#555] text-xs">{Object.keys(p.videos).length}</span>
                  <button
                    onClick={e => handleDeleteProject(p.id, e)}
                    className="text-[#555] hover:text-red-400 p-0.5 rounded"
                  >
                    <Trash2 size={12} />
                  </button>
                </button>
              ))}
            </div>
          )}

          {/* Current project videos */}
          {project && (
            <div className="space-y-1">
              <label className="text-xs text-[#666] uppercase tracking-wider">
                Videos — {project.name}
              </label>
              {currentVideos.length > 0 && (
                <div
                  className="flex flex-wrap items-center gap-x-3 gap-y-1 px-1 pt-0.5 pb-1 text-[10px] text-[#777] leading-tight border-b border-[#1e1e1e] mb-1"
                  aria-label="Legend for progress rings"
                >
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="h-2 w-2 shrink-0 rounded-full ring-1 ring-orange-500/40"
                      style={{ backgroundColor: '#f97316' }}
                      aria-hidden
                    />
                    <span>Orange — anchor frames labeled</span>
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="h-2 w-2 shrink-0 rounded-full ring-1 ring-green-500/40"
                      style={{ backgroundColor: '#22c55e' }}
                      aria-hidden
                    />
                    <span>Green — whole-video frames tracked</span>
                  </span>
                </div>
              )}
              {currentVideos.length > 0 && !project.registration && (
                <button
                  type="button"
                  disabled={registrationBusy}
                  onClick={() => void handleStartRegistration()}
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm text-sky-300 bg-sky-500/10 hover:bg-sky-500/20 border border-sky-500/30 disabled:opacity-50"
                >
                  <ScanLine size={14} />
                  <span>Register videos</span>
                </button>
              )}
              {project.registration && (
                <div className={`rounded-lg border ${
                  registrationMode ? 'border-sky-500/50 bg-sky-500/10' : 'border-[#333] bg-[#171717]'
                }`}>
                  <div className="flex items-center gap-2 px-3 py-2 text-sm text-sky-200">
                    <ScanLine size={14} />
                    <span className="font-medium flex-1">Registration</span>
                    <span className="text-[10px] text-[#888]">
                      {Object.values(project.registration.videos).filter(v => v.labeled || v.registered).length}/
                      {currentVideos.length} labeled
                    </span>
                  </div>
                  <div className="border-t border-[#2a2a2a] px-1 py-1 space-y-0.5">
                    {currentVideos.map(v => {
                      const reg = project.registration!.videos[v.id]
                      return (
                        <button
                          key={`registration-${v.id}`}
                          type="button"
                          onClick={() => {
                            enterRegistrationVideo(v.id)
                            setDrawerOpen(false)
                          }}
                          className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left text-xs ${
                            registrationMode && currentVideoId === v.id
                              ? 'bg-sky-500/20 text-sky-100'
                              : 'text-[#aaa] hover:bg-white/5'
                          }`}
                        >
                          {reg?.labeled || reg?.registered
                            ? <CheckCircle2 size={12} className="text-emerald-400 shrink-0" />
                            : <span className="w-3 h-3 rounded-full border border-[#555] shrink-0" />}
                          <span className="truncate flex-1">{v.name} · frame 0</span>
                        </button>
                      )
                    })}
                  </div>
                  <div className="border-t border-[#2a2a2a] p-2">
                    <button
                      type="button"
                      disabled={
                        registrationBusy ||
                        !currentVideos.every(v => {
                          const entry = project.registration?.videos[v.id]
                          return entry?.labeled || entry?.registered
                        })
                      }
                      onClick={() => void handleComputeRegistration()}
                      className="w-full px-2 py-1.5 rounded text-xs font-medium bg-emerald-600/80 hover:bg-emerald-600 text-white disabled:bg-[#292929] disabled:text-[#666]"
                    >
                      {project.registration.status === 'complete'
                        ? 'Recompute registration parameters'
                        : 'Generate registration parameters'}
                    </button>
                  </div>
                </div>
              )}
              {currentVideos.map(v => (
                <div
                  key={v.id}
                  className={`w-full flex items-center gap-1 pl-3 pr-1 py-2 rounded-lg text-sm transition-colors border ${
                    currentVideoId === v.id
                      ? 'bg-blue-600/18 text-[#dce9ff] border-blue-500/35'
                      : 'border-transparent text-[#ccc] hover:bg-[#1a1a1a]'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => handleSelectVideo(v.id)}
                    className="flex-1 flex items-center gap-2 min-w-0 text-left rounded-md -my-1 py-1 pr-1 hover:bg-white/5"
                  >
                    <Film size={14} className="flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <div className="truncate">{v.name}</div>
                      <div className="text-[#555] text-xs">{v.num_frames} frames • {v.fps.toFixed(1)} fps</div>
                    </div>
                    <VideoProgressRings video={v} />
                  </button>
                  <button
                    type="button"
                    onClick={e => void handleDeleteVideo(v.id, e)}
                    className="text-[#555] hover:text-red-400 p-1.5 rounded shrink-0"
                    title="Delete video from project"
                    aria-label={`Delete video ${v.name}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}

              <button
                onClick={() => { setUploadModalOpen(true); setDrawerOpen(false) }}
                className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-[#666] hover:text-[#ccc] hover:bg-[#1a1a1a] transition-colors border border-dashed border-[#333]"
              >
                <Plus size={14} />
                <span>Add video</span>
              </button>
            </div>
          )}
        </div>

        {/* Drag handle — resize panel width */}
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label={`Resize projects panel, ${drawerWidthPx} pixels wide. Drag sideways.`}
          title="Drag sideways to widen or narrow the menu"
          onMouseDown={onDrawerResizeMouseDown}
          className="absolute top-0 right-0 z-[60] h-full w-2 cursor-col-resize select-none hover:bg-[#3b82f641] active:bg-[#3b82f666]"
        />

        <ProjectsFolderBrowserModal
          open={folderBrowserOpen}
          rootInfo={projectsRoot}
          onClose={() => setFolderBrowserOpen(false)}
          onApplied={() => {
            void getProjectsRoot().then(setProjectsRootState)
            void loadProjects()
          }}
          addToast={addToast}
        />
      </div>
    </>
  )
}
