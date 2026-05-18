import React, { useEffect, useMemo, useState } from 'react'
import { X, Plus, Film, FolderOpen, Trash2 } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useResizable } from '../hooks/useResizable'
import ProjectsFolderBrowserModal from './ProjectsFolderBrowserModal'
import { listProjects, createProject, deleteProject, getProject, getProjectsRoot, removeVideo } from '../api/client'
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
  } = useStore()

  const [projects, setProjects] = useState<Project[]>([])
  const [projectsRoot, setProjectsRootState] = useState<ProjectsRootInfo | null>(null)
  const [rootInfoError, setRootInfoError] = useState<string>('')
  const [folderBrowserOpen, setFolderBrowserOpen] = useState(false)
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [drawerWidthPx, onDrawerResizeMouseDown] = useResizable(320, {
    min: 220,
    max: 560,
    direction: 'horizontal',
  })

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

  async function handleCreate() {
    if (!newName.trim()) return
    setCreating(true)
    try {
      const p = await createProject(newName.trim())
      setProjects(prev => [...prev, p])
      setProject(p)
      setCurrentVideo(null)
      setNewName('')
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
            <label className="text-xs text-[#666] uppercase tracking-wider">New Project</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={newName}
                onChange={e => setNewName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleCreate()}
                placeholder="Project name..."
                className="flex-1 text-sm"
              />
              <button
                onClick={handleCreate}
                disabled={creating || !newName.trim()}
                className="btn btn-primary px-2"
              >
                <Plus size={14} />
              </button>
            </div>
          </div>

          {/* Projects list */}
          {projects.length > 0 && (
            <div className="space-y-1">
              <label className="text-xs text-[#666] uppercase tracking-wider">Projects</label>
              {projects.map(p => (
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
