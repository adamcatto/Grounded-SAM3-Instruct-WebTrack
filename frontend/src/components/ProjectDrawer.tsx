import React, { useEffect, useState } from 'react'
import { X, Plus, Film, FolderOpen, Trash2 } from 'lucide-react'
import { useStore } from '../store/useStore'
import { listProjects, createProject, deleteProject } from '../api/client'
import type { Project } from '../types'

export default function ProjectDrawer() {
  const {
    drawerOpen, setDrawerOpen,
    project, setProject, setCurrentVideo,
    setUploadModalOpen,
  } = useStore()

  const [projects, setProjects] = useState<Project[]>([])
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)

  useEffect(() => {
    if (drawerOpen) loadProjects()
  }, [drawerOpen])

  async function loadProjects() {
    try {
      setProjects(await listProjects())
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

  const currentVideos = project ? Object.values(project.videos) : []

  return (
    <>
      {/* Backdrop */}
      {drawerOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50"
          onClick={() => setDrawerOpen(false)}
        />
      )}

      {/* Drawer */}
      <div
        className={`fixed top-0 left-0 h-full z-50 w-72 bg-[#111111] border-r border-[#2a2a2a] flex flex-col
          transition-transform duration-200 ease-out
          ${drawerOpen ? 'translate-x-0' : '-translate-x-full'}`}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a2a2a]">
          <span className="text-sm font-semibold text-white">Projects</span>
          <button onClick={() => setDrawerOpen(false)} className="btn btn-ghost p-1">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-4">
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
              {currentVideos.map(v => (
                <button
                  key={v.id}
                  onClick={() => handleSelectVideo(v.id)}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm hover:bg-[#1a1a1a] text-[#ccc] transition-colors"
                >
                  <Film size={14} className="flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="truncate">{v.name}</div>
                    <div className="text-[#555] text-xs">{v.num_frames} frames • {v.fps.toFixed(1)} fps</div>
                  </div>
                  {v.propagation_complete && (
                    <span className="text-green-500 text-xs">✓</span>
                  )}
                </button>
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
      </div>
    </>
  )
}
