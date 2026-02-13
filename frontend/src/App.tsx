import React, { useEffect, useState, useCallback } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from './store/useStore'
import { listProjects } from './api/client'
import Header from './components/Header'
import ProjectDrawer from './components/ProjectDrawer'
import UploadModal from './components/UploadModal'
import LeftPanel from './components/LeftPanel/LeftPanel'
import FrameViewer from './components/FrameViewer/FrameViewer'
import Timeline from './components/Timeline/Timeline'
import LoadingScreen from './components/LoadingScreen'

export default function App() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project, currentVideoId,
    setProject, setCurrentVideo, setUploadModalOpen, setDrawerOpen,
  } = store

  const [backendReady, setBackendReady] = useState(false)

  const handleBackendReady = useCallback(() => {
    setBackendReady(true)
  }, [])

  // ── Load initial project on mount (after backend is ready) ────────────────

  useEffect(() => {
    if (!backendReady) return
    listProjects().then(projects => {
      if (projects.length > 0 && !project) {
        const last = projects[projects.length - 1]
        setProject(last)
        const vids = Object.keys(last.videos)
        if (vids.length > 0) {
          setCurrentVideo(vids[vids.length - 1])
        }
      }
    }).catch(() => {
      // ignore
    })
  }, [backendReady])

  // ── Show loading screen until SAM model is loaded ─────────────────────────

  if (!backendReady) {
    return <LoadingScreen onReady={handleBackendReady} />
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#111111]">
      {/* Top header */}
      <Header />

      {/* Main content area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel (fixed width) */}
        <div className="w-64 flex-shrink-0 border-r border-[#1e1e1e] flex flex-col overflow-hidden bg-[#111111]">
          <LeftPanel />
        </div>

        {/* Right: frame viewer + timeline stacked */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Frame viewer (grows to fill) */}
          <FrameViewer />

          {/* Timeline (fixed height at bottom) */}
          {video && (
            <div className="flex-shrink-0">
              <Timeline />
            </div>
          )}
        </div>
      </div>

      {/* Overlays */}
      <ProjectDrawer />
      <UploadModal />

      {/* No video selected — welcome prompt (inline, does NOT block header) */}
      {!video && (
        <div className="fixed inset-0 top-12 flex items-center justify-center z-10 pointer-events-none">
          <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-2xl p-8 max-w-sm w-full text-center shadow-2xl pointer-events-auto">
            <h2 className="text-xl font-bold text-white mb-2">
              {project ? `Project: ${project.name}` : 'Welcome to SAM3 Web Tracker'}
            </h2>
            <p className="text-sm text-[#888] mb-6 leading-relaxed">
              {project
                ? 'This project has no videos yet. Import a video to get started, or open the menu to switch projects.'
                : 'Import a video to start labeling and tracking objects, or open the menu to browse existing projects.'}
            </p>
            <div className="flex flex-col gap-2">
              <button
                onClick={() => setUploadModalOpen(true)}
                className="btn btn-primary w-full py-2.5 text-sm font-medium"
              >
                {project ? 'Add a video' : 'New project'}
              </button>
              <button
                onClick={() => setDrawerOpen(true)}
                className="btn btn-secondary w-full py-2.5 text-sm font-medium"
              >
                Browse projects
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
