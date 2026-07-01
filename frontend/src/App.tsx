import React, { useEffect, useState, useCallback } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from './store/useStore'
import type { ViewerTab } from './store/useStore'
import { listProjects } from './api/client'
import { Crosshair, PlayCircle, BrainCircuit, Settings2 } from 'lucide-react'
import Header from './components/Header'
import ProjectDrawer from './components/ProjectDrawer'
import UploadModal from './components/UploadModal'
import LeftPanel from './components/LeftPanel/LeftPanel'
import FrameViewer from './components/FrameViewer/FrameViewer'
import VideoPlayer from './components/VideoPlayer/VideoPlayer'
import InferenceStatePanel from './components/InferenceStatePanel'
import ConfigPanel from './components/ConfigPanel'
import Timeline from './components/Timeline/Timeline'
import LoadingScreen from './components/LoadingScreen'
import ResizeHandle from './components/ResizeHandle'
import ToastContainer from './components/ToastContainer'
import { useResizable } from './hooks/useResizable'
import { useMaskLoader } from './hooks/useMaskLoader'

function TabButton({ active, icon, label, onClick, badge }: {
  active: boolean
  icon: React.ReactNode
  label: string
  onClick: () => void
  badge?: number
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-colors ${
        active
          ? 'bg-[#2a2a2a] text-white'
          : 'text-[#888] hover:text-white hover:bg-[#1a1a1a]'
      }`}
    >
      {icon}
      {label}
      {badge != null && badge > 0 && (
        <span className="ml-0.5 bg-amber-500 text-black text-[9px] font-bold rounded-full w-4 h-4 flex items-center justify-center">
          {badge}
        </span>
      )}
    </button>
  )
}

export default function App() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project, currentVideoId, viewerTab,
    setProject, setCurrentVideo, setUploadModalOpen, setDrawerOpen, setViewerTab,
    configDirty, revertConfig,
  } = store

  // Guard tab switches away from Settings when there are unsaved changes.
  const handleSetViewerTab = useCallback((tab: ViewerTab) => {
    if (viewerTab === 'config' && configDirty && tab !== 'config') {
      if (!window.confirm('You have unsaved changes in Settings. Discard them?')) return
      revertConfig()
    }
    setViewerTab(tab)
  }, [viewerTab, configDirty, revertConfig, setViewerTab])

  const [backendReady, setBackendReady] = useState(false)

  // ── Resizable panels ──────────────────────────────────────────────────────
  const [leftWidth, leftHandleMouseDown] = useResizable(256, { min: 180, max: 450 })
  const [timelineHeight, timelineHandleMouseDown] = useResizable(130, {
    min: 70,
    max: 280,
    direction: 'vertical',
    reverse: true,  // drag up → timeline grows
  })

  const handleBackendReady = useCallback(() => {
    setBackendReady(true)
  }, [])

  // Central mask fetch + prefetch while annotating.
  useMaskLoader()

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
        {/* Left panel (resizable width) */}
        <div
          className="flex-shrink-0 flex flex-col overflow-hidden bg-[#111111]"
          style={{ width: leftWidth }}
        >
          <LeftPanel />
        </div>

        {/* Vertical resize handle between left panel and center */}
        <ResizeHandle direction="horizontal" onMouseDown={leftHandleMouseDown} />

        {/* Right: tab bar + viewer + timeline stacked */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Tab bar — always visible so Config is accessible without a video */}
          <div className="flex-shrink-0 flex items-center gap-1 px-3 py-1.5 bg-[#111111] border-b border-[#1e1e1e]">
            {video && (
              <>
                <TabButton
                  active={viewerTab === 'annotate'}
                  icon={<Crosshair size={14} />}
                  label="Annotate"
                  onClick={() => handleSetViewerTab('annotate')}
                />
                <TabButton
                  active={viewerTab === 'player'}
                  icon={<PlayCircle size={14} />}
                  label="Player"
                  onClick={() => handleSetViewerTab('player')}
                />
                <TabButton
                  active={viewerTab === 'inference'}
                  icon={<BrainCircuit size={14} />}
                  label="Inference State"
                  onClick={() => handleSetViewerTab('inference')}
                />
              </>
            )}
            <div className="flex-1" />
            <TabButton
              active={viewerTab === 'config'}
              icon={<Settings2 size={14} />}
              label="Settings"
              onClick={() => handleSetViewerTab('config')}
            />
          </div>

          {/* Viewer (grows to fill) */}
          {viewerTab === 'annotate' && <FrameViewer />}
          {viewerTab === 'player' && <VideoPlayer />}
          {viewerTab === 'inference' && <InferenceStatePanel />}
          {viewerTab === 'config' && <ConfigPanel />}

          {/* Horizontal resize handle between viewer and timeline */}
          {video && viewerTab === 'annotate' && (
            <ResizeHandle direction="vertical" onMouseDown={timelineHandleMouseDown} />
          )}

          {/* Timeline (resizable height) */}
          {video && viewerTab === 'annotate' && (
            <div className="flex-shrink-0 overflow-hidden" style={{ height: timelineHeight }}>
              <Timeline />
            </div>
          )}
        </div>
      </div>

      {/* Overlays */}
      <ProjectDrawer />
      <UploadModal />
      <ToastContainer />

      {/* No video selected — welcome prompt (inline, does NOT block header) */}
      {!video && (
        <div className="fixed inset-0 top-12 flex items-center justify-center z-10 pointer-events-none">
          <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-2xl p-8 max-w-sm w-full text-center shadow-2xl pointer-events-auto">
            <h2 className="text-xl font-bold text-white mb-2">
              {project ? `Project: ${project.name}` : 'Welcome to SAM3 Web Tracker'}
            </h2>
            <p className="text-sm text-[#888] mb-6 leading-relaxed">
              {project && Object.keys(project.videos).length === 0
                ? 'This project has no videos yet. Import a video to get started, or open the menu to switch projects.'
                : project
                ? 'Select a video from the menu to continue, or import a new one.'
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
                {project && Object.keys(project.videos).length > 0 ? 'Switch video' : 'Browse projects'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
