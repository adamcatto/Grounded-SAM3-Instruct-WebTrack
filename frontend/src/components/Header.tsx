import React from 'react'
import { Menu, Info, Database } from 'lucide-react'
import { useStore } from '../store/useStore'

export default function Header() {
  const { project, setDrawerOpen, drawerOpen } = useStore()

  return (
    <header className="flex items-center h-12 px-4 border-b border-[#2a2a2a] bg-[#111111] flex-shrink-0">
      {/* Hamburger */}
      <button
        onClick={() => setDrawerOpen(!drawerOpen)}
        className="btn btn-ghost p-2 mr-3 rounded-lg"
        title="Switch video / project"
      >
        <Menu size={18} />
      </button>

      {/* Title */}
      <div className="flex flex-col leading-none">
        <span className="text-sm font-semibold text-white tracking-tight">
          {project?.tracking_mode === 'pose_tracking' ? 'CoTracker3 Pose Tracker' : 'SAM3 Web Tracker'}
        </span>
        {project && (
          <span className="text-[11px] text-[#666] mt-0.5">{project.name}</span>
        )}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Nav links */}
      <nav className="flex items-center gap-1 text-sm text-[#999]">
        <a
          href={project?.tracking_mode === 'pose_tracking'
            ? 'https://github.com/facebookresearch/co-tracker'
            : 'https://github.com/facebookresearch/sam3'}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1 px-2 py-1 rounded hover:text-white hover:bg-[#1a1a1a] transition-colors"
        >
          <Info size={14} />
          <span>About</span>
        </a>
        <a
          href={project?.tracking_mode === 'pose_tracking'
            ? 'https://huggingface.co/facebook/cotracker3'
            : 'https://huggingface.co/facebook/sam3'}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1 px-2 py-1 rounded hover:text-white hover:bg-[#1a1a1a] transition-colors"
        >
          <Database size={14} />
          <span>Model</span>
        </a>
      </nav>
    </header>
  )
}
