import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  ArrowUp,
  Check,
  ChevronRight,
  File,
  Folder,
  FolderOpen,
  HardDrive,
  Home,
  Loader,
  Pencil,
  Target,
  Terminal,
  X,
} from 'lucide-react'
import { listDir, setProjectsRoot, type FsListEntry, type ProjectsRootInfo } from '../api/client'

type Props = {
  open: boolean
  rootInfo: ProjectsRootInfo | null
  onClose: () => void
  onApplied: () => void
  addToast: (message: string, type: 'success' | 'error' | 'info') => void
}

function pathParent(absPath: string): string {
  const normalized = absPath.replace(/\/+$/, '')
  if (!normalized || normalized === '/') return '/'
  const idx = normalized.lastIndexOf('/')
  if (idx <= 0) return '/'
  return normalized.slice(0, idx)
}

function pathBreadcrumbSegments(absPath: string): { label: string; path: string }[] {
  const normalized = absPath.replace(/\/+$/, '') || '/'
  if (normalized === '/') {
    return [{ label: '/', path: '/' }]
  }
  const parts = normalized.split('/').filter(Boolean)
  const out: { label: string; path: string }[] = [{ label: '/', path: '/' }]
  let acc = ''
  for (const p of parts) {
    acc += `/${p}`
    out.push({ label: p, path: acc })
  }
  return out
}

export default function ProjectsFolderBrowserModal({
  open,
  rootInfo,
  onClose,
  onApplied,
  addToast,
}: Props) {
  const [entries, setEntries] = useState<FsListEntry[]>([])
  const [resolvedPath, setResolvedPath] = useState('')
  const [parentPath, setParentPath] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [applying, setApplying] = useState(false)
  const [error, setError] = useState('')

  // Manual path entry
  const [pathEditing, setPathEditing] = useState(false)
  const [pathInput, setPathInput] = useState('')
  const pathInputRef = useRef<HTMLInputElement | null>(null)

  const loadPath = useCallback(async (path: string): Promise<boolean> => {
    setLoading(true)
    setError('')
    try {
      const data = await listDir(path)
      setResolvedPath(data.path)
      setParentPath(data.parent)
      setEntries(data.entries)
      return true
    } catch (e: unknown) {
      const ax = e as { response?: { data?: { detail?: string } } }
      const detail = ax?.response?.data?.detail
      setError(typeof detail === 'string' ? detail : 'Could not read directory')
      setEntries([])
      return false
    } finally {
      setLoading(false)
    }
  }, [])

  // Fallback chain so a missing/denied parent doesn't strand the user.
  const tryDefaultPath = useCallback(
    async (info: ProjectsRootInfo) => {
      const parent = pathParent(info.active_root)
      const candidates = [
        parent,
        info.active_root,
        info.home,
        info.app_root,
      ].filter((v, i, arr) => v && arr.indexOf(v) === i)
      for (const c of candidates) {
        if (await loadPath(c)) return
      }
    },
    [loadPath],
  )

  useEffect(() => {
    if (!open || !rootInfo) return
    setPathEditing(false)
    void tryDefaultPath(rootInfo)
  }, [open, rootInfo, tryDefaultPath])

  useEffect(() => {
    if (pathEditing) {
      setPathInput(resolvedPath)
      window.setTimeout(() => pathInputRef.current?.focus(), 0)
    }
  }, [pathEditing, resolvedPath])

  async function handleUseFolder() {
    setApplying(true)
    setError('')
    try {
      await setProjectsRoot(resolvedPath)
      addToast(`Projects folder: ${resolvedPath}`, 'success')
      onApplied()
      onClose()
    } catch (e: unknown) {
      const ax = e as { response?: { data?: { detail?: string } } }
      const detail = ax?.response?.data?.detail
      addToast(typeof detail === 'string' ? detail : 'Could not set projects folder', 'error')
    } finally {
      setApplying(false)
    }
  }

  function handleEntryClick(ent: FsListEntry) {
    if (!ent.is_dir) return
    void loadPath(ent.path)
  }

  function commitPathInput() {
    const v = pathInput.trim()
    if (!v) {
      setPathEditing(false)
      return
    }
    setPathEditing(false)
    void loadPath(v)
  }

  if (!open) return null

  const crumbs = pathBreadcrumbSegments(resolvedPath)
  const home = rootInfo?.home ?? ''
  const appRoot = rootInfo?.app_root ?? ''
  const activeRoot = rootInfo?.active_root ?? ''

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/65 p-3">
      <div
        className="flex flex-col w-full max-w-xl max-h-[min(620px,88vh)] rounded-xl border border-[#333] bg-[#141414] shadow-2xl overflow-hidden"
        role="dialog"
        aria-modal="true"
        aria-label="Choose projects folder"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a2a2a] shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <HardDrive size={18} className="text-blue-400 flex-shrink-0" />
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-white truncate">Projects folder</h2>
              <p className="text-[10px] text-[#666] truncate">
                Folders containing a <code className="font-mono">config.json</code> show as projects
              </p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="btn btn-ghost p-1.5 flex-shrink-0" aria-label="Close">
            <X size={18} />
          </button>
        </div>

        {/* Quick jumps */}
        <div className="px-3 py-2 border-b border-[#1e1e1e] flex flex-wrap items-center gap-1 shrink-0">
          {home && (
            <button
              type="button"
              title={`Home (${home})`}
              onClick={() => void loadPath(home)}
              className="btn btn-ghost px-2 py-1 text-[11px] text-[#aaa] hover:text-white flex items-center gap-1"
            >
              <Home size={12} />
              Home
            </button>
          )}
          {appRoot && (
            <button
              type="button"
              title={`App folder — where start_frontend.sh lives (${appRoot})`}
              onClick={() => void loadPath(appRoot)}
              className="btn btn-ghost px-2 py-1 text-[11px] text-[#aaa] hover:text-white flex items-center gap-1"
            >
              <Terminal size={12} />
              App folder
            </button>
          )}
          {activeRoot && (
            <button
              type="button"
              title={`Current projects folder (${activeRoot})`}
              onClick={() => void loadPath(activeRoot)}
              className="btn btn-ghost px-2 py-1 text-[11px] text-[#aaa] hover:text-white flex items-center gap-1"
            >
              <Target size={12} />
              Active
            </button>
          )}
          <div className="ml-auto flex items-center gap-1">
            <button
              type="button"
              title="Parent folder"
              disabled={!parentPath}
              onClick={() => parentPath && void loadPath(parentPath)}
              className="btn btn-ghost p-1.5 text-[#888] hover:text-white disabled:opacity-30"
            >
              <ArrowUp size={14} />
            </button>
            <button
              type="button"
              title="Enter path"
              onClick={() => setPathEditing(true)}
              className="btn btn-ghost p-1.5 text-[#888] hover:text-white"
            >
              <Pencil size={14} />
            </button>
          </div>
        </div>

        {/* Path bar (breadcrumbs OR text input) */}
        <div className="px-3 py-2 border-b border-[#1e1e1e] shrink-0">
          {pathEditing ? (
            <div className="flex items-center gap-1.5">
              <input
                ref={pathInputRef}
                type="text"
                value={pathInput}
                onChange={e => setPathInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    commitPathInput()
                  } else if (e.key === 'Escape') {
                    e.preventDefault()
                    setPathEditing(false)
                  }
                }}
                placeholder="/absolute/path/to/folder"
                className="flex-1 text-xs py-1 px-2 rounded bg-[#0f0f0f] border border-[#333] text-[#ddd] font-mono"
              />
              <button
                type="button"
                onClick={commitPathInput}
                className="btn btn-primary px-2 py-1 text-[11px] flex items-center gap-1"
              >
                <Check size={12} />
                Go
              </button>
              <button
                type="button"
                onClick={() => setPathEditing(false)}
                className="btn btn-ghost px-2 py-1 text-[11px]"
              >
                Cancel
              </button>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-0.5 text-[11px] text-[#888] font-mono leading-snug min-h-[1.5rem]">
              {crumbs.map((c, i) => (
                <React.Fragment key={c.path}>
                  {i > 0 && <ChevronRight size={12} className="text-[#555] flex-shrink-0" />}
                  <button
                    type="button"
                    onClick={() => void loadPath(c.path)}
                    className={`truncate max-w-[10rem] hover:text-blue-400 transition-colors px-0.5 ${
                      c.path === resolvedPath ? 'text-blue-300' : ''
                    }`}
                    title={c.path}
                  >
                    {c.label}
                  </button>
                </React.Fragment>
              ))}
            </div>
          )}
        </div>

        {/* Folder/file viewer */}
        <div className="flex-1 min-h-[220px] overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center gap-2 py-16 text-[#666] text-xs">
              <Loader size={16} className="animate-spin" />
              Loading…
            </div>
          )}
          {!loading && error && (
            <p className="text-xs text-red-400 px-4 py-6">{error}</p>
          )}
          {!loading && !error && (
            <ul className="py-1">
              {entries.length === 0 && (
                <li className="px-4 py-8 text-center text-xs text-[#555]">This folder is empty</li>
              )}
              {entries.map(ent => (
                <li key={ent.path}>
                  <button
                    type="button"
                    disabled={!ent.is_dir}
                    onClick={() => handleEntryClick(ent)}
                    className={`w-full flex items-center gap-2 px-4 py-2.5 text-left text-sm transition-colors
                      ${ent.is_dir ? 'hover:bg-[#1e1e1e] text-[#ddd] cursor-pointer' : 'text-[#555] cursor-default opacity-70'}`}
                  >
                    {ent.is_dir ? (
                      ent.is_project ? (
                        <FolderOpen size={16} className="text-amber-400 flex-shrink-0" />
                      ) : (
                        <Folder size={16} className="text-blue-400/90 flex-shrink-0" />
                      )
                    ) : (
                      <File size={16} className="text-[#555] flex-shrink-0" />
                    )}
                    <span className="flex-1 truncate font-medium">{ent.name}</span>
                    {ent.is_project && (
                      <span className="text-[10px] uppercase tracking-wide text-amber-500/90 flex-shrink-0 px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/25">
                        Project
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="px-4 py-3 border-t border-[#2a2a2a] flex flex-col gap-2 shrink-0">
          <p className="text-[10px] text-[#666] font-mono break-all leading-relaxed">
            {resolvedPath || '—'}
          </p>
          <div className="flex gap-2">
            <button type="button" onClick={onClose} className="btn btn-ghost flex-1 py-2 text-xs">
              Cancel
            </button>
            <button
              type="button"
              disabled={applying || !!loading || !resolvedPath}
              onClick={() => void handleUseFolder()}
              className="btn btn-primary flex-1 py-2 text-xs font-medium disabled:opacity-40"
            >
              {applying ? 'Applying…' : 'Use this folder'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
