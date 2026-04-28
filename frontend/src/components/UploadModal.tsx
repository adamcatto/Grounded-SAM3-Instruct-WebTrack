import React, { useCallback, useRef, useState, useMemo } from 'react'
import { X, Upload, Film, Server, FolderOpen, Search, CheckSquare, Square, AlertTriangle } from 'lucide-react'
import { useStore } from '../store/useStore'
import { createProject, addVideo, importVideo, getProject, browseDirectory } from '../api/client'
import type { BrowseEntry, DownsampleOptions } from '../api/client'

type InputMode = 'upload' | 'server' | 'folder'

/** Convert a simple glob pattern (*, ?) to a case-insensitive RegExp. */
function globToRegex(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&')
  const re = escaped.replace(/\*/g, '.*').replace(/\?/g, '.')
  return new RegExp('^' + re + '$', 'i')
}

function matchesPatterns(name: string, include: string, exclude: string): boolean {
  const inc = include.trim()
  const exc = exclude.trim()
  if (inc && inc !== '*') {
    if (!globToRegex(inc).test(name)) return false
  }
  if (exc) {
    if (globToRegex(exc).test(name)) return false
  }
  return true
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/** FastAPI `{ detail }` shape (string or validation error list). */
function formatAxiosDetail(err: unknown): string {
  const ax = err as { response?: { data?: { detail?: unknown } } }
  const detail = ax?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item: unknown) =>
        typeof item === 'object' && item !== null && 'msg' in item
          ? String((item as { msg: unknown }).msg)
          : String(item),
      )
      .join('; ')
  }
  if (detail !== undefined && detail !== null) return String(detail)
  return err instanceof Error ? err.message : 'Import failed'
}

type ImportProceedPrompt = {
  failedName: string
  detail: string
  remainingCount: number
}

export default function UploadModal() {
  const {
    uploadModalOpen, setUploadModalOpen,
    project, setProject, setCurrentVideo,
    addToast,
  } = useStore()

  const [projectName, setProjectName] = useState('')
  const [inputMode, setInputMode] = useState<InputMode>('server')

  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  // Server path state
  const [serverPath, setServerPath] = useState('')
  const [importing, setImporting] = useState(false)

  // Folder browse state
  const [folderPath, setFolderPath] = useState('')
  const [scanning, setScanning] = useState(false)
  const [browseFiles, setBrowseFiles] = useState<BrowseEntry[] | null>(null)
  const [includePattern, setIncludePattern] = useState('*')
  const [excludePattern, setExcludePattern] = useState('')
  const [checkedPaths, setCheckedPaths] = useState<Set<string>>(new Set())
  const [folderImporting, setFolderImporting] = useState(false)
  const [folderImportProgress, setFolderImportProgress] = useState<{
    phase: number
    total: number
    importedOk: number
    skipped: number
  } | null>(null)
  const [importProceedPrompt, setImportProceedPrompt] = useState<ImportProceedPrompt | null>(null)
  const proceedResolverRef = useRef<((v: boolean) => void) | null>(null)
  const [recursive, setRecursive] = useState(false)
  const [scanDepth, setScanDepth] = useState(3)

  // Symlink (server/folder modes only)
  const [useSymlink, setUseSymlink] = useState(false)

  // Downsample (shared across all modes)
  const [downsample, setDownsample] = useState(false)
  const [dsMode, setDsMode] = useState<'max_dim' | 'factor'>('max_dim')
  const [maxDim, setMaxDim] = useState(1080)
  const [dsFactor, setDsFactor] = useState(2)

  // Shared
  const [extracting, setExtracting] = useState(false)
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const isNewProject = !project
  const busy = uploading || importing || folderImporting

  // Filtered file list based on patterns
  const filteredFiles = useMemo(() => {
    if (!browseFiles) return []
    return browseFiles.filter(f => matchesPatterns(f.name, includePattern, excludePattern))
  }, [browseFiles, includePattern, excludePattern])

  function handleClose() {
    if (busy) return
    setUploadModalOpen(false)
    setSelectedFile(null)
    setProjectName('')
    setServerPath('')
    setError('')
    setUploadProgress(0)
    setExtracting(false)
    setBrowseFiles(null)
    setFolderPath('')
    setCheckedPaths(new Set())
    setFolderImportProgress(null)
    setIncludePattern('*')
    setExcludePattern('')
    setUseSymlink(false)
    setImportProceedPrompt(null)
    proceedResolverRef.current = null
  }

  function resolveImportProceed(cont: boolean) {
    proceedResolverRef.current?.(cont)
    proceedResolverRef.current = null
    setImportProceedPrompt(null)
  }

  function handleFile(file: File) {
    const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
    if (!['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext)) {
      setError('Please select a video file (mp4, avi, mov, mkv, webm)')
      return
    }
    setSelectedFile(file)
    setError('')
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [])

  async function ensureProject(): Promise<string> {
    if (project) return project.id
    if (!projectName.trim()) throw new Error('Please enter a project name')
    const proj = await createProject(projectName.trim())
    setProject(proj)
    return proj.id
  }

  async function pollUntilReady(pid: string, vid: string) {
    setExtracting(true)
    let attempts = 0
    while (attempts < 120) {
      await new Promise(r => setTimeout(r, 2000))
      const freshProject = await getProject(pid)
      const freshVideo = freshProject.videos[vid]
      if (freshVideo?.frames_extracted) {
        setProject(freshProject)
        setCurrentVideo(vid)
        return
      }
      attempts++
    }
    throw new Error('Frame extraction timed out')
  }

  function getDsOptions(): DownsampleOptions | undefined {
    if (!downsample) return undefined
    return dsMode === 'factor' ? { scaleFactor: dsFactor } : { maxDim }
  }

  async function handleUpload() {
    if (!selectedFile) return
    setUploading(true)
    setError('')
    try {
      const pid = await ensureProject()
      const video = await addVideo(pid, selectedFile, pct => setUploadProgress(pct), getDsOptions())
      await pollUntilReady(pid, video.id)
      handleClose()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
      setExtracting(false)
    }
  }

  async function handleImport() {
    if (!serverPath.trim()) {
      setError('Please enter a file path')
      return
    }
    setImporting(true)
    setError('')
    try {
      const pid = await ensureProject()
      const video = await importVideo(pid, serverPath.trim(), getDsOptions(), useSymlink)
      await pollUntilReady(pid, video.id)
      handleClose()
    } catch (e: unknown) {
      setError(formatAxiosDetail(e))
    } finally {
      setImporting(false)
      setExtracting(false)
    }
  }

  async function handleScanFolder() {
    if (!folderPath.trim()) {
      setError('Please enter a folder path')
      return
    }
    setScanning(true)
    setError('')
    setBrowseFiles(null)
    setCheckedPaths(new Set())
    try {
      const result = await browseDirectory(folderPath.trim(), recursive ? scanDepth : 1)
      setBrowseFiles(result.files)
      // Auto-select all video files
      const videoPaths = new Set(result.files.filter(f => f.is_video).map(f => f.path))
      setCheckedPaths(videoPaths)
    } catch (e: unknown) {
      const axiosErr = e as any
      const detail = axiosErr?.response?.data?.detail
      setError(detail ?? (e instanceof Error ? e.message : 'Scan failed'))
    } finally {
      setScanning(false)
    }
  }

  // Re-sync checked set when filter changes: uncheck files no longer visible
  function handlePatternChange(newInclude: string, newExclude: string) {
    if (browseFiles) {
      const nowVisible = new Set(
        browseFiles
          .filter(f => matchesPatterns(f.name, newInclude, newExclude))
          .map(f => f.path)
      )
      setCheckedPaths(prev => new Set([...prev].filter(p => nowVisible.has(p))))
    }
  }

  function toggleFile(path: string) {
    setCheckedPaths(prev => {
      const next = new Set(prev)
      next.has(path) ? next.delete(path) : next.add(path)
      return next
    })
  }

  function selectAll() {
    setCheckedPaths(new Set(filteredFiles.map(f => f.path)))
  }

  function selectNone() {
    setCheckedPaths(new Set())
  }

  async function handleFolderImport() {
    const toImport = filteredFiles.filter(f => checkedPaths.has(f.path))
    if (!toImport.length) {
      setError('No files selected')
      return
    }
    const promptContinue = (info: ImportProceedPrompt) =>
      new Promise<boolean>(resolve => {
        proceedResolverRef.current = resolve
        setImportProceedPrompt(info)
      })

    setFolderImporting(true)
    setError('')
    let successCount = 0
    let lastVid: string | null = null
    const skipped: { name: string; detail: string }[] = []

    try {
      const pid = await ensureProject()
      for (let i = 0; i < toImport.length; ) {
        const entry = toImport[i]
        setFolderImportProgress({
          phase: i + 1,
          total: toImport.length,
          importedOk: successCount,
          skipped: skipped.length,
        })
        try {
          const video = await importVideo(pid, entry.path, getDsOptions(), useSymlink)
          lastVid = video.id
          successCount++
          i++
        } catch (e: unknown) {
          const detail = formatAxiosDetail(e)
          const remainingCount = toImport.length - i - 1
          const proceed = await promptContinue({
            failedName: entry.name,
            detail,
            remainingCount,
          })
          if (!proceed) {
            const freshProject = await getProject(pid).catch(() => null)
            if (freshProject) setProject(freshProject)
            if (successCount === 0 && skipped.length === 0) {
              setError(`Could not import ${entry.name}: ${detail}`)
            } else {
              setError(
                `Stopped after ${successCount} successful import${successCount !== 1 ? 's' : ''}. Problem file: ${entry.name}: ${detail}`,
              )
            }
            return
          }
          skipped.push({ name: entry.name, detail })
          i++
        }
      }
      if (lastVid) {
        setExtracting(true)
        await pollUntilReady(pid, lastVid)
      }
      handleClose()
      if (skipped.length > 0) {
        if (successCount > 0)
          addToast(
            `Imported ${successCount} video(s); skipped ${skipped.length} (${skipped.map(s => s.name).join(', ')}).`,
            'info',
          )
        else
          addToast(
            `Skipped ${skipped.length} ${skipped.length === 1 ? 'file' : 'files'} (${skipped.map(s => s.name).join(', ')}).`,
            'info',
          )
      }
    } catch (e: unknown) {
      setError(formatAxiosDetail(e))
    } finally {
      setFolderImporting(false)
      setExtracting(false)
      setFolderImportProgress(null)
      const pending = proceedResolverRef.current
      proceedResolverRef.current = null
      setImportProceedPrompt(null)
      pending?.(false)
    }
  }

  if (!uploadModalOpen) return null

  const checkedCount = filteredFiles.filter(f => checkedPaths.has(f.path)).length

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-2xl w-[580px] max-w-[95vw] shadow-2xl flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a2a] flex-shrink-0">
          <h2 className="text-base font-semibold text-white">
            {isNewProject ? 'Start a new project' : `Add video to "${project?.name}"`}
          </h2>
          <button onClick={handleClose} className="btn btn-ghost p-1" disabled={busy}>
            <X size={16} />
          </button>
        </div>

        <div className="p-6 space-y-4 overflow-y-auto flex-1">
          {/* Project name (new project only) */}
          {isNewProject && (
            <div className="space-y-1">
              <label className="text-xs text-[#888] font-medium">Project name</label>
              <input
                type="text"
                value={projectName}
                onChange={e => setProjectName(e.target.value)}
                placeholder="e.g. My Tracking Project"
                className="w-full"
                disabled={busy}
              />
            </div>
          )}

          {/* Mode tabs */}
          <div className="flex gap-1 bg-[#111] rounded-lg p-0.5">
            <button
              onClick={() => setInputMode('server')}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-medium transition-colors
                ${inputMode === 'server' ? 'bg-[#2a2a2a] text-white' : 'text-[#666] hover:text-[#aaa]'}`}
              disabled={busy}
            >
              <Server size={13} />
              Server file
            </button>
            <button
              onClick={() => setInputMode('folder')}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-medium transition-colors
                ${inputMode === 'folder' ? 'bg-[#2a2a2a] text-white' : 'text-[#666] hover:text-[#aaa]'}`}
              disabled={busy}
            >
              <FolderOpen size={13} />
              Server folder
            </button>
            <button
              onClick={() => setInputMode('upload')}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-medium transition-colors
                ${inputMode === 'upload' ? 'bg-[#2a2a2a] text-white' : 'text-[#666] hover:text-[#aaa]'}`}
              disabled={busy}
            >
              <Upload size={13} />
              Upload file
            </button>
          </div>

          {/* ─── Server file input ─── */}
          {inputMode === 'server' && (
            <div className="space-y-2">
              <label className="text-xs text-[#888] font-medium">Video file path on server</label>
              <input
                type="text"
                value={serverPath}
                onChange={e => setServerPath(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleImport()}
                placeholder="/path/to/video.mp4"
                className="w-full font-mono text-sm"
                disabled={busy}
              />
              <p className="text-xs text-[#555]">
                Enter the absolute path to a video file on the server filesystem.
                Supported formats: MP4, AVI, MOV, MKV, WebM
              </p>
            </div>
          )}

          {/* ─── Server folder browser ─── */}
          {inputMode === 'folder' && (
            <div className="space-y-3">
              {/* Folder path + scan */}
              <div className="space-y-1">
                <label className="text-xs text-[#888] font-medium">Folder path on server</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={folderPath}
                    onChange={e => setFolderPath(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleScanFolder()}
                    placeholder="/path/to/folder"
                    className="flex-1 font-mono text-sm"
                    disabled={busy || scanning}
                  />
                  <button
                    onClick={handleScanFolder}
                    disabled={!folderPath.trim() || busy || scanning}
                    className="btn btn-secondary flex items-center gap-1.5 px-3 flex-shrink-0"
                  >
                    {scanning
                      ? <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      : <Search size={13} />}
                    Scan
                  </button>
                </div>
              </div>

              {/* Recursive options */}
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={recursive}
                    onChange={e => setRecursive(e.target.checked)}
                    disabled={busy || scanning}
                    className="w-3.5 h-3.5 accent-blue-500"
                  />
                  <span className="text-xs text-[#aaa]">Recursive</span>
                </label>
                {recursive && (
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-[#888]">Depth</label>
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={scanDepth === 0 ? '' : scanDepth}
                      onChange={e => {
                        const v = parseInt(e.target.value, 10)
                        setScanDepth(isNaN(v) || v < 1 ? 0 : Math.min(v, 20))
                      }}
                      placeholder="∞"
                      className="w-16 text-xs font-mono text-center text-black"
                      disabled={busy || scanning}
                    />
                    <span className="text-xs text-[#555]">(blank = unlimited)</span>
                  </div>
                )}
              </div>

              {/* Pattern filters */}
              {browseFiles !== null && (
                <>
                  <div className="flex gap-3">
                    <div className="flex-1 space-y-1">
                      <label className="text-xs text-[#888] font-medium">Include pattern</label>
                      <input
                        type="text"
                        value={includePattern}
                        onChange={e => {
                          setIncludePattern(e.target.value)
                          handlePatternChange(e.target.value, excludePattern)
                        }}
                        placeholder="* (all files)"
                        className="w-full font-mono text-xs"
                        disabled={busy}
                      />
                    </div>
                    <div className="flex-1 space-y-1">
                      <label className="text-xs text-[#888] font-medium">Exclude pattern</label>
                      <input
                        type="text"
                        value={excludePattern}
                        onChange={e => {
                          setExcludePattern(e.target.value)
                          handlePatternChange(includePattern, e.target.value)
                        }}
                        placeholder="e.g. *.h264"
                        className="w-full font-mono text-xs"
                        disabled={busy}
                      />
                    </div>
                  </div>
                  <p className="text-xs text-[#555]">
                    Use <code className="text-[#888]">*</code> as wildcard — e.g.{' '}
                    <code className="text-[#888]">*_video_*.mp4</code> or{' '}
                    <code className="text-[#888]">*.h264</code>
                  </p>

                  {/* File list */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-[#888]">
                        {filteredFiles.length} file{filteredFiles.length !== 1 ? 's' : ''} matched
                        {browseFiles.length !== filteredFiles.length && ` (${browseFiles.length} total)`}
                      </span>
                      <div className="flex gap-2">
                        <button
                          onClick={selectAll}
                          className="text-xs text-[#666] hover:text-[#aaa] transition-colors"
                          disabled={busy}
                        >
                          All
                        </button>
                        <span className="text-[#444]">·</span>
                        <button
                          onClick={selectNone}
                          className="text-xs text-[#666] hover:text-[#aaa] transition-colors"
                          disabled={busy}
                        >
                          None
                        </button>
                      </div>
                    </div>

                    {filteredFiles.length === 0 ? (
                      <div className="text-xs text-[#555] py-4 text-center border border-[#2a2a2a] rounded-lg">
                        No files match the current pattern
                      </div>
                    ) : (
                      <div className="border border-[#2a2a2a] rounded-lg overflow-hidden max-h-52 overflow-y-auto">
                        {filteredFiles.map(file => {
                          const checked = checkedPaths.has(file.path)
                          return (
                            <label
                              key={file.path}
                              className={`flex items-center gap-3 px-3 py-2 cursor-pointer transition-colors border-b border-[#222] last:border-b-0
                                ${checked ? 'bg-blue-500/5' : 'hover:bg-[#222]'}
                                ${busy ? 'cursor-not-allowed opacity-60' : ''}`}
                            >
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => !busy && toggleFile(file.path)}
                                className="hidden"
                              />
                              {checked
                                ? <CheckSquare size={14} className="text-blue-400 flex-shrink-0" />
                                : <Square size={14} className="text-[#555] flex-shrink-0" />}
                              <span className="flex-1 text-xs font-mono text-[#ccc] truncate" title={file.name}>
                                {file.name}
                              </span>
                              <span className="text-xs text-[#555] flex-shrink-0">{formatSize(file.size)}</span>
                              {file.is_video && (
                                <Film size={11} className="text-blue-400/60 flex-shrink-0" />
                              )}
                            </label>
                          )
                        })}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {/* ─── File upload drop zone ─── */}
          {inputMode === 'upload' && (
            <div
              onDrop={onDrop}
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onClick={() => !busy && fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors
                ${dragOver ? 'border-blue-500 bg-blue-500/10' : 'border-[#333] hover:border-[#555]'}
                ${busy ? 'cursor-not-allowed opacity-60' : ''}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
              {selectedFile ? (
                <div className="flex flex-col items-center gap-2">
                  <Film size={32} className="text-blue-400" />
                  <span className="text-sm font-medium text-white">{selectedFile.name}</span>
                  <span className="text-xs text-[#666]">
                    {(selectedFile.size / 1024 / 1024).toFixed(1)} MB
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <Upload size={32} className="text-[#555]" />
                  <span className="text-sm text-[#888]">Drop a video here or click to browse</span>
                  <span className="text-xs text-[#555]">MP4, AVI, MOV, MKV, WebM</span>
                </div>
              )}
            </div>
          )}

          {/* Progress */}
          {(uploading || importing || folderImporting) && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-[#888]">
                <span>
                  {extracting
                    ? 'Extracting frames...'
                    : folderImporting && folderImportProgress
                      ? `Importing · file ${folderImportProgress.phase} / ${folderImportProgress.total}${folderImportProgress.importedOk > 0 ? ` (${folderImportProgress.importedOk} done)` : ''}${folderImportProgress.skipped ? ` (${folderImportProgress.skipped} skipped)` : ''}`
                      : importing
                        ? 'Importing...'
                        : `Uploading... ${uploadProgress}%`}
                </span>
              </div>
              <div className="h-1.5 bg-[#333] rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-300"
                  style={{
                    width: extracting || importing
                      ? '100%'
                      : folderImporting && folderImportProgress
                        ? `${(folderImportProgress.phase / folderImportProgress.total) * 100}%`
                        : `${uploadProgress}%`
                  }}
                />
              </div>
              {extracting && (
                <p className="text-xs text-[#666]">
                  Extracting preview frames — this should be quick.
                </p>
              )}
            </div>
          )}

          {/* Symlink (server/folder modes only) */}
          {(inputMode === 'server' || inputMode === 'folder') && (
            <div className="border border-[#2a2a2a] rounded-xl px-4 py-3 space-y-1">
              <label className="flex items-center gap-2.5 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={useSymlink}
                  onChange={e => setUseSymlink(e.target.checked)}
                  disabled={busy}
                  className="w-3.5 h-3.5 accent-blue-500"
                />
                <span className="text-sm text-[#ccc] font-medium">Symlink (don't copy)</span>
              </label>
              <p className="text-xs text-[#555] pl-6 leading-relaxed">
                Creates a symlink to the original file instead of copying it — saves disk space.
                The original is <strong className="text-[#888]">never modified</strong>.
                {useSymlink && downsample && (
                  <span className="block mt-1 text-amber-400/80">
                    Downsampling is applied lazily per-frame when extracted — the source file is unchanged.
                  </span>
                )}
              </p>
            </div>
          )}

          {/* Downsample */}
          <div className="border border-[#2a2a2a] rounded-xl px-4 py-3 space-y-2">
            <label className="flex items-center gap-2.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={downsample}
                onChange={e => setDownsample(e.target.checked)}
                disabled={busy}
                className="w-3.5 h-3.5 accent-blue-500"
              />
              <span className="text-sm text-[#ccc] font-medium">Downsample video</span>
            </label>
            {downsample && (
              <div className="pl-6 space-y-2">
                <div className="flex gap-3">
                  {(['max_dim', 'factor'] as const).map(m => (
                    <label key={m} className="flex items-center gap-1.5 cursor-pointer select-none">
                      <input
                        type="radio"
                        checked={dsMode === m}
                        onChange={() => setDsMode(m)}
                        disabled={busy}
                        className="accent-blue-500"
                      />
                      <span className="text-xs text-[#aaa]">{m === 'max_dim' ? 'Max dimension' : 'Scale factor'}</span>
                    </label>
                  ))}
                </div>
                {dsMode === 'max_dim' ? (
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min={64}
                      max={7680}
                      value={maxDim}
                      onChange={e => setMaxDim(Math.max(64, Math.min(7680, parseInt(e.target.value) || 1080)))}
                      className="w-20 text-xs text-black font-mono text-center"
                      disabled={busy}
                    />
                    <span className="text-xs text-[#555]">px — longest side clamped to this value</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <select
                      value={dsFactor}
                      onChange={e => setDsFactor(Number(e.target.value))}
                      disabled={busy}
                      className="text-xs text-black font-mono px-2 py-1 rounded border border-[#333] bg-white"
                    >
                      {[2, 3, 4, 6, 8].map(f => <option key={f} value={f}>{f}×</option>)}
                    </select>
                    <span className="text-xs text-[#555]">each dimension divided by this factor</span>
                  </div>
                )}
                <p className="text-xs text-[#555] leading-relaxed">
                  Frames are resized lazily on extraction — no re-encoding of the source file.
                </p>
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <p className="text-sm text-red-400 bg-red-400/10 rounded-lg px-3 py-2">{error}</p>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-3 px-6 py-4 border-t border-[#2a2a2a] flex-shrink-0">
          <button onClick={handleClose} className="btn btn-secondary flex-1" disabled={busy}>
            Cancel
          </button>
          {inputMode === 'server' && (
            <button
              onClick={handleImport}
              disabled={!serverPath.trim() || busy || (isNewProject && !projectName.trim())}
              className="btn btn-primary flex-1 flex items-center justify-center gap-2"
            >
              {importing ? (
                <>
                  <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  {extracting ? 'Extracting...' : 'Importing...'}
                </>
              ) : (
                <>
                  <FolderOpen size={14} />
                  {isNewProject ? 'Create & Import' : 'Import Video'}
                </>
              )}
            </button>
          )}
          {inputMode === 'folder' && (
            <button
              onClick={handleFolderImport}
              disabled={checkedCount === 0 || busy || (isNewProject && !projectName.trim())}
              className="btn btn-primary flex-1 flex items-center justify-center gap-2"
            >
              {folderImporting ? (
                <>
                  <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  {extracting ? 'Extracting...' : `Importing ${folderImportProgress?.phase ?? 0}/${folderImportProgress?.total ?? 0}...`}
                </>
              ) : (
                <>
                  <FolderOpen size={14} />
                  {checkedCount > 0
                    ? (isNewProject ? `Create & Import ${checkedCount} file${checkedCount !== 1 ? 's' : ''}` : `Import ${checkedCount} file${checkedCount !== 1 ? 's' : ''}`)
                    : 'Select files to import'}
                </>
              )}
            </button>
          )}
          {inputMode === 'upload' && (
            <button
              onClick={handleUpload}
              disabled={!selectedFile || busy || (isNewProject && !projectName.trim())}
              className="btn btn-primary flex-1 flex items-center justify-center gap-2"
            >
              {uploading ? (
                <>
                  <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  {extracting ? 'Extracting...' : 'Uploading...'}
                </>
              ) : (
                <>
                  <Upload size={14} />
                  {isNewProject ? 'Create & Upload' : 'Upload Video'}
                </>
              )}
            </button>
          )}
        </div>
      </div>
      {importProceedPrompt && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/75 p-4 pointer-events-auto">
          <div className="bg-[#252525] border border-[#3a3a3a] rounded-xl max-w-lg w-full p-5 shadow-2xl pointer-events-auto">
            <div className="flex gap-3">
              <AlertTriangle size={22} className="text-amber-400 flex-shrink-0 mt-0.5" />
              <div className="min-w-0 flex-1 space-y-2">
                <h3 className="text-sm font-semibold text-white">Could not import this video</h3>
                <p className="text-xs font-medium text-[#ccc] truncate" title={importProceedPrompt.failedName}>
                  {importProceedPrompt.failedName}
                </p>
                <p className="text-xs text-amber-200/90 break-words">{importProceedPrompt.detail}</p>
                {importProceedPrompt.remainingCount > 0 ? (
                  <p className="text-xs text-[#888] leading-relaxed pt-1">
                    {importProceedPrompt.remainingCount} other file
                    {importProceedPrompt.remainingCount !== 1 ? 's' : ''} in this batch can still be imported.
                    Continue with those and skip this one?
                  </p>
                ) : (
                  <p className="text-xs text-[#888] leading-relaxed pt-1">
                    This was the last file in the batch. You can skip it and finish, or stop and keep what was
                    already imported.
                  </p>
                )}
              </div>
            </div>
            <div className="flex flex-wrap gap-2 justify-end mt-5">
              <button
                type="button"
                className="btn btn-secondary px-4"
                onClick={() => resolveImportProceed(false)}
              >
                {importProceedPrompt.remainingCount > 0 ? 'Stop import' : 'Stop'}
              </button>
              <button
                type="button"
                className="btn btn-primary px-4"
                onClick={() => resolveImportProceed(true)}
              >
                {importProceedPrompt.remainingCount > 0 ? 'Continue with rest' : 'Skip file & finish'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
