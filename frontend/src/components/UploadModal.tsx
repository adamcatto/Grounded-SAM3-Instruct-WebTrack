import React, { useCallback, useRef, useState } from 'react'
import { X, Upload, Film, Server, FolderOpen } from 'lucide-react'
import { useStore } from '../store/useStore'
import { createProject, addVideo, importVideo, getProject } from '../api/client'

type InputMode = 'upload' | 'server'

export default function UploadModal() {
  const {
    uploadModalOpen, setUploadModalOpen,
    project, setProject, setCurrentVideo,
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
  // Shared
  const [extracting, setExtracting] = useState(false)
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const isNewProject = !project
  const busy = uploading || importing

  function handleClose() {
    if (busy) return
    setUploadModalOpen(false)
    setSelectedFile(null)
    setProjectName('')
    setServerPath('')
    setError('')
    setUploadProgress(0)
    setExtracting(false)
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

  async function handleUpload() {
    if (!selectedFile) return
    setUploading(true)
    setError('')
    try {
      const pid = await ensureProject()
      const video = await addVideo(pid, selectedFile, pct => setUploadProgress(pct))
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
      const video = await importVideo(pid, serverPath.trim())
      await pollUntilReady(pid, video.id)
      handleClose()
    } catch (e: unknown) {
      if (e instanceof Error) {
        // Extract backend error message from axios response
        const axiosErr = e as any
        const detail = axiosErr?.response?.data?.detail
        setError(detail ?? e.message)
      } else {
        setError('Import failed')
      }
    } finally {
      setImporting(false)
      setExtracting(false)
    }
  }

  if (!uploadModalOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-2xl w-[520px] max-w-[95vw] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a2a]">
          <h2 className="text-base font-semibold text-white">
            {isNewProject ? 'Start a new project' : `Add video to "${project?.name}"`}
          </h2>
          <button onClick={handleClose} className="btn btn-ghost p-1" disabled={busy}>
            <X size={16} />
          </button>
        </div>

        <div className="p-6 space-y-4">
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
              Server path
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

          {/* ─── Server path input ─── */}
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
          {(uploading || importing) && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-[#888]">
                <span>
                  {extracting
                    ? 'Extracting frames...'
                    : importing
                      ? 'Importing...'
                      : `Uploading... ${uploadProgress}%`}
                </span>
              </div>
              <div className="h-1.5 bg-[#333] rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-300"
                  style={{ width: extracting || importing ? '100%' : `${uploadProgress}%` }}
                />
              </div>
              {extracting && (
                <p className="text-xs text-[#666]">
                  Extracting preview frames — this should be quick.
                </p>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <p className="text-sm text-red-400 bg-red-400/10 rounded-lg px-3 py-2">{error}</p>
          )}

          {/* Actions */}
          <div className="flex gap-3 pt-1">
            <button onClick={handleClose} className="btn btn-secondary flex-1" disabled={busy}>
              Cancel
            </button>
            {inputMode === 'server' ? (
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
            ) : (
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
      </div>
    </div>
  )
}
