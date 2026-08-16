import React from 'react'
import { Save } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import { checkHealth, updateVideoMeta, downsampleVideo, getAgentStatus, type AgentLlmStatus } from '../api/client'
import type { DownsampleOptions } from '../api/client'
import { useEffect, useState } from 'react'
import NumericDraftInput from './NumericDraftInput'
import {
  ANCHOR_BATCH_SIZE_MAX,
  ANCHOR_BATCH_SIZE_MIN,
  computeAnchorFrames,
  hasAnchorLabelingStarted,
  normalizeAnchorBatchSize,
  videoAnchorBatchSize,
} from '../utils/anchorFrames'

// ─── Primitive controls ────────────────────────────────────────────────────────

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      role="switch"
      aria-checked={value}
      onClick={() => onChange(!value)}
      className={`relative inline-flex h-5 w-9 flex-shrink-0 rounded-full border-2 transition-colors duration-150 focus:outline-none ${
        value ? 'bg-blue-500 border-blue-500' : 'bg-[#333] border-[#444]'
      }`}
    >
      <span
        className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform duration-150 mt-px ${
          value ? 'translate-x-4' : 'translate-x-0.5'
        }`}
      />
    </button>
  )
}

function Slider({
  value, onChange, min, max, step, format,
}: {
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  format: (v: number) => string
}) {
  return (
    <div className="flex items-center gap-3 flex-shrink-0">
      <span className="text-xs font-mono text-[#888] w-10 text-right tabular-nums">
        {format(value)}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-28 accent-blue-500"
      />
    </div>
  )
}

// ─── Row variants ──────────────────────────────────────────────────────────────

function ToggleRow({ label, description, value, onChange }: {
  label: string
  description: string
  value: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-start gap-4 py-3 border-b border-[#1e1e1e] last:border-0">
      <div className="flex-1 min-w-0">
        <p className="text-sm text-[#ddd] font-medium">{label}</p>
        <p className="text-xs text-[#666] mt-0.5 leading-relaxed">{description}</p>
      </div>
      <Toggle value={value} onChange={onChange} />
    </div>
  )
}

function SliderRow({ label, description, value, onChange, min, max, step, format }: {
  label: string
  description: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  format: (v: number) => string
}) {
  return (
    <div className="flex items-start gap-4 py-3 border-b border-[#1e1e1e] last:border-0">
      <div className="flex-1 min-w-0">
        <p className="text-sm text-[#ddd] font-medium">{label}</p>
        <p className="text-xs text-[#666] mt-0.5 leading-relaxed">{description}</p>
      </div>
      <Slider value={value} onChange={onChange} min={min} max={max} step={step} format={format} />
    </div>
  )
}

function NumberRow({ label, description, value, onChange, onCommit, min, max, disabled }: {
  label: string
  description: string
  value: number
  onChange: (v: number) => void
  onCommit: (v: number) => void
  min: number
  max: number
  disabled?: boolean
}) {
  return (
    <div className={`flex items-start gap-4 py-3 border-b border-[#1e1e1e] last:border-0 ${disabled ? 'opacity-50' : ''}`}>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-[#ddd] font-medium">{label}</p>
        <p className="text-xs text-[#666] mt-0.5 leading-relaxed">{description}</p>
      </div>
      <NumericDraftInput
        value={value}
        onChange={v => {
          onChange(v)
          onCommit(v)
        }}
        min={min}
        max={max}
        disabled={disabled}
        className="w-20 flex-shrink-0 text-xs py-1 px-2 rounded bg-[#1a1a1a] border border-[#333] text-[#ccc] text-right disabled:cursor-not-allowed"
      />
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-3">{title}</h2>
      <div className="rounded-xl border border-[#2a2a2a] bg-[#111] px-4 divide-y divide-[#1e1e1e]">
        {children}
      </div>
    </div>
  )
}

// ─── Main panel ───────────────────────────────────────────────────────────────

export default function ConfigPanel() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    config, setConfig, configDirty, persistConfig,
    propagationStartFrame, setPropagationStartFrame, setCurrentFrame,
    project, currentVideoId, updateVideo, addToast,
  } = store

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''

  const [modelInfo, setModelInfo] = useState<{ model: string; sam_ready: boolean } | null>(null)
  const [agentInfo, setAgentInfo] = useState<AgentLlmStatus | null>(null)
  const [dsMode, setDsMode] = useState<'max_dim' | 'factor'>('max_dim')
  const [dsMaxDim, setDsMaxDim] = useState(1080)
  const [dsFactor, setDsFactor] = useState(2)
  const [dsRunning, setDsRunning] = useState(false)
  const [dsMessage, setDsMessage] = useState<{ text: string; ok: boolean } | null>(null)
  const [dsAllRunning, setDsAllRunning] = useState(false)
  const [dsAllProgress, setDsAllProgress] = useState<{ done: number; total: number; current: string } | null>(null)
  const [dsAllMessage, setDsAllMessage] = useState<{ text: string; ok: boolean } | null>(null)
  const [anchorBatchSizeLocal, setAnchorBatchSizeLocal] = useState(() =>
    videoAnchorBatchSize(video, config.anchorBatchSize),
  )

  useEffect(() => {
    checkHealth().then(h => setModelInfo({ model: h.sam_model, sam_ready: h.sam_ready })).catch(() => {})
    getAgentStatus().then(setAgentInfo).catch(() => {})
  }, [])

  useEffect(() => {
    setAnchorBatchSizeLocal(videoAnchorBatchSize(video, config.anchorBatchSize))
  }, [vid, video?.anchor_batch_size, config.anchorBatchSize, video])

  function handleSave() {
    persistConfig()
  }

  // Navigate to start frame and persist to video config on backend
  async function commitStartFrame(v: number) {
    setPropagationStartFrame(v)
    setCurrentFrame(v)
    if (pid && vid) {
      try {
        await updateVideoMeta(pid, vid, { start_frame: v })
        updateVideo({ start_frame: v })
      } catch { /* non-critical */ }
    }
  }

  const anchorLabelingLocked = video
    ? hasAnchorLabelingStarted(video, config.anchorBatchSize)
    : false

  const anchorPreviewCount = video
    ? computeAnchorFrames(propagationStartFrame, video.num_frames, anchorBatchSizeLocal).length
    : null

  async function commitAnchorBatchSize(v: number) {
    const normalized = normalizeAnchorBatchSize(v, config.anchorBatchSize)
    setAnchorBatchSizeLocal(normalized)
    setConfig({ anchorBatchSize: normalized })
    if (!pid || !vid) return
    try {
      await updateVideoMeta(pid, vid, { anchor_batch_size: normalized })
      updateVideo({ anchor_batch_size: normalized })
    } catch (e: unknown) {
      const detail =
        e && typeof e === 'object' && 'response' in e
          ? (e as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : undefined
      addToast(
        typeof detail === 'string' ? detail : 'Could not update anchor frame interval',
        'error',
      )
      setAnchorBatchSizeLocal(videoAnchorBatchSize(video, config.anchorBatchSize))
    }
  }

  async function handleDownsample() {
    if (!pid || !vid) return
    setDsRunning(true)
    setDsMessage(null)
    const ds: DownsampleOptions = dsMode === 'factor' ? { scaleFactor: dsFactor } : { maxDim: dsMaxDim }
    try {
      const result = await downsampleVideo(pid, vid, ds)
      if (result.status === 'skipped') {
        setDsMessage({ text: result.message ?? 'Already within target size.', ok: true })
      } else {
        updateVideo({ width: result.width, height: result.height })
        setDsMessage({ text: `Done — ${result.width}×${result.height}`, ok: true })
      }
    } catch (e: unknown) {
      const detail = (e as any)?.response?.data?.detail
      setDsMessage({ text: detail ?? (e instanceof Error ? e.message : 'Downsample failed'), ok: false })
    } finally {
      setDsRunning(false)
    }
  }

  async function handleDownsampleAll() {
    if (!pid || !project) return
    const videos = Object.values(project.videos)
    if (!videos.length) return
    setDsAllRunning(true)
    setDsAllMessage(null)
    setDsAllProgress({ done: 0, total: videos.length, current: '' })
    const ds: DownsampleOptions = dsMode === 'factor' ? { scaleFactor: dsFactor } : { maxDim: dsMaxDim }
    let skipped = 0, done = 0, failed = 0
    for (let i = 0; i < videos.length; i++) {
      const v = videos[i]
      setDsAllProgress({ done: i, total: videos.length, current: v.name })
      try {
        const result = await downsampleVideo(pid, v.id, ds)
        if (result.status === 'skipped') skipped++
        else done++
      } catch {
        failed++
      }
    }
    setDsAllProgress(null)
    const parts = []
    if (done) parts.push(`${done} downsampled`)
    if (skipped) parts.push(`${skipped} skipped`)
    if (failed) parts.push(`${failed} failed`)
    setDsAllMessage({ text: parts.join(', ') || 'Done', ok: failed === 0 })
    setDsAllRunning(false)
  }

  const maxFrame = video ? video.num_frames - 1 : 999999

  return (
    <div className="flex-1 overflow-y-auto bg-[#0d0d0d] text-[#ccc]">
      <div className="max-w-lg mx-auto p-6 space-y-8">

        {/* Header row with dirty indicator + save button */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {configDirty && (
              <span className="text-xs text-amber-400 font-medium">Unsaved changes</span>
            )}
          </div>
          <button
            onClick={handleSave}
            disabled={!configDirty}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              configDirty
                ? 'bg-blue-600 hover:bg-blue-500 text-white'
                : 'bg-[#1e1e1e] text-[#555] cursor-not-allowed'
            }`}
          >
            <Save size={12} />
            Save
          </button>
        </div>

        {/* Model info */}
        <div>
          <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-3">Model</h2>
          <div className="rounded-xl border border-[#2a2a2a] bg-[#111] px-4 py-3 flex items-center gap-3">
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: modelInfo?.sam_ready ? '#22c55e' : '#555' }} />
            <span className="text-sm text-[#aaa] font-mono">{modelInfo?.model ?? '…'}</span>
            <span className="text-xs text-[#555] ml-auto">{modelInfo?.sam_ready ? 'ready' : 'loading'}</span>
          </div>
        </div>

        {/* Video */}
        {project && (
          <Section title="Video">
            {/* Shared controls row */}
            <div className="py-3 border-b border-[#1e1e1e]">
              <p className="text-xs text-[#555] mb-2">Downsample settings</p>
              <div className="flex items-center gap-2">
                <select
                  value={dsMode}
                  onChange={e => setDsMode(e.target.value as 'max_dim' | 'factor')}
                  disabled={dsRunning || dsAllRunning}
                  className="text-xs text-black font-mono px-1.5 py-1 rounded border border-[#333] bg-white"
                >
                  <option value="max_dim">Max px</option>
                  <option value="factor">Factor</option>
                </select>
                {dsMode === 'max_dim' ? (
                  <>
                    <input
                      type="number"
                      min={64}
                      max={7680}
                      value={dsMaxDim}
                      onChange={e => setDsMaxDim(Math.max(64, Math.min(7680, parseInt(e.target.value) || 1080)))}
                      disabled={dsRunning || dsAllRunning}
                      className="w-20 text-xs text-black font-mono text-center py-1 px-2 rounded border border-[#333]"
                    />
                    <span className="text-xs text-[#555]">px (longest side)</span>
                  </>
                ) : (
                  <select
                    value={dsFactor}
                    onChange={e => setDsFactor(Number(e.target.value))}
                    disabled={dsRunning || dsAllRunning}
                    className="text-xs text-black font-mono px-1.5 py-1 rounded border border-[#333] bg-white"
                  >
                    {[2, 3, 4, 6, 8].map(f => <option key={f} value={f}>{f}×</option>)}
                  </select>
                )}
              </div>
            </div>

            {/* Current video row */}
            {video && (
            <div className="py-3 border-b border-[#1e1e1e]">
              <div className="flex items-start gap-4">
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-[#ddd] font-medium">Downsample current video</p>
                  <p className="text-xs text-[#666] mt-0.5 leading-relaxed">
                    Current: <span className="font-mono text-[#888]">{video.width}×{video.height}</span>.
                    Clears cached frames. Irreversible — do before labeling.
                  </p>
                  {dsMessage && (
                    <p className={`text-xs mt-1.5 ${dsMessage.ok ? 'text-green-400' : 'text-red-400'}`}>
                      {dsMessage.text}
                    </p>
                  )}
                </div>
                <button
                  onClick={handleDownsample}
                  disabled={dsRunning || dsAllRunning || !pid || !vid}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-[#2a2a2a] hover:bg-[#333] text-[#ccc] disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5 flex-shrink-0 mt-0.5"
                >
                  {dsRunning && <span className="inline-block w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
                  {dsRunning ? 'Running…' : 'Apply'}
                </button>
              </div>
            </div>
            )}

            {/* All videos row */}
            <div className="py-3">
              <div className="flex items-start gap-4">
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-[#ddd] font-medium">Downsample all videos</p>
                  <p className="text-xs text-[#666] mt-0.5 leading-relaxed">
                    Apply the settings above to every video in this project sequentially.
                    Videos already within the target are skipped.
                  </p>
                  {dsAllProgress && (
                    <div className="mt-1.5 space-y-1">
                      <p className="text-xs text-[#888] truncate">
                        {dsAllProgress.done}/{dsAllProgress.total} — {dsAllProgress.current}
                      </p>
                      <div className="h-1 bg-[#333] rounded-full overflow-hidden w-full">
                        <div
                          className="h-full bg-blue-500 transition-all duration-300"
                          style={{ width: `${(dsAllProgress.done / dsAllProgress.total) * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                  {dsAllMessage && (
                    <p className={`text-xs mt-1.5 ${dsAllMessage.ok ? 'text-green-400' : 'text-red-400'}`}>
                      {dsAllMessage.text}
                    </p>
                  )}
                </div>
                <button
                  onClick={handleDownsampleAll}
                  disabled={dsRunning || dsAllRunning || !pid}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-[#2a2a2a] hover:bg-[#333] text-[#ccc] disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5 flex-shrink-0 mt-0.5"
                >
                  {dsAllRunning && <span className="inline-block w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
                  {dsAllRunning ? 'Running…' : 'Apply to all'}
                </button>
              </div>
            </div>
          </Section>
        )}

        {/* Annotation */}
        <Section title="Annotation">
          <NumberRow
            label="Start frame"
            description={
              'First frame to annotate and run inference on. Frames before this are hidden from ' +
              'the timeline and excluded from propagation. Press Enter or leave the field to navigate there.'
            }
            value={propagationStartFrame}
            onChange={setPropagationStartFrame}
            onCommit={commitStartFrame}
            min={0}
            max={maxFrame}
          />
        </Section>

        {/* Rendering */}
        <Section title="Rendering">
          <ToggleRow
            label="Show masks"
            description="Display segmentation mask overlays on the annotation canvas."
            value={config.showMasks}
            onChange={v => setConfig({ showMasks: v })}
          />
          <SliderRow
            label="Mask opacity"
            description="Transparency of mask overlays. Lower values let the underlying frame show through."
            value={config.maskOpacity}
            onChange={v => setConfig({ maskOpacity: v })}
            min={0.05}
            max={1.0}
            step={0.05}
            format={v => `${Math.round(v * 100)}%`}
          />
          <SliderRow
            label="Point marker size"
            description="Scale factor for annotation point markers drawn on the canvas."
            value={config.pointSize}
            onChange={v => setConfig({ pointSize: v })}
            min={0.5}
            max={2.5}
            step={0.1}
            format={v => `${v.toFixed(1)}×`}
          />
        </Section>

        {/* Tracking */}
        <Section title="Tracking">
          <NumberRow
            label="Anchor frame interval (frames)"
            description={
              (anchorLabelingLocked
                ? 'Locked after anchor labeling has started on this video. '
                : '') +
              'Spacing between anchor frames to label before tracking (from start frame through end). ' +
              'Also sets whole-video propagation batch size. ' +
              (anchorPreviewCount != null
                ? `Current video: ${anchorPreviewCount} anchor frame${anchorPreviewCount === 1 ? '' : 's'}.`
                : 'Select a video to preview anchor count.')
            }
            value={anchorBatchSizeLocal}
            onChange={setAnchorBatchSizeLocal}
            onCommit={commitAnchorBatchSize}
            min={ANCHOR_BATCH_SIZE_MIN}
            max={ANCHOR_BATCH_SIZE_MAX}
            disabled={anchorLabelingLocked || !video}
          />
          <ToggleRow
            label="Default to all-anchor context mode"
            description={
              'When all anchor frames are labeled and you click "Start Tracking", ' +
              'pre-select the "All anchor frames as context" method. Each batch will ' +
              'load all labeled frames into SAM\'s session for global context, which ' +
              'may improve accuracy when objects change appearance across batches.'
            }
            value={config.useAllAnchors}
            onChange={v => setConfig({ useAllAnchors: v })}
          />
          <ToggleRow
            label="Automatically infer remaining anchor frames"
            description={
              'After you finish the manual anchor prefix, run SAM in the background to fill ' +
              'the rest of the anchor frames. When off, continue labeling anchors manually ' +
              'or use “Infer remaining anchors” in the sidebar.'
            }
            value={config.autoInferAnchorRemainder}
            onChange={v => setConfig({ autoInferAnchorRemainder: v })}
          />
        </Section>

        <Section title="Agent">
          <div className="py-3">
            <p className="text-sm text-[#ddd] font-medium">LLM backend</p>
            <p className="text-xs text-[#666] mt-0.5 leading-relaxed">
              Same pattern as SAM 3 Agent: a vision model served locally over an OpenAI-compatible
              API. The backend auto-detects <span className="font-mono text-[#888]">Ollama</span> on
              :11434 or <span className="font-mono text-[#888]">vLLM</span> on :8001 (dummy API key).
              Optional env: <span className="font-mono text-[#888]">AGENT_LLM_PROVIDER</span>
              (ollama|vllm), <span className="font-mono text-[#888]">AGENT_LLM_MODEL</span>,{' '}
              <span className="font-mono text-[#888]">AGENT_LLM_BASE_URL</span>. Cloud keys are a
              fallback, not required.
            </p>
            <p className="text-xs mt-2 text-[#999] whitespace-pre-wrap">
              {agentInfo == null
                ? 'Checking…'
                : agentInfo.configured
                  ? `Ready · ${agentInfo.local ? 'local' : 'cloud'} · ${agentInfo.provider} · ${agentInfo.model}`
                    + (agentInfo.base_url ? `\n${agentInfo.base_url}` : '')
                  : agentInfo.missing_reason}
            </p>
          </div>
        </Section>

        {/* Single-frame prediction */}
        <Section title="Single-frame prediction">
          <ToggleRow
            label="Seed from previous frame mask"
            description={
              'When running "Predict Frame", if the immediately preceding frame has a saved ' +
              'mask on disk, inject it as a temporary SAM seed before propagating. ' +
              'The seed is removed from the inference state after prediction completes.'
            }
            value={config.usePrevFrameMask}
            onChange={v => setConfig({ usePrevFrameMask: v })}
          />
        </Section>

        <p className="text-xs text-[#444] leading-relaxed px-1">
          Settings are saved to your browser's local storage. Changes take effect immediately but are
          only persisted across refreshes when you click Save.
        </p>

      </div>
    </div>
  )
}
