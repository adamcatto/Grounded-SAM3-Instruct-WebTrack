import React from 'react'
import { Save } from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import { checkHealth, updateVideoMeta, getTrackingParams, updateTrackingParams, type TrackingParams } from '../api/client'
import { useEffect, useState } from 'react'

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

function NumberRow({ label, description, value, onChange, onCommit, min, max }: {
  label: string
  description: string
  value: number
  onChange: (v: number) => void
  onCommit: (v: number) => void
  min: number
  max: number
}) {
  return (
    <div className="flex items-start gap-4 py-3 border-b border-[#1e1e1e] last:border-0">
      <div className="flex-1 min-w-0">
        <p className="text-sm text-[#ddd] font-medium">{label}</p>
        <p className="text-xs text-[#666] mt-0.5 leading-relaxed">{description}</p>
      </div>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={e => {
          const v = Math.max(min, Math.min(max, parseInt(e.target.value) || min))
          onChange(v)
        }}
        onKeyDown={e => { if (e.key === 'Enter') onCommit(value) }}
        onBlur={() => onCommit(value)}
        className="w-20 flex-shrink-0 text-xs py-1 px-2 rounded bg-[#1a1a1a] border border-[#333] text-[#ccc] text-right"
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
  const [trackingParams, setTrackingParams] = useState<TrackingParams | null>(null)
  const [trackingParamsDirty, setTrackingParamsDirty] = useState(false)

  useEffect(() => {
    checkHealth().then(h => setModelInfo({ model: h.sam_model, sam_ready: h.sam_ready })).catch(() => {})
  }, [])

  // Fetch tracking params when video changes
  useEffect(() => {
    if (pid && vid) {
      getTrackingParams(pid, vid)
        .then(params => {
          setTrackingParams(params)
          setTrackingParamsDirty(false)
        })
        .catch(() => {})
    }
  }, [pid, vid])

  function handleSave() {
    persistConfig()
  }

  // Update a single tracking param locally
  function handleTrackingParamChange<K extends keyof TrackingParams>(key: K, value: TrackingParams[K]) {
    if (trackingParams) {
      setTrackingParams({ ...trackingParams, [key]: value })
      setTrackingParamsDirty(true)
    }
  }

  // Save tracking params to backend
  async function saveTrackingParams() {
    if (!pid || !vid || !trackingParams) return
    try {
      const updated = await updateTrackingParams(pid, vid, trackingParams)
      setTrackingParams(updated)
      setTrackingParamsDirty(false)
      addToast?.('Tracking params saved', 'success')
    } catch (e) {
      addToast?.('Failed to save tracking params', 'error')
    }
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

        {/* Track correction */}
        <Section title="Track correction">
          <div className="py-3">
            <p className="text-sm text-[#ddd] font-medium mb-0.5">Correction method</p>
            <p className="text-xs text-[#666] mb-3 leading-relaxed">
              How to fix detected identity swaps. Deterministic swap is instant; re-propagate reruns
              SAM from before the swap onset using the correct seed masks.
            </p>
            <div className="flex gap-2">
              {(['swap', 'repropagate'] as const).map(m => (
                <button
                  key={m}
                  onClick={() => setConfig({ correctionMethod: m })}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                    config.correctionMethod === m
                      ? 'bg-blue-600/20 border-blue-500/60 text-blue-400'
                      : 'border-[#333] text-[#666] hover:border-[#555] hover:text-[#aaa]'
                  }`}
                >
                  {m === 'swap' ? 'Deterministic swap' : 'Re-propagate'}
                </button>
              ))}
            </div>
          </div>
        </Section>

        {/* Tracking Parameters */}
        {trackingParams && (
          <Section title="Tracking parameters">
            <SliderRow
              label="Min IoU threshold"
              description="Minimum overlap (IoU) required between consecutive frames. Lower values allow more mask drift."
              value={trackingParams.min_iou_threshold}
              onChange={v => handleTrackingParamChange('min_iou_threshold', v)}
              min={0.0}
              max={0.5}
              step={0.01}
              format={v => v.toFixed(2)}
            />
            <SliderRow
              label="Max area ratio"
              description="Maximum allowed change in mask area between frames. Higher values tolerate larger size changes."
              value={trackingParams.max_area_ratio}
              onChange={v => handleTrackingParamChange('max_area_ratio', v)}
              min={1.5}
              max={10.0}
              step={0.5}
              format={v => `${v.toFixed(1)}×`}
            />
            <SliderRow
              label="Max centroid jump"
              description="Maximum normalized distance the mask center can move between frames (0-1 = fraction of frame)."
              value={trackingParams.max_centroid_jump}
              onChange={v => handleTrackingParamChange('max_centroid_jump', v)}
              min={0.05}
              max={0.5}
              step={0.01}
              format={v => v.toFixed(2)}
            />
            <NumberRow
              label="Consecutive reject limit"
              description="After this many consecutive rejected frames, accept the mask anyway to avoid getting stuck."
              value={trackingParams.consecutive_reject_limit}
              onChange={v => handleTrackingParamChange('consecutive_reject_limit', v)}
              onCommit={() => {}}
              min={1}
              max={20}
            />
            <div className="py-3">
              <button
                onClick={saveTrackingParams}
                disabled={!trackingParamsDirty}
                className={`w-full py-2 rounded-lg text-xs font-medium transition-colors ${
                  trackingParamsDirty
                    ? 'bg-blue-600 hover:bg-blue-500 text-white'
                    : 'bg-[#1e1e1e] text-[#555] cursor-not-allowed'
                }`}
              >
                {trackingParamsDirty ? 'Save Tracking Params' : 'Tracking Params Saved'}
              </button>
            </div>
          </Section>
        )}

        <p className="text-xs text-[#444] leading-relaxed px-1">
          Settings are saved to your browser's local storage. Changes take effect immediately but are
          only persisted across refreshes when you click Save.
        </p>

      </div>
    </div>
  )
}
