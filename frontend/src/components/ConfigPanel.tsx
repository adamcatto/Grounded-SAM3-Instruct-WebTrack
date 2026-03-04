import React from 'react'
import { useStore } from '../store/useStore'
import { checkHealth } from '../api/client'
import { useEffect, useState } from 'react'

// ─── Toggle row ───────────────────────────────────────────────────────────────

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

function ConfigRow({ label, description, value, onChange }: {
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

// ─── Main panel ───────────────────────────────────────────────────────────────

export default function ConfigPanel() {
  const { config, setConfig } = useStore()
  const [modelInfo, setModelInfo] = useState<{ model: string; sam_ready: boolean } | null>(null)

  useEffect(() => {
    checkHealth().then(h => setModelInfo({ model: h.sam_model, sam_ready: h.sam_ready })).catch(() => {})
  }, [])

  return (
    <div className="flex-1 overflow-y-auto bg-[#0d0d0d] text-[#ccc]">
      <div className="max-w-lg mx-auto p-6 space-y-8">

        {/* Model info */}
        <div>
          <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-3">Model</h2>
          <div className="rounded-xl border border-[#2a2a2a] bg-[#111] px-4 py-3 flex items-center gap-3">
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: modelInfo?.sam_ready ? '#22c55e' : '#555' }} />
            <span className="text-sm text-[#aaa] font-mono">{modelInfo?.model ?? '…'}</span>
            <span className="text-xs text-[#555] ml-auto">{modelInfo?.sam_ready ? 'ready' : 'loading'}</span>
          </div>
        </div>

        {/* Prediction settings */}
        <div>
          <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-3">Single-frame prediction</h2>
          <div className="rounded-xl border border-[#2a2a2a] bg-[#111] px-4 divide-y divide-[#1e1e1e]">
            <ConfigRow
              label="Seed from previous frame mask"
              description={
                'When running "Predict Frame", if the immediately preceding frame has a saved ' +
                'mask on disk, inject it as a temporary SAM seed before propagating. ' +
                'The seed is removed from the inference state after prediction completes.'
              }
              value={config.usePrevFrameMask}
              onChange={v => setConfig({ usePrevFrameMask: v })}
            />
          </div>
          <p className="text-xs text-[#444] mt-2 leading-relaxed px-1">
            Settings are saved locally in your browser.
          </p>
        </div>

      </div>
    </div>
  )
}
