import React, { useEffect, useState } from 'react'
import { checkHealth } from '../api/client'

interface Props {
  onReady: () => void
}

export default function LoadingScreen({ onReady }: Props) {
  const [status, setStatus] = useState<'connecting' | 'loading' | 'error'>('connecting')
  const [modelName, setModelName] = useState('')
  const [errorMsg, setErrorMsg] = useState('')

  useEffect(() => {
    let cancelled = false
    let attempts = 0

    async function poll() {
      while (!cancelled) {
        try {
          const health = await checkHealth()
          if (cancelled) return

          if (health.sam_loaded) {
            setModelName(health.sam_model)
            setStatus('loading') // brief flash before ready
            // Small delay so user sees the "ready" state
            setTimeout(() => {
              if (!cancelled) onReady()
            }, 400)
            return
          }

          if (health.sam_load_error) {
            setStatus('error')
            setErrorMsg(health.sam_load_error)
            return
          }

          if (health.sam_loading) {
            setStatus('loading')
            setModelName(health.sam_model)
          } else {
            setStatus('connecting')
          }
        } catch {
          // Backend not reachable yet
          setStatus('connecting')
          attempts++
        }

        // Poll every 2 seconds
        await new Promise(r => setTimeout(r, 2000))
      }
    }

    poll()
    return () => { cancelled = true }
  }, [onReady])

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-[#0a0a0a]">
      <div className="flex flex-col items-center gap-6 max-w-md text-center px-6">
        {/* Logo / title */}
        <div className="flex flex-col items-center gap-2">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="23 7 16 12 23 17 23 7" />
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight mt-2">
            SAM3 Web Tracker
          </h1>
        </div>

        {/* Status indicator */}
        {status === 'error' ? (
          <div className="space-y-3">
            <div className="flex items-center justify-center gap-2 text-red-400">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              <span className="text-sm font-medium">Failed to load model</span>
            </div>
            <p className="text-xs text-[#666] leading-relaxed">{errorMsg}</p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Spinner */}
            <div className="flex justify-center">
              <div className="w-8 h-8 border-2 border-[#333] border-t-blue-500 rounded-full animate-spin" />
            </div>

            <div className="space-y-1">
              <p className="text-sm text-[#ccc] font-medium">
                {status === 'connecting'
                  ? 'Connecting to backend...'
                  : `Loading ${modelName.toUpperCase()} model...`}
              </p>
              <p className="text-xs text-[#555]">
                {status === 'connecting'
                  ? 'Waiting for the server to start'
                  : 'Loading model weights into GPU memory. This may take a moment.'}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
