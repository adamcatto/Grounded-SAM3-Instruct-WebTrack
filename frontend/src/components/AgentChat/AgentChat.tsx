import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Bot, ChevronDown, ChevronRight, Loader, Send, Sparkles, Square, X,
  Eye, MousePointerClick, Type, Film, Boxes, Play, MapPin, RefreshCw,
} from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import {
  cancelAgentRun, getAgentStatus, getProject, startAgentRun, type AgentLlmStatus,
} from '../../api/client'
import type { MaskData } from '../../types'

interface TraceItem {
  id: string
  kind: 'user' | 'assistant' | 'reasoning' | 'tool' | 'status' | 'error'
  text?: string
  name?: string
  arguments?: Record<string, unknown>
  result?: Record<string, unknown>
  ok?: boolean
  elapsedMs?: number
  collapsed?: boolean
}

const EXAMPLES = [
  'Exactly two mice: one shaved (HeadShave, object 1, far right by the water port) and one unshaved (NoShave, object 2, left/center). Inspect, then text_segment each. Do not finish until both have their own mask.',
  'Every 1000th frame, segment each mouse with those names. If a frame is unclear, try about 20 frames away.',
  'Plan anchors for the current video, inspect the first one, and tell me what you see before segmenting.',
]

function toolIcon(name?: string) {
  switch (name) {
    case 'inspect_frame':
    case 'goto_frame':
      return <Eye size={12} />
    case 'text_segment':
      return <Type size={12} />
    case 'add_point_prompt':
      return <MousePointerClick size={12} />
    case 'select_video':
    case 'get_video_details':
    case 'get_project_overview':
      return <Film size={12} />
    case 'create_object':
    case 'update_object':
      return <Boxes size={12} />
    case 'start_propagation':
      return <Play size={12} />
    case 'plan_frames':
      return <MapPin size={12} />
    case 'think':
      return <Sparkles size={12} />
    default:
      return <RefreshCw size={12} />
  }
}

function toolLabel(name?: string, args?: Record<string, unknown>) {
  const a = args || {}
  switch (name) {
    case 'think':
      return 'Reasoning'
    case 'goto_frame':
      return `Load frame ${a.frame_idx ?? ''}`
    case 'inspect_frame':
      return `Inspect frame ${a.frame_idx ?? ''}`.trim()
    case 'text_segment':
      return `Text segment: ${String(a.text || a.object_name || '').slice(0, 48)}`
    case 'add_point_prompt':
      return `Point prompt on obj ${a.object_id ?? ''}`.trim()
    case 'create_object':
      return `Create object ${a.name ?? ''}`
    case 'select_video':
      return 'Select video'
    case 'start_propagation':
      return 'Start tracking / propagation'
    case 'plan_frames':
      return a.around_frame != null ? `Nearby frames around ${a.around_frame}` : `Plan every ${a.interval ?? 'N'}th frame`
    case 'evaluate_segmentation':
      return `Evaluate masks on frame ${a.frame_idx ?? ''}`
    case 'commit_anchor':
      return `Commit anchor frame ${a.frame_idx ?? ''}`
    default:
      return name?.replace(/_/g, ' ') || 'tool'
  }
}

function asString(v: unknown): string | undefined {
  return typeof v === 'string' && v ? v : undefined
}
function asNumber(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}
function asBool(v: unknown): boolean | undefined {
  return typeof v === 'boolean' ? v : undefined
}

async function applyUiEvent(ev: Record<string, unknown>) {
  const store = useStore.getState()
  const pid = store.project?.id
  const action = asString(ev.action)
  const videoId = asString(ev.video_id)
  switch (action) {
    case 'select_video':
      if (videoId) store.setCurrentVideo(videoId)
      store.setViewerTab('annotate')
      break
    case 'goto_frame': {
      const frameIdx = asNumber(ev.frame_idx)
      if (videoId && videoId !== store.currentVideoId) store.setCurrentVideo(videoId)
      if (frameIdx != null) store.setCurrentFrame(frameIdx)
      store.setViewerTab('annotate')
      break
    }
    case 'refresh_project':
      if (!pid) break
      try {
        const p = await getProject(pid)
        store.setProject(p)
      } catch { /* ignore */ }
      break
    case 'set_masks': {
      const frameIdx = asNumber(ev.frame_idx)
      if (videoId && videoId !== store.currentVideoId) break
      if (frameIdx == null) break
      const masks = (ev.masks || {}) as MaskData
      store.setCurrentFrame(frameIdx)
      store.setCurrentFrameMasks(masks, frameIdx)
      store.setSavedMask(frameIdx, masks)
      break
    }
    case 'select_object': {
      const oid = asString(ev.object_id)
      if (oid) store.setCurrentObject(oid)
      break
    }
    case 'set_points': {
      const frameIdx = asNumber(ev.frame_idx)
      const oid = asString(ev.object_id)
      if (frameIdx == null || !oid) break
      const rawPts = Array.isArray(ev.points) ? ev.points : []
      const rawLabs = Array.isArray(ev.labels) ? ev.labels : []
      const pts = rawPts.map((pt, i) => {
        const pair = Array.isArray(pt) ? pt : [0, 0]
        return { x: Number(pair[0]), y: Number(pair[1]), label: (Number(rawLabs[i] ?? 1) ? 1 : 0) as 0 | 1 }
      })
      const local = { ...store.localAnnotations }
      local[oid] = { ...(local[oid] ?? {}), [String(frameIdx)]: { points: pts } }
      useStore.setState({ localAnnotations: local })
      break
    }
    case 'start_propagation':
      if (videoId && videoId !== store.currentVideoId) store.setCurrentVideo(videoId)
      store.requestAgentPropagation({
        video_id: videoId,
        start_frame: asNumber(ev.start_frame),
        end_frame: asNumber(ev.end_frame),
        use_all_anchors: asBool(ev.use_all_anchors),
      })
      break
    case 'set_anchor_phase': {
      const frames = Array.isArray(ev.frames) ? ev.frames.filter((n): n is number => typeof n === 'number') : []
      if (!frames.length) break
      store.setAnchorFrames(frames)
      store.setAnchorPhase(true)
      store.setCurrentAnchorIndex(asNumber(ev.current_index) ?? 0)
      const committed = asNumber(ev.committed_frame)
      if (committed != null) {
        const idx = frames.indexOf(committed)
        if (idx >= 0) store.addAnnotatedAnchor(idx)
      }
      break
    }
    default:
      break
  }
}

export default function AgentChat() {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project, currentVideoId, currentFrame, setAgentPaneOpen,
  } = store
  const pid = project?.id ?? ''

  const [status, setStatus] = useState<AgentLlmStatus | null>(null)
  const [input, setInput] = useState('')
  const [running, setRunning] = useState(false)
  const [items, setItems] = useState<TraceItem[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const scrollerRef = useRef<HTMLDivElement>(null)
  const historyRef = useRef<{ role: 'user' | 'assistant'; content: string }[]>([])

  useEffect(() => {
    getAgentStatus(pid || undefined).then(setStatus).catch(() => {})
  }, [pid])

  useEffect(() => {
    const el = scrollerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [items, running])

  const videoHint = useMemo(() => {
    if (!project) return 'Open a project first.'
    const n = Object.keys(project.videos).length
    const name = video?.name || 'no video selected'
    return `${project.name} · ${n} video${n === 1 ? '' : 's'} · ${name} · frame ${currentFrame}`
  }, [project, video, currentFrame])

  const stop = useCallback(async () => {
    abortRef.current?.abort()
    abortRef.current = null
    if (pid) {
      try { await cancelAgentRun(pid) } catch { /* ignore */ }
    }
    setRunning(false)
  }, [pid])

  const send = useCallback(async (text?: string) => {
    const message = (text ?? input).trim()
    if (!message || !pid || running) return
    setInput('')
    const userItem: TraceItem = { id: `u-${Date.now()}`, kind: 'user', text: message }
    setItems(prev => [...prev, userItem])
    historyRef.current = [...historyRef.current, { role: 'user', content: message }]
    setRunning(true)
    const ac = new AbortController()
    abortRef.current = ac
    let assistantParts: string[] = []
    const upsertTool = (id: string, patch: Partial<TraceItem>) => {
      setItems(prev => {
        const idx = prev.findIndex(it => it.id === id)
        if (idx < 0) return [...prev, { id, kind: 'tool', ...patch }]
        const next = [...prev]
        next[idx] = { ...next[idx], ...patch }
        return next
      })
    }
    try {
      await startAgentRun(
        pid,
        {
          message,
          video_id: currentVideoId,
          frame_idx: currentFrame,
          history: historyRef.current.slice(0, -1),
        },
        (event, data) => {
          if (event === 'heartbeat') return
          if (event === 'status') {
            const phase = String(data.phase || '')
            if (phase === 'started' || phase === 'thinking') {
              setItems(prev => {
                const last = prev[prev.length - 1]
                if (last?.kind === 'status') {
                  const next = [...prev]
                  next[next.length - 1] = {
                    ...last,
                    text: phase === 'thinking'
                      ? `Thinking… step ${data.step ?? ''}/${data.max_steps ?? ''}`
                      : `Agent started (${data.model || 'llm'})`,
                  }
                  return next
                }
                return [...prev, {
                  id: `s-${Date.now()}`,
                  kind: 'status',
                  text: phase === 'thinking' ? `Thinking… step ${data.step ?? ''}` : 'Agent started',
                }]
              })
            }
            return
          }
          if (event === 'reasoning') {
            const textVal = String(data.text || '').trim()
            if (!textVal) return
            setItems(prev => [...prev, { id: `r-${Date.now()}-${Math.random()}`, kind: 'reasoning', text: textVal }])
            return
          }
          if (event === 'message') {
            const textVal = String(data.text || '').trim()
            if (!textVal) return
            assistantParts.push(textVal)
            setItems(prev => [...prev, { id: `a-${Date.now()}-${Math.random()}`, kind: 'assistant', text: textVal }])
            return
          }
          if (event === 'tool_call') {
            const id = String(data.id || `t-${Date.now()}`)
            upsertTool(id, {
              name: String(data.name || 'tool'),
              arguments: (data.arguments || {}) as Record<string, unknown>,
              collapsed: false,
            })
            return
          }
          if (event === 'tool_result') {
            const id = String(data.id || `t-${Date.now()}`)
            upsertTool(id, {
              name: String(data.name || 'tool'),
              ok: Boolean(data.ok),
              result: (data.result || {}) as Record<string, unknown>,
              elapsedMs: typeof data.elapsed_ms === 'number' ? data.elapsed_ms : undefined,
              collapsed: true,
            })
            return
          }
          if (event === 'ui') {
            void applyUiEvent(data)
            return
          }
          if (event === 'error') {
            setItems(prev => [...prev, {
              id: `e-${Date.now()}`,
              kind: 'error',
              text: String(data.message || 'Agent error'),
            }])
            return
          }
          if (event === 'done') {
            const textVal = String(data.text || '').trim()
            if (textVal && !assistantParts.includes(textVal)) {
              assistantParts.push(textVal)
              setItems(prev => [...prev, { id: `d-${Date.now()}`, kind: 'assistant', text: textVal }])
            }
          }
        },
        ac.signal,
      )
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        setItems(prev => [...prev, {
          id: `e-${Date.now()}`,
          kind: 'error',
          text: e instanceof Error ? e.message : 'Agent failed',
        }])
      }
    } finally {
      const joined = assistantParts.join('\n').trim()
      if (joined) historyRef.current = [...historyRef.current, { role: 'assistant', content: joined }]
      abortRef.current = null
      setRunning(false)
    }
  }, [input, pid, running, currentVideoId, currentFrame])

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send()
    }
  }

  return (
    <div className="h-full flex flex-col bg-[#111111] border-l border-[#2a2a2a]">
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-[#2a2a2a] shrink-0">
        <Bot size={16} className="text-violet-400" />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-white leading-none">Agent</div>
          <div className="text-[10px] text-[#666] truncate mt-0.5" title={videoHint}>{videoHint}</div>
        </div>
        {running && <Loader size={14} className="text-violet-400 animate-spin" />}
        <button onClick={() => setAgentPaneOpen(false)} className="btn btn-ghost p-1" title="Close agent">
          <X size={16} />
        </button>
      </div>

      {status && !status.configured && (
        <div className="px-3 py-2 text-[11px] leading-snug text-amber-300/90 bg-amber-500/10 border-b border-amber-500/20">
          {status.missing_reason || 'No local Ollama/vLLM server detected. Start one, then reopen Agent.'}
        </div>
      )}
      {status?.configured && (
        <div className="px-3 py-1.5 text-[10px] text-[#555] border-b border-[#1e1e1e] truncate" title={status.base_url ?? ''}>
          {status.local ? 'local' : 'cloud'} · {status.provider} · {status.model}
        </div>
      )}

      <div ref={scrollerRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-2 min-h-0">
        {items.length === 0 && (
          <div className="space-y-3">
            <p className="text-xs text-[#777] leading-relaxed">
              Describe what to segment on this frame or across the video. The agent can inspect frames,
              run SAM3 text prompts, click points, skip unclear frames, and start tracking — live on the canvas.
            </p>
            {EXAMPLES.map(ex => (
              <button
                key={ex}
                type="button"
                onClick={() => void send(ex)}
                disabled={!pid || running}
                className="w-full text-left text-[11px] leading-relaxed text-[#bbb] bg-[#1a1a1a] hover:bg-[#222] border border-[#2a2a2a] rounded-lg px-2.5 py-2 disabled:opacity-40"
              >
                {ex}
              </button>
            ))}
          </div>
        )}
        {items.map(item => (
          <TraceRow
            key={item.id}
            item={item}
            onToggle={() => setItems(prev => prev.map(it => it.id === item.id ? { ...it, collapsed: !it.collapsed } : it))}
          />
        ))}
      </div>

      <div className="shrink-0 border-t border-[#2a2a2a] p-2">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={pid ? 'Prompt the agent…' : 'Open a project first'}
            disabled={!pid}
            rows={3}
            className="flex-1 resize-none bg-[#1a1a1a] border border-[#333] rounded-lg px-2.5 py-2 text-sm text-[#f0f0f0] outline-none focus:border-violet-500 disabled:opacity-40"
          />
          {running ? (
            <button type="button" onClick={() => void stop()} className="btn btn-secondary p-2" title="Stop">
              <Square size={14} />
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void send()}
              disabled={!pid || !input.trim()}
              className="btn btn-primary p-2"
              title="Send"
            >
              <Send size={14} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function TraceRow({ item, onToggle }: { item: TraceItem; onToggle: () => void }) {
  if (item.kind === 'user') {
    return (
      <div className="ml-6 rounded-lg bg-blue-600/20 border border-blue-600/30 px-2.5 py-2 text-xs text-[#ddd] whitespace-pre-wrap">
        {item.text}
      </div>
    )
  }
  if (item.kind === 'assistant') {
    return (
      <div className="mr-4 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a] px-2.5 py-2 text-xs text-[#ccc] whitespace-pre-wrap">
        {item.text}
      </div>
    )
  }
  if (item.kind === 'reasoning') {
    return (
      <div className="text-[11px] italic text-[#888] border-l-2 border-violet-700/60 pl-2 py-0.5 whitespace-pre-wrap">
        {item.text}
      </div>
    )
  }
  if (item.kind === 'status') {
    return <div className="text-[10px] text-[#555]">{item.text}</div>
  }
  if (item.kind === 'error') {
    return (
      <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-2.5 py-2 text-xs text-red-300">
        {item.text}
      </div>
    )
  }
  const collapsed = item.collapsed !== false
  return (
    <div className="rounded-md border border-[#2a2a2a] bg-[#161616] overflow-hidden">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center gap-1.5 px-2 py-1.5 text-[11px] text-[#bbb] hover:bg-[#1c1c1c]"
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
        <span className={item.ok === false ? 'text-amber-400' : 'text-violet-300'}>{toolIcon(item.name)}</span>
        <span className="flex-1 text-left truncate">{toolLabel(item.name, item.arguments)}</span>
        {item.elapsedMs != null && <span className="text-[10px] text-[#555]">{item.elapsedMs}ms</span>}
      </button>
      {!collapsed && (
        <pre className="px-2 pb-2 text-[10px] text-[#888] overflow-x-auto whitespace-pre-wrap break-all">
          {JSON.stringify({ arguments: item.arguments, result: item.result }, null, 2)}
        </pre>
      )}
    </div>
  )
}
