import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Bot, ChevronDown, ChevronRight, Loader, Send, Sparkles, Square, X,
  Eye, MousePointerClick, Type, Film, Boxes, Play, MapPin, RefreshCw, Settings2, Database,
} from 'lucide-react'
import { useStore, currentVideo as selectCurrentVideo } from '../../store/useStore'
import {
  cancelAgentRun, getAgentStatus, getProject, getSavedMask, startAgentRun, type AgentLlmStatus,
  getAgentPromptTemplates, getAgentSystemPrompt, setAgentSystemPrompt, type AgentPromptTemplate,
  saveAgentRlSample,
} from '../../api/client'
import type { MaskData } from '../../types'

const CUSTOM_TEMPLATE: AgentPromptTemplate = {
  id: 'custom',
  name: 'Custom (blank)',
  description: 'Start from scratch. The model’s base instructions and your project metadata are still added automatically.',
  content: '',
}

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
  'Segment the two dark mice.',
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

async function hydrateSavedMasks(pid: string, videoId: string | undefined, frameIdx: number | undefined) {
  if (!videoId || frameIdx == null) return
  try {
    const data = await getSavedMask(pid, videoId, frameIdx)
    const masks = data.masks ?? {}
    if (Object.keys(masks).length === 0) return
    const s = useStore.getState()
    if (s.currentVideoId !== videoId) return
    const live = s.currentFrameMasksFrame === frameIdx ? s.currentFrameMasks : {}
    const merged = { ...(s.savedMaskCache[frameIdx] ?? {}), ...live, ...masks }
    s.setSavedMask(frameIdx, merged)
    const now = useStore.getState()
    if (now.currentVideoId === videoId && now.currentFrame === frameIdx) {
      now.setCurrentFrameMasks(merged, frameIdx)
    }
  } catch { /* ignore */ }
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
      if (frameIdx != null) useStore.getState().setCurrentFrame(frameIdx)
      useStore.getState().setViewerTab('annotate')
      break
    }
    case 'refresh_project':
      if (!pid) break
      try {
        const p = await getProject(pid)
        const s = useStore.getState()
        s.setProject(p)
        await hydrateSavedMasks(pid, s.currentVideoId ?? undefined, s.currentFrame)
      } catch { /* ignore */ }
      break
    case 'set_masks': {
      const frameIdx = asNumber(ev.frame_idx)
      const s = useStore.getState()
      if (videoId && videoId !== s.currentVideoId) s.setCurrentVideo(videoId)
      if (frameIdx == null) break
      const masks = (ev.masks || {}) as MaskData
      const s2 = useStore.getState()
      s2.setCurrentFrame(frameIdx)
      const live = s2.currentFrameMasksFrame === frameIdx ? s2.currentFrameMasks : {}
      const merged = { ...(s2.savedMaskCache[frameIdx] ?? {}), ...live, ...masks }
      s2.setCurrentFrameMasks(merged, frameIdx)
      s2.setSavedMask(frameIdx, merged)
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
      const local = { ...useStore.getState().localAnnotations }
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
  const [streaming, setStreaming] = useState<{ text: string } | null>(null)
  const [usage, setUsage] = useState<{ total: number; max: number | null } | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const scrollerRef = useRef<HTMLDivElement>(null)
  const historyRef = useRef<{ role: 'user' | 'assistant'; content: string }[]>([])

  // System-prompt gate: the user must pick/confirm a prompt before chatting.
  const [promptSet, setPromptSet] = useState<boolean | null>(null) // null = loading
  const [promptContext, setPromptContext] = useState('')
  const [templates, setTemplates] = useState<AgentPromptTemplate[]>([])
  const [showPromptEditor, setShowPromptEditor] = useState(false)

  useEffect(() => {
    getAgentStatus(pid || undefined).then(setStatus).catch(() => {})
  }, [pid])

  useEffect(() => {
    getAgentPromptTemplates().then(r => setTemplates(r.templates)).catch(() => {})
  }, [])

  useEffect(() => {
    if (!pid) { setPromptSet(null); return }
    setPromptSet(null)
    getAgentSystemPrompt(pid)
      .then(r => { setPromptSet(r.is_set); setPromptContext(r.context) })
      .catch(() => { setPromptSet(false); setPromptContext('') })
  }, [pid])

  const savePrompt = useCallback(async (text: string) => {
    if (!pid) return
    const r = await setAgentSystemPrompt(pid, text)
    setPromptContext(r.context)
    setPromptSet(true)
    setShowPromptEditor(false)
  }, [pid])

  const [savingRl, setSavingRl] = useState(false)
  const saveRlSample = useCallback(async () => {
    if (!pid || savingRl) return
    setSavingRl(true)
    try {
      const r = await saveAgentRlSample(pid)
      setItems(prev => [...prev, {
        id: `rl-${Date.now()}`,
        kind: 'status',
        text: `Saved RL sample #${r.id} (${r.count} total, ${r.images} image${r.images === 1 ? '' : 's'}) → ${r.db_path}`,
      }])
    } catch (e) {
      setItems(prev => [...prev, {
        id: `rle-${Date.now()}`,
        kind: 'error',
        text: e instanceof Error ? e.message : 'Failed to save RL sample',
      }])
    } finally {
      setSavingRl(false)
    }
  }, [pid, savingRl])

  useEffect(() => {
    const el = scrollerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [items, running, streaming])

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
    setStreaming(null)
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
    let uiTail = Promise.resolve()
    const queueUi = (ev: Record<string, unknown>) => {
      uiTail = uiTail.then(() => applyUiEvent(ev)).catch(() => {})
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
          if (event === 'token') {
            const t = String(data.text || '')
            if (t) setStreaming(prev => ({ text: (prev?.text || '') + t }))
            return
          }
          if (event === 'usage') {
            const total = Number(data.total_tokens ?? 0)
            const max = data.max_context == null ? null : Number(data.max_context)
            setStreaming(null)
            if (total > 0) setUsage({ total, max })
            return
          }
          if (event === 'reasoning' || event === 'message' || event === 'tool_call' || event === 'done' || event === 'error') {
            setStreaming(null)
          }
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
            queueUi(data)
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
            const s = useStore.getState()
            uiTail = uiTail.then(() => hydrateSavedMasks(pid, s.currentVideoId ?? undefined, s.currentFrame))
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
      try { await uiTail } catch { /* ignore */ }
      const s = useStore.getState()
      await hydrateSavedMasks(pid, s.currentVideoId ?? undefined, s.currentFrame)
      const joined = assistantParts.join('\n').trim()
      if (joined) historyRef.current = [...historyRef.current, { role: 'assistant', content: joined }]
      abortRef.current = null
      setStreaming(null)
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
    <div className="h-full flex flex-col bg-[#111111] border-l border-[#2a2a2a] [&_svg]:w-[1.15em] [&_svg]:h-[1.15em]">
      <div className="flex items-center gap-[0.55em] px-[0.9em] py-[0.6em] border-b border-[#2a2a2a] shrink-0">
        <Bot size={16} className="text-violet-400" />
        <div className="flex-1 min-w-0">
          <div className="text-[1em] font-semibold text-white leading-none">Agent</div>
          <div className="text-[0.8em] text-[#666] truncate mt-0.5" title={videoHint}>{videoHint}</div>
        </div>
        {running && <Loader size={14} className="text-violet-400 animate-spin" />}
        {pid && items.length > 0 && !running && (
          <button
            onClick={() => void saveRlSample()}
            disabled={savingRl}
            className="btn btn-ghost p-[0.3em] disabled:opacity-40"
            title="Save this run to the RL dataset (trace + state + images)"
          >
            <Database size={16} />
          </button>
        )}
        {pid && promptSet && !showPromptEditor && (
          <button
            onClick={() => setShowPromptEditor(true)}
            disabled={running}
            className="btn btn-ghost p-[0.3em] disabled:opacity-40"
            title="Edit system prompt"
          >
            <Settings2 size={16} />
          </button>
        )}
        <button onClick={() => setAgentPaneOpen(false)} className="btn btn-ghost p-[0.3em]" title="Close agent">
          <X size={16} />
        </button>
      </div>

      {status && !status.configured && (
        <div className="px-[0.9em] py-[0.5em] text-[1em] leading-snug text-amber-300/90 bg-amber-500/10 border-b border-amber-500/20">
          {status.missing_reason || 'No local Ollama/vLLM server detected. Start one, then reopen Agent.'}
        </div>
      )}
      {status?.configured && (
        <div className="px-[0.9em] py-[0.4em] text-[0.8em] text-[#555] border-b border-[#1e1e1e] truncate" title={status.base_url ?? ''}>
          {status.local ? 'local' : 'cloud'} · {status.provider} · {status.model}
        </div>
      )}

      {pid && status?.configured && (promptSet === false || showPromptEditor) ? (
        <PromptGate
          templates={templates}
          initialContext={promptContext}
          editing={promptSet === true}
          onSave={savePrompt}
          onCancel={promptSet ? () => setShowPromptEditor(false) : undefined}
        />
      ) : pid && status?.configured && promptSet === null ? (
        <div className="flex-1 flex items-center justify-center text-[#555]">
          <Loader size={16} className="animate-spin" />
        </div>
      ) : (
        <>
          <div ref={scrollerRef} className="flex-1 overflow-y-auto px-[0.9em] py-[0.7em] space-y-[0.55em] min-h-0">
            {items.length === 0 && (
              <div className="space-y-[0.85em]">
                <p className="text-[1em] text-[#777] leading-relaxed">
                  Describe what to segment on this frame or across the video. The agent can inspect frames,
                  run SAM3 text prompts, click points, skip unclear frames, and start tracking — live on the canvas.
                </p>
                {EXAMPLES.map(ex => (
                  <button
                    key={ex}
                    type="button"
                    onClick={() => void send(ex)}
                    disabled={!pid || running}
                    className="w-full text-left text-[1em] leading-relaxed text-[#bbb] bg-[#1a1a1a] hover:bg-[#222] border border-[#2a2a2a] rounded-[0.5em] px-[0.7em] py-[0.5em] disabled:opacity-40"
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
            {streaming && (
              <div className="text-[1em] italic text-[#9a9a9a] border-l-2 border-violet-600/60 pl-[0.5em] py-[0.15em] whitespace-pre-wrap">
                {streaming.text}
                <span className="inline-block w-[0.35em] h-[0.9em] ml-[0.15em] align-middle bg-violet-400 animate-pulse" />
              </div>
            )}
          </div>

          <div className="shrink-0 border-t border-[#2a2a2a] p-[0.5em]">
            <ContextMeter usage={usage} />
            <div className="flex items-end gap-[0.55em]">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={pid ? 'Prompt the agent…' : 'Open a project first'}
                disabled={!pid}
                rows={3}
                className="flex-1 resize-none bg-[#1a1a1a] border border-[#333] rounded-[0.5em] px-[0.7em] py-[0.5em] text-[1.15em] text-[#f0f0f0] outline-none focus:border-violet-500 disabled:opacity-40"
              />
              {running ? (
                <button type="button" onClick={() => void stop()} className="btn btn-secondary p-[0.5em]" title="Stop">
                  <Square size={14} />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => void send()}
                  disabled={!pid || !input.trim()}
                  className="btn btn-primary p-[0.5em]"
                  title="Send"
                >
                  <Send size={14} />
                </button>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function ContextMeter({ usage }: { usage: { total: number; max: number | null } | null }) {
  if (!usage) return null
  const { total, max } = usage
  const pct = max ? Math.min(100, (total / max) * 100) : null
  const barColor = pct == null ? 'bg-violet-500'
    : pct > 90 ? 'bg-red-500'
    : pct > 75 ? 'bg-amber-500'
    : 'bg-violet-500'
  return (
    <div className="px-[0.3em] pb-2" title="Approximate context window usage from the last agent turn">
      <div className="flex justify-between text-[0.8em] text-[#666] mb-1">
        <span>Context{pct == null ? '' : ` · ${Math.round(pct)}%`}</span>
        <span>
          {total.toLocaleString()}{max ? ` / ${max.toLocaleString()}` : ''} tok
        </span>
      </div>
      <div className="h-1 rounded-full bg-[#222] overflow-hidden">
        <div className={`h-full ${barColor} transition-[width] duration-300`} style={{ width: `${pct ?? 100}%` }} />
      </div>
    </div>
  )
}

function PromptGate({
  templates,
  initialContext,
  editing,
  onSave,
  onCancel,
}: {
  templates: AgentPromptTemplate[]
  initialContext: string
  editing: boolean
  onSave: (text: string) => Promise<void>
  onCancel?: () => void
}) {
  const options = useMemo(() => [...templates, CUSTOM_TEMPLATE], [templates])
  // When editing an already-set prompt, start from its text; otherwise nothing picked yet.
  const [selectedId, setSelectedId] = useState<string | null>(editing ? 'custom' : null)
  const [text, setText] = useState(editing ? initialContext : '')
  const [saving, setSaving] = useState(false)

  const pick = (tpl: AgentPromptTemplate) => {
    setSelectedId(tpl.id)
    setText(tpl.content)
  }

  const save = async () => {
    setSaving(true)
    try { await onSave(text) } finally { setSaving(false) }
  }

  return (
    <div className="flex-1 overflow-y-auto min-h-0 px-[0.9em] py-[0.7em] space-y-[0.85em]">
      <div>
        <div className="text-[1em] font-semibold text-white">
          {editing ? 'Edit the agent’s system prompt' : 'Choose the agent’s system prompt'}
        </div>
        <p className="text-[1em] text-[#888] leading-relaxed mt-1">
          Pick a starting point and edit it for this project. The model’s base instructions and your
          project’s video metadata (names, frame counts, fps…) are added automatically and stay hidden.
        </p>
      </div>

      <div className="space-y-[0.4em]">
        {options.map(tpl => (
          <button
            key={tpl.id}
            type="button"
            onClick={() => pick(tpl)}
            className={`w-full text-left rounded-[0.5em] px-[0.7em] py-[0.5em] border transition-colors ${
              selectedId === tpl.id
                ? 'bg-violet-600/20 border-violet-500/60'
                : 'bg-[#1a1a1a] border-[#2a2a2a] hover:bg-[#222]'
            }`}
          >
            <div className="text-[1em] font-medium text-[#ddd]">{tpl.name}</div>
            <div className="text-[0.8em] text-[#888] leading-snug mt-0.5">{tpl.description}</div>
          </button>
        ))}
      </div>

      {selectedId !== null && (
        <div className="space-y-[0.55em]">
          <div className="text-[0.8em] uppercase tracking-wide text-[#666]">System prompt</div>
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            rows={12}
            placeholder="Describe what the agent should segment/track and any rules it should follow…"
            className="w-full resize-y bg-[#1a1a1a] border border-[#333] rounded-[0.5em] px-[0.7em] py-[0.5em] text-[1em] leading-relaxed text-[#f0f0f0] outline-none focus:border-violet-500 font-mono"
          />
          <div className="flex items-center gap-[0.55em]">
            <button
              type="button"
              onClick={() => void save()}
              disabled={saving}
              className="btn btn-primary px-[0.9em] py-[0.4em] text-[1em] disabled:opacity-40"
            >
              {saving ? 'Saving…' : editing ? 'Save prompt' : 'Save & start'}
            </button>
            {onCancel && (
              <button type="button" onClick={onCancel} disabled={saving} className="btn btn-ghost px-[0.9em] py-[0.4em] text-[1em]">
                Cancel
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function TraceRow({ item, onToggle }: { item: TraceItem; onToggle: () => void }) {
  if (item.kind === 'user') {
    return (
      <div className="ml-[1.4em] rounded-[0.5em] bg-blue-600/20 border border-blue-600/30 px-[0.7em] py-[0.5em] text-[1em] text-[#ddd] whitespace-pre-wrap">
        {item.text}
      </div>
    )
  }
  if (item.kind === 'assistant') {
    return (
      <div className="mr-[1em] rounded-[0.5em] bg-[#1a1a1a] border border-[#2a2a2a] px-[0.7em] py-[0.5em] text-[1em] text-[#ccc] whitespace-pre-wrap">
        {item.text}
      </div>
    )
  }
  if (item.kind === 'reasoning') {
    return (
      <div className="text-[1em] italic text-[#888] border-l-2 border-violet-700/60 pl-[0.5em] py-[0.15em] whitespace-pre-wrap">
        {item.text}
      </div>
    )
  }
  if (item.kind === 'status') {
    return <div className="text-[0.8em] text-[#555]">{item.text}</div>
  }
  if (item.kind === 'error') {
    return (
      <div className="rounded-[0.5em] bg-red-500/10 border border-red-500/30 px-[0.7em] py-[0.5em] text-[1em] text-red-300">
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
        className="w-full flex items-center gap-[0.4em] px-[0.55em] py-[0.4em] text-[1em] text-[#bbb] hover:bg-[#1c1c1c]"
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
        <span className={item.ok === false ? 'text-amber-400' : 'text-violet-300'}>{toolIcon(item.name)}</span>
        <span className="flex-1 text-left truncate">{toolLabel(item.name, item.arguments)}</span>
        {item.elapsedMs != null && <span className="text-[0.8em] text-[#555]">{item.elapsedMs}ms</span>}
      </button>
      {!collapsed && (
        <pre className="px-[0.55em] pb-2 text-[0.8em] text-[#888] overflow-x-auto whitespace-pre-wrap break-all">
          {JSON.stringify({ arguments: item.arguments, result: item.result }, null, 2)}
        </pre>
      )}
    </div>
  )
}
