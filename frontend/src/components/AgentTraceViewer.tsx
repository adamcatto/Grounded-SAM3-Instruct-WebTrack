import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Bot, ChevronDown, ChevronRight, Code2, Database, Eye, Loader, MessageSquareText, RefreshCw, Search, Sparkles, Wrench } from 'lucide-react'
import { displayMaskUrl, frameUrl, getAgentTrace, getAgentTraces, type AgentTrace, type AgentTraceSummary } from '../api/client'

type TraceEvent = { event: string; data: Record<string, unknown> }

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="max-h-64 overflow-auto rounded-md bg-[#0b0b0b] border border-[#292929] p-3 text-[11px] leading-relaxed text-[#a9b4c5] whitespace-pre-wrap">{JSON.stringify(value, null, 2)}</pre>
}

function Disclosure({ title, icon, children, defaultOpen = false }: { title: string; icon?: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return <section className="rounded-lg border border-[#2a2a2a] bg-[#161616]">
    <button onClick={() => setOpen(v => !v)} className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs font-medium text-[#c9c9c9] hover:bg-[#1c1c1c]">
      {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}{icon}{title}
    </button>
    {open && <div className="border-t border-[#282828] p-3">{children}</div>}
  </section>
}

function FramePair({ projectId, videoId, frame }: { projectId?: string | null; videoId?: string | null; frame?: number }) {
  if (!projectId || !videoId || frame == null) return null
  const raw = frameUrl(projectId, videoId, frame)
  const mask = displayMaskUrl(projectId, videoId, frame)
  return <div className="grid grid-cols-1 xl:grid-cols-2 gap-3 mt-3">
    <div className="rounded-md overflow-hidden border border-[#303030] bg-black">
      <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-[#777] border-b border-[#282828]">Before · frame {frame}</div>
      <img src={raw} alt={`Frame ${frame} before segmentation`} className="w-full max-h-72 object-contain" />
    </div>
    <div className="rounded-md overflow-hidden border border-violet-500/30 bg-black">
      <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-violet-300 border-b border-[#282828]">After · mask overlay</div>
      <div className="relative">
        <img src={raw} alt={`Frame ${frame} after segmentation`} className="w-full max-h-72 object-contain" />
        <img src={mask} alt="Segmentation mask overlay" className="absolute inset-0 w-full h-full object-contain opacity-75" />
      </div>
    </div>
  </div>
}

function eventFrame(event: TraceEvent): number | undefined {
  const d = event.data || {}
  const a = (d.arguments || {}) as Record<string, unknown>
  const r = (d.result || {}) as Record<string, unknown>
  for (const v of [a.frame_idx, r.frame_idx]) if (typeof v === 'number') return v
  return undefined
}

function eventVideo(event: TraceEvent, fallback?: string | null): string | undefined {
  const d = event.data || {}
  const a = (d.arguments || {}) as Record<string, unknown>
  const r = (d.result || {}) as Record<string, unknown>
  return typeof a.video_id === 'string' ? a.video_id : typeof r.video_id === 'string' ? r.video_id : fallback || undefined
}

function TraceEventCard({ event, trace }: { event: TraceEvent; trace: AgentTrace }) {
  const [raw, setRaw] = useState(false)
  const d = event.data || {}
  const kind = event.event
  if (kind === 'reasoning' || kind === 'message') return <div className={`rounded-lg border p-3 ${kind === 'reasoning' ? 'border-violet-500/20 bg-violet-500/5' : 'border-blue-500/20 bg-blue-500/5'}`}>
    <div className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-[#888] mb-1">{kind === 'reasoning' ? <Sparkles size={13} /> : <MessageSquareText size={13} />}{kind}</div>
    <p className="text-sm whitespace-pre-wrap text-[#ddd] leading-relaxed">{String(d.text || '')}</p>
  </div>
  if (kind === 'status' || kind === 'usage') return <div className="text-xs text-[#777] px-1">{kind === 'status' ? String(d.phase || 'status') : `${d.total_tokens || '?'} tokens`}</div>
  if (kind !== 'tool_call' && kind !== 'tool_result' && kind !== 'ui') return null
  if (kind === 'ui') return null
  const name = String(d.name || 'tool')
  const frame = eventFrame(event)
  const videoId = eventVideo(event, trace.video_id)
  const result = (d.result || {}) as Record<string, unknown>
  return <article className="rounded-lg border border-[#303030] bg-[#151515] overflow-hidden">
    <div className="flex items-center gap-2 px-3 py-2 border-b border-[#292929]">
      <Wrench size={14} className={kind === 'tool_result' && d.ok === false ? 'text-red-400' : 'text-cyan-400'} />
      <span className="font-mono text-xs text-[#e1e1e1]">{name}</span>
      {frame != null && <span className="text-[11px] text-[#777]">frame {frame}</span>}
      {kind === 'tool_result' && <span className={`ml-auto text-[11px] ${d.ok === false ? 'text-red-400' : 'text-green-400'}`}>{d.ok === false ? 'failed' : 'complete'} {typeof d.elapsed_ms === 'number' ? `· ${d.elapsed_ms}ms` : ''}</span>}
    </div>
    <div className="p-3 space-y-2">
      {kind === 'tool_call' && <><div className="text-[11px] text-[#777]">Input</div><JsonBlock value={d.arguments || {}} /></>}
      {kind === 'tool_result' && <><div className="text-[11px] text-[#777]">Output</div><JsonBlock value={result} /></>}
      {(name === 'goto_frame' || name === 'inspect_frame' || name === 'text_segment' || name === 'segment_text_interval' || name === 'anchor_frame_labeling_loop') && <FramePair projectId={trace.project_id} videoId={videoId} frame={frame} />}
      <button onClick={() => setRaw(v => !v)} className="flex items-center gap-1 text-[11px] text-[#777] hover:text-white"><Code2 size={12} />{raw ? 'Hide raw event' : 'View raw event'}</button>
      {raw && <JsonBlock value={event} />}
    </div>
  </article>
}

export default function AgentTraceViewer() {
  const [data, setData] = useState<{ traces: AgentTraceSummary[]; projects: { id: string; name: string }[]; videos: { project_id: string; id: string; name: string }[] }>({ traces: [], projects: [], videos: [] })
  const [projectId, setProjectId] = useState('')
  const [videoId, setVideoId] = useState('')
  const [selected, setSelected] = useState<number | null>(null)
  const [trace, setTrace] = useState<AgentTrace | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    setLoading(true); setError('')
    try { setData(await getAgentTraces({ projectId: projectId || undefined, videoId: videoId || undefined })) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not load saved traces') }
    finally { setLoading(false) }
  }, [projectId, videoId])
  useEffect(() => { void load() }, [load])
  useEffect(() => {
    if (selected == null) { setTrace(null); return }
    getAgentTrace(selected).then(setTrace).catch(e => setError(e instanceof Error ? e.message : 'Could not load trace'))
  }, [selected])
  useEffect(() => { if (data.traces.length && !data.traces.some(t => t.id === selected)) setSelected(data.traces[0].id) }, [data.traces, selected])

  const visibleVideos = useMemo(() => data.videos.filter(v => !projectId || v.project_id === projectId), [data.videos, projectId])
  const events = (trace?.trace_json || []) as TraceEvent[]
  return <div className="flex min-h-0 flex-1 bg-[#101010] text-[#ddd] overflow-hidden">
    <aside className="w-[320px] shrink-0 border-r border-[#292929] flex flex-col bg-[#141414]">
      <div className="p-3 border-b border-[#292929] space-y-2"><div className="flex items-center gap-2"><Database size={16} className="text-violet-400" /><h2 className="text-sm font-semibold">Agent traces</h2><button onClick={() => void load()} className="ml-auto text-[#777] hover:text-white"><RefreshCw size={14} /></button></div>
        <select value={projectId} onChange={e => { setProjectId(e.target.value); setVideoId('') }} className="w-full bg-[#0d0d0d] border border-[#303030] rounded px-2 py-1.5 text-xs"><option value="">All projects</option>{data.projects.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
        <select value={videoId} onChange={e => setVideoId(e.target.value)} className="w-full bg-[#0d0d0d] border border-[#303030] rounded px-2 py-1.5 text-xs"><option value="">All videos</option>{visibleVideos.map(v => <option key={`${v.project_id}:${v.id}`} value={v.id}>{v.name}</option>)}</select>
      </div>
      <div className="flex-1 overflow-auto">{loading && <div className="p-4 text-xs text-[#777] flex gap-2"><Loader size={14} className="animate-spin" />Loading traces…</div>}{!loading && !data.traces.length && <div className="p-4 text-xs text-[#777]">No saved traces yet. Run the agent once; completed runs are saved automatically.</div>}{data.traces.map(t => <button key={t.id} onClick={() => setSelected(t.id)} className={`w-full text-left p-3 border-b border-[#252525] hover:bg-[#1c1c1c] ${selected === t.id ? 'bg-violet-500/10 border-l-2 border-l-violet-400' : ''}`}><div className="text-xs font-medium truncate text-[#ddd]">{t.user_text || 'Untitled agent run'}</div><div className="mt-1 text-[11px] text-[#777] truncate">{t.project_name || t.project_id} · {t.video_name || t.video_id}</div><div className="mt-1 text-[10px] text-[#555]">#{t.id} · {t.created_at} · {t.model || 'model unknown'}</div></button>)}</div>
    </aside>
    <main className="flex-1 min-w-0 overflow-auto p-5">{error && <div className="mb-3 rounded bg-red-500/10 text-red-300 p-3 text-sm">{error}</div>}{!trace && !loading && <div className="h-full flex items-center justify-center text-[#666]"><Search size={18} className="mr-2" />Select a trace to inspect its execution.</div>}{trace && <div className="max-w-6xl mx-auto space-y-4"><header><div className="flex items-center gap-2 text-violet-300 text-xs uppercase tracking-widest"><Bot size={14} />Trace #{trace.id}</div><h1 className="mt-1 text-xl font-semibold text-white">{trace.user_text}</h1><p className="mt-1 text-xs text-[#777]">{trace.project_id} / {trace.video_id} · {trace.model} · {trace.created_at}</p></header>
      <Disclosure title="System prompt" icon={<Sparkles size={13} />}><pre className="whitespace-pre-wrap text-xs leading-relaxed text-[#c8c8c8]">{trace.system_prompt || '(none)'}</pre></Disclosure>
      <Disclosure title={`Full LLM transcript (${trace.messages_json?.length || 0} messages)`} icon={<MessageSquareText size={13} />}><JsonBlock value={trace.messages_json} /></Disclosure>
      <section><div className="mb-2 text-xs font-semibold uppercase tracking-wider text-[#777]">Execution timeline · {events.length} events</div><div className="space-y-3">{events.map((event, i) => <TraceEventCard key={`${i}-${event.event}`} event={event} trace={trace} />)}</div></section>
    </div>}</main>
  </div>
}
