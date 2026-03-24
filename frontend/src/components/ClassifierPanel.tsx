/**
 * ClassifierPanel — per-pixel DINOv2 identity classifier.
 *
 * Training and inference run on the backend (PyTorch + HuggingFace DINOv2).
 * This panel streams progress via SSE and displays per-frame results.
 */

import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import {
  openClassifierStream,
  cancelClassifier,
  getClassifierResults,
  type ClassifierSSEEvent,
} from '../api/client'
import type { ClassifierResults } from '../types/index'
import {
  Brain, Play, Square, ToggleLeft, ToggleRight,
  CheckCircle, XCircle, Loader, AlertCircle, RefreshCw,
} from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────

type Phase =
  | 'idle'
  | 'connecting'
  | 'loading_frames'
  | 'extracting'
  | 'training'
  | 'inferring'
  | 'done'
  | 'error'
  | 'cancelled'

interface TrainState {
  phase:        Phase
  statusMsg:    string
  epoch:        number
  totalEpochs:  number
  loss:         number
  trainAcc:     number
  inferDone:    number
  inferTotal:   number
  errorMsg:     string
  trainFrames:  number[]
  evalFrames:   number[]
}

const INIT_STATE: TrainState = {
  phase: 'idle', statusMsg: '', epoch: 0, totalEpochs: 0,
  loss: 0, trainAcc: 0, inferDone: 0, inferTotal: 0,
  errorMsg: '', trainFrames: [], evalFrames: [],
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function progressPct(s: TrainState): number {
  if (s.phase === 'training' && s.totalEpochs > 0)
    return (s.epoch / s.totalEpochs) * 80
  if (s.phase === 'inferring' && s.inferTotal > 0)
    return 80 + (s.inferDone / s.inferTotal) * 20
  if (s.phase === 'done') return 100
  return 0
}

function phaseLabel(s: TrainState): string {
  switch (s.phase) {
    case 'connecting':     return 'Connecting…'
    case 'loading_frames': return s.statusMsg || 'Loading frames…'
    case 'extracting':     return s.statusMsg || 'Extracting features…'
    case 'training':
      return s.epoch > 0
        ? `Epoch ${s.epoch}/${s.totalEpochs}  ·  loss ${s.loss.toFixed(4)}  ·  acc ${(s.trainAcc * 100).toFixed(1)}%`
        : s.statusMsg || 'Training…'
    case 'inferring':
      return `Inference ${s.inferDone}/${s.inferTotal}`
    case 'done':      return 'Done'
    case 'error':     return `Error: ${s.errorMsg}`
    case 'cancelled': return 'Cancelled'
    default:          return ''
  }
}

// Convert backend snake_case done payload → store ClassifierResults
function toClassifierResults(ev: Extract<ClassifierSSEEvent, { type: 'done' }>): ClassifierResults {
  const frameAssignments: ClassifierResults['frameAssignments'] = {}
  for (const [fidxStr, asgn] of Object.entries(ev.frame_assignments)) {
    frameAssignments[Number(fidxStr)] = asgn
  }
  return {
    objectIds:       ev.object_ids,
    frameAssignments,
    trainFrames:     ev.train_frames,
    evalFrames:      ev.eval_frames,
    trainAccuracy:   ev.train_accuracy,
    evalAgreement:   ev.eval_agreement,
  }
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ClassifierPanel() {
  const project              = useStore(s => s.project)
  const video                = useStore(s => selectCurrentVideo(s))
  const setCurrentFrame      = useStore(s => s.setCurrentFrame)
  const setViewerTab         = useStore(s => s.setViewerTab)
  const classifierResults    = useStore(s => s.classifierResults)
  const showClassifierOverlay = useStore(s => s.showClassifierOverlay)
  const setClassifierResults  = useStore(s => s.setClassifierResults)
  const setShowClassifierOverlay = useStore(s => s.setShowClassifierOverlay)

  const numFrames = video?.num_frames ?? 0
  const objectIds = useMemo(() => (video ? Object.keys(video.objects) : []), [video])

  // Config
  const [frameA,      setFrameA]     = useState(0)
  const [frameB,      setFrameB]     = useState(Math.max(0, numFrames - 1))
  const [step,        setStep]       = useState(5)
  const [epochs,      setEpochs]     = useState(20)
  const [lr,          setLr]         = useState(0.0005)

  // Sync frameB when video changes
  useEffect(() => {
    setFrameB(Math.max(0, numFrames - 1))
  }, [numFrames])

  // Training state
  const [ts, setTs] = useState<TrainState>(INIT_STATE)
  const cleanupRef  = useRef<(() => void) | null>(null)

  const pid = project?.id ?? ''
  const vid = video?.id   ?? ''

  // Split preview
  const splitInfo = useMemo(() => {
    const n       = Math.ceil((frameB - frameA + 1) / Math.max(1, step))
    const nTrain  = Math.floor(n * 0.75)
    const nSkip   = Math.ceil(n * 0.05)
    const nEval   = Math.max(0, n - nTrain - nSkip)
    return { n, nTrain, nSkip, nEval }
  }, [frameA, frameB, step])

  // Load persisted results on mount
  useEffect(() => {
    if (!pid || !vid || classifierResults) return
    getClassifierResults(pid, vid)
      .then(results => {
        setClassifierResults(results)
      })
      .catch(() => { /* no saved results yet — fine */ })
  }, [pid, vid])

  const handleTrain = useCallback(() => {
    if (!pid || !vid || objectIds.length < 2) return

    // Clean up any previous stream
    cleanupRef.current?.()
    setTs({ ...INIT_STATE, phase: 'connecting' })
    setClassifierResults(null)

    const cleanup = openClassifierStream(
      pid, vid,
      { startFrame: frameA, endFrame: frameB, step, epochs, lr, trainRatio: 0.75, evalRatio: 0.20 },
      (ev: ClassifierSSEEvent) => {
        setTs(prev => {
          switch (ev.type) {
            case 'split':
              return { ...prev, trainFrames: ev.train_frames, evalFrames: ev.eval_frames }
            case 'status':
              return { ...prev, phase: 'loading_frames', statusMsg: ev.message }
            case 'epoch':
              return {
                ...prev, phase: 'training',
                epoch: ev.epoch, totalEpochs: ev.epochs,
                loss: ev.loss, trainAcc: ev.train_acc,
              }
            case 'inference_progress':
              return { ...prev, phase: 'inferring', inferDone: ev.done, inferTotal: ev.total }
            case 'done': {
              const results = toClassifierResults(ev)
              setClassifierResults(results)
              return { ...prev, phase: 'done' }
            }
            case 'cancelled':
              return { ...prev, phase: 'cancelled' }
            case 'error':
              return { ...prev, phase: 'error', errorMsg: ev.message }
            default:
              return prev
          }
        })

        // Map specific status messages to phases
        if (ev.type === 'status') {
          const msg = ev.message.toLowerCase()
          setTs(prev => ({
            ...prev,
            phase: msg.includes('backbone') || msg.includes('extract')
              ? 'extracting'
              : 'loading_frames',
          }))
        }
      },
    )
    cleanupRef.current = cleanup
  }, [pid, vid, objectIds, frameA, frameB, step, epochs, lr, setClassifierResults])

  const handleStop = useCallback(() => {
    cleanupRef.current?.()
    cleanupRef.current = null
    cancelClassifier(pid, vid).catch(() => {})
    setTs(prev => ({ ...prev, phase: 'cancelled' }))
  }, [pid, vid])

  const jumpToFrame = (f: number) => {
    setCurrentFrame(f)
    setViewerTab('annotate')
  }

  const isRunning = ts.phase === 'connecting' || ts.phase === 'loading_frames' ||
                    ts.phase === 'extracting'  || ts.phase === 'training' || ts.phase === 'inferring'

  // ── Guard: no video / no objects ──────────────────────────────────────────

  if (!video || objectIds.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0d0d0d] text-[#555] text-sm">
        No video selected or no objects defined.
      </div>
    )
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  const pct = progressPct(ts)

  return (
    <div className="flex-1 overflow-y-auto bg-[#0d0d0d] text-[#ccc]">
      <div className="max-w-2xl mx-auto p-6 space-y-6">

        {/* Header */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Brain size={16} className="text-violet-400" />
            <h1 className="text-sm font-semibold text-white">Identity Classifier</h1>
          </div>
          <p className="text-xs text-[#666] leading-relaxed">
            Trains a DINOv2-backed per-pixel classifier on the backend to learn each
            object's visual appearance, then flags frames where its predictions disagree
            with SAM3's identity assignment.
          </p>
        </div>

        {/* Config */}
        <section className="rounded-xl border border-[#2a2a2a] bg-[#111] p-4 space-y-4">
          <h2 className="text-[10px] font-semibold text-[#555] uppercase tracking-wider">Configuration</h2>

          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-[10px] text-[#666] uppercase tracking-wide">Start frame</span>
              <input
                type="number" min={0} max={frameB} value={frameA}
                onChange={e => setFrameA(Math.max(0, Math.min(frameB, Number(e.target.value))))}
                className="bg-[#1a1a1a] border border-[#2a2a2a] rounded px-2 py-1 text-sm text-white font-mono"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[10px] text-[#666] uppercase tracking-wide">End frame</span>
              <input
                type="number" min={frameA} max={numFrames - 1} value={frameB}
                onChange={e => setFrameB(Math.max(frameA, Math.min(numFrames - 1, Number(e.target.value))))}
                className="bg-[#1a1a1a] border border-[#2a2a2a] rounded px-2 py-1 text-sm text-white font-mono"
              />
            </label>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-[10px] text-[#666] uppercase tracking-wide">Sample every Nth frame</span>
              <input type="number" min={1} max={100} value={step}
                onChange={e => setStep(Math.max(1, Number(e.target.value)))}
                className="bg-[#1a1a1a] border border-[#2a2a2a] rounded px-2 py-1 text-sm text-white font-mono" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[10px] text-[#666] uppercase tracking-wide">Epochs</span>
              <input type="number" min={1} max={100} value={epochs}
                onChange={e => setEpochs(Math.max(1, Number(e.target.value)))}
                className="bg-[#1a1a1a] border border-[#2a2a2a] rounded px-2 py-1 text-sm text-white font-mono" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[10px] text-[#666] uppercase tracking-wide">Learning rate</span>
              <input type="number" min={1e-5} max={0.01} step={1e-4} value={lr}
                onChange={e => setLr(Number(e.target.value))}
                className="bg-[#1a1a1a] border border-[#2a2a2a] rounded px-2 py-1 text-sm text-white font-mono" />
            </label>
          </div>

          {/* Split summary */}
          <div className="rounded-lg bg-[#0d0d0d] border border-[#1e1e1e] p-3 grid grid-cols-3 gap-3 text-center">
            <div>
              <div className="text-lg font-bold text-emerald-400">{splitInfo.nTrain}</div>
              <div className="text-[10px] text-[#555]">Train (75%)</div>
            </div>
            <div>
              <div className="text-lg font-bold text-[#444]">{splitInfo.nSkip}</div>
              <div className="text-[10px] text-[#555]">Discarded (5%)</div>
            </div>
            <div>
              <div className="text-lg font-bold text-blue-400">{splitInfo.nEval}</div>
              <div className="text-[10px] text-[#555]">Eval (20%)</div>
            </div>
          </div>
          <p className="text-[10px] text-[#555]">
            Backbone: DINOv2-small (frozen) · Head trained on-device
            {ts.trainFrames.length > 0 && ` · ${ts.trainFrames.length} train / ${ts.evalFrames.length} eval frames with masks`}
          </p>
        </section>

        {/* Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          {!isRunning ? (
            <button
              onClick={handleTrain}
              disabled={objectIds.length < 2}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Play size={14} /> Train Classifier
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-700 hover:bg-red-600 text-white text-sm font-medium transition-colors"
            >
              <Square size={13} /> Stop
            </button>
          )}

          {classifierResults && (
            <button
              onClick={() => setShowClassifierOverlay(!showClassifierOverlay)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium transition-colors ${
                showClassifierOverlay
                  ? 'bg-violet-900/40 border-violet-600 text-violet-300'
                  : 'bg-[#1a1a1a] border-[#2a2a2a] text-[#888] hover:text-white'
              }`}
            >
              {showClassifierOverlay ? <ToggleRight size={14} /> : <ToggleLeft size={14} />}
              Overlay {showClassifierOverlay ? 'ON' : 'OFF'}
            </button>
          )}

          {objectIds.length < 2 && (
            <span className="text-xs text-amber-400">Need ≥ 2 objects to train.</span>
          )}
        </div>

        {/* Progress */}
        {(isRunning || ts.phase === 'cancelled') && (
          <div className="rounded-xl border border-[#2a2a2a] bg-[#111] p-4 space-y-3">
            <div className="flex items-center gap-2 text-sm text-[#aaa]">
              {isRunning
                ? <Loader size={13} className="animate-spin text-violet-400 flex-shrink-0" />
                : <RefreshCw size={13} className="text-[#555] flex-shrink-0" />}
              <span className="truncate">{phaseLabel(ts)}</span>
            </div>
            {pct > 0 && (
              <div className="h-1.5 rounded-full bg-[#1a1a1a] overflow-hidden">
                <div
                  className="h-full rounded-full bg-violet-500 transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {ts.phase === 'error' && (
          <div className="rounded-xl border border-red-900 bg-red-950/30 p-4 flex items-start gap-3">
            <AlertCircle size={14} className="text-red-400 mt-0.5 flex-shrink-0" />
            <p className="text-sm text-red-300">{ts.errorMsg}</p>
          </div>
        )}

        {/* Results */}
        {classifierResults && (
          <>
            <section className="rounded-xl border border-[#2a2a2a] bg-[#111] p-4 space-y-3">
              <h2 className="text-[10px] font-semibold text-[#555] uppercase tracking-wider">Results</h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg bg-[#0d0d0d] border border-[#1e1e1e] p-3 text-center">
                  <div className="text-xl font-bold text-emerald-400">
                    {(classifierResults.trainAccuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-[#555] mt-0.5">Train accuracy</div>
                </div>
                <div className="rounded-lg bg-[#0d0d0d] border border-[#1e1e1e] p-3 text-center">
                  <div className="text-xl font-bold text-blue-400">
                    {(classifierResults.evalAgreement * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-[#555] mt-0.5">Eval agreement with SAM3</div>
                </div>
              </div>
              {classifierResults.evalAgreement < 0.9 && (
                <p className="text-xs text-amber-400 flex items-center gap-1.5">
                  <AlertCircle size={11} />
                  {((1 - classifierResults.evalAgreement) * 100).toFixed(1)}% of eval frames have
                  identity discrepancies — enable the overlay to inspect them in the annotate view.
                </p>
              )}
            </section>

            <EvalFrameTable
              results={classifierResults}
              objects={video.objects}
              onJump={jumpToFrame}
            />
          </>
        )}

      </div>
    </div>
  )
}

// ── Eval frame table ──────────────────────────────────────────────────────────

interface EvalFrameTableProps {
  results: ClassifierResults
  objects: Record<string, { name: string; color: string }>
  onJump:  (f: number) => void
}

function EvalFrameTable({ results, objects, onJump }: EvalFrameTableProps) {
  const [showAll, setShowAll] = useState(false)

  const discrepancies = results.evalFrames.filter(fidx => {
    const asgn = results.frameAssignments[fidx]
    return asgn && Object.entries(asgn).some(([samId, a]) => a.predictedClass !== samId)
  })

  const rows = showAll ? results.evalFrames : discrepancies.slice(0, 50)

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-[10px] font-semibold text-[#555] uppercase tracking-wider">
          {showAll
            ? `All eval frames (${results.evalFrames.length})`
            : `Discrepancies — ${discrepancies.length} / ${results.evalFrames.length} eval frames`}
        </h2>
        <button
          onClick={() => setShowAll(v => !v)}
          className="text-[10px] text-[#555] hover:text-[#888] transition-colors"
        >
          {showAll ? 'Discrepancies only' : 'Show all eval frames'}
        </button>
      </div>

      <div className="rounded-xl border border-[#2a2a2a] bg-[#111] overflow-hidden">
        <div className="grid grid-cols-[56px_1fr_1fr_52px] text-[10px] text-[#555] uppercase tracking-wide px-3 py-2 border-b border-[#1e1e1e] bg-[#151515]">
          <span>Frame</span>
          <span>SAM3</span>
          <span>Classifier</span>
          <span className="text-right">Conf</span>
        </div>

        {rows.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-emerald-400 flex items-center justify-center gap-2">
            <CheckCircle size={14} />
            No discrepancies on eval frames.
          </div>
        )}

        {rows.flatMap(fidx => {
          const asgn = results.frameAssignments[fidx]
          if (!asgn) return []
          return Object.entries(asgn).map(([samObjId, a]) => {
            const samObj = objects[samObjId]
            const clsObj = objects[a.predictedClass]
            const match  = a.predictedClass === samObjId
            return (
              <div
                key={`${fidx}_${samObjId}`}
                className={`grid grid-cols-[56px_1fr_1fr_52px] items-center px-3 py-1.5 border-b border-[#181818] hover:bg-[#161616] ${!match ? 'bg-amber-950/10' : ''}`}
              >
                <button
                  onClick={() => onJump(fidx)}
                  className="text-[11px] font-mono text-blue-400 hover:text-blue-300 text-left"
                >
                  {fidx}
                </button>
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: samObj?.color ?? '#888' }} />
                  <span className="text-xs text-[#aaa] truncate">{samObj?.name ?? samObjId}</span>
                </div>
                <div className="flex items-center gap-1.5 min-w-0">
                  {match
                    ? <CheckCircle size={10} className="text-emerald-500 flex-shrink-0" />
                    : <XCircle    size={10} className="text-amber-500 flex-shrink-0" />}
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: clsObj?.color ?? '#888' }} />
                  <span
                    className={`text-[11px] truncate ${match ? 'text-[#aaa]' : 'text-amber-300'}`}
                  >
                    {clsObj?.name ?? a.predictedClass}
                  </span>
                </div>
                <span className="text-[11px] font-mono text-[#666] text-right">
                  {(a.confidence * 100).toFixed(0)}%
                </span>
              </div>
            )
          })
        })}
      </div>
    </section>
  )
}
