export interface ObjectDef {
  id: string
  name: string
  color: string
  description?: string     // textual instruction / prompt (uses name if empty)
  min_instances?: number   // minimum expected instances per frame (default 1)
  max_instances?: number   // maximum expected instances per frame (default 1)
}

export interface PointPrompts {
  [frameIdx: string]: {
    points: [number, number][]
    labels: number[]
  }
}

export interface VideoMeta {
  id: string
  name: string
  source_path: string
  num_frames: number
  fps: number
  width: number
  height: number
  start_frame?: number
  objects: Record<string, ObjectDef>
  point_prompts: Record<string, PointPrompts>
  instance_groups?: Record<string, number[]>  // ui_obj_id → [sam_obj_id, ...]
  sam3_session_id: string | null
  propagated_frames: number[]
  propagation_complete: boolean
  frames_extracted?: boolean
  all_frames_extracted?: boolean
  preview_indices?: number[]
}

export interface Project {
  id: string
  name: string
  created_at: string
  videos: Record<string, VideoMeta>
}

export interface PointAnnotation {
  x: number  // normalized 0-1
  y: number  // normalized 0-1
  label: 1 | 0  // 1=positive, 0=negative
}

export interface MaskData {
  [objId: string]: string  // base64 PNG
}

export interface PropagationEvent {
  frame: number
  progress: number
  done: boolean
  total_frames?: number
  obj_ids?: string[]
  error?: string
  batch?: number
  batch_start?: number
  batch_end?: number
  status?: string          // 'extracting' | 'initializing_session' for batch_start events
  total_batches?: number
  uncertainty_score?: number  // per-frame confusion score from IdentityTracker
  confusion_windows?: number  // count of confusion windows (in done event)
  // catch_up event fields (sent when reconnecting to a running propagation)
  frames_done?: number
  last_frame?: number
  start_frame?: number
  // init event fields
  actual_start?: number
  frames_to_process?: number
  // extract_progress event fields
  extracted?: number
}

// ─── Uncertainty & Identity Correction types ──────────────────────────────────

export interface ConfusionWindow {
  start: number
  end: number
  obj_ids: string[]
  avg_score: number
}

export interface UncertaintyData {
  per_frame: Record<string, {
    confusion_score: number
    confused_objects?: string[]
    temporal_rejections?: Record<string, string>
    per_object?: Record<string, { anomaly_score: number }>
  }>
  confusion_windows: ConfusionWindow[]
  similarity_matrix: Record<string, number>  // "objA_objB" → similarity score
}

export interface CorrectionRecord {
  id: string
  status: 'pending' | 'applied' | 'rejected'
  window_start: number
  window_end: number
  swap_onset: number
  obj_id_a: string
  obj_id_b: string
  reason: string
  avg_confusion_score: number
  created_at: string
  method?: string
}
