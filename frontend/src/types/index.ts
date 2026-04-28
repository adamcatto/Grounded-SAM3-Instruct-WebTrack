export interface ObjectDef {
  id: string
  name: string
  color: string
  description?: string     // textual instruction / prompt (uses name if empty)
}

export interface PointPrompts {
  [frameIdx: string]: {
    points: [number, number][]
    labels: number[]
  }
}

/** Persisted anchor annotation timing — keys are real video frame indices (strings). */
export interface AnchorLabelFrameTiming {
  /** When the user navigated onto this anchor to label it (after finishing the previous anchor). */
  entered_frontend_ms: number
  /** When the user clicked Done / next on this anchor. */
  committed_ms: number
  /** committed_ms − entered_frontend_ms */
  duration_ms: number
}

export interface AnchorLabelingTimingVideoAgg {
  first_anchor_entered_ms?: number | null
  last_anchor_committed_ms?: number | null
  /** Wall time from landing on the first anchor through committing the last anchor. */
  whole_video_labeling_wall_ms?: number | null
  /** Sum of per-anchor `duration_ms` values. */
  sum_anchor_durations_ms?: number | null
}

export interface AnchorLabelingTimingConfig {
  frames: Record<string, AnchorLabelFrameTiming>
  video: AnchorLabelingTimingVideoAgg
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
  sam3_session_id: string | null
  propagated_frames: number[]
  propagation_complete: boolean
  frames_extracted?: boolean
  all_frames_extracted?: boolean
  preview_indices?: number[]
  annotated_anchors?: number[]
  anchor_labeling_timing?: AnchorLabelingTimingConfig
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
