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
