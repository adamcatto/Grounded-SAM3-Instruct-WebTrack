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
  /** Frames between anchor samples and propagation batches (default 1000). Locked after labeling starts. */
  anchor_batch_size?: number
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
  /** Derived from annotated_anchors covering all batch anchor indices; persists in config. */
  anchor_labeling_complete?: boolean
  /** Sync-friendly tracking state separate from SSE in-memory propagation. */
  whole_video_inference?: {
    status?: 'none' | 'running' | 'complete' | 'failed'
    updated_at?: string | null
    host?: string | null
  }
  pose_objects?: Record<string, PoseObject>
  pose_annotations?: Record<string, Record<string, Record<string, PoseAnnotation>>>
  pose_memory_frames?: number[]
  pose_skipped_anchor_frames?: number[]
  pose_tracking?: {
    status?: 'none' | 'running' | 'complete' | 'failed'
    start_frame?: number | null
    end_frame?: number | null
    requested_end_frame?: number | null
    tracks_file?: string | null
    model?: string
    error?: string
  }
}

export interface Project {
  id: string
  name: string
  created_at: string
  tracking_mode?: 'segmentation_tracking' | 'pose_tracking'
  videos: Record<string, VideoMeta>
  registration?: ProjectRegistration
}

export interface PosePoint {
  x: number
  y: number
  visible?: boolean
}

export interface PosePart {
  id: string
  name: string
  color: string
}

export interface PoseObject {
  id: string
  name: string
  color: string
  parts: Record<string, PosePart>
}

export interface VideoRegistration {
  frame_idx: 0
  points: [number, number][]
  labels: number[]
  polygon_vertices?: [number, number][]
  mask_source?: 'point_prompts' | 'convex_hull_polygon' | null
  mask_file?: string | null
  source_corners?: [number, number][] | null
  homography?: number[][] | null
  labeled: boolean
  registered: boolean
  morphology_history?: RegistrationMorphologyEdit[]
  morphology_cursor?: number
  edge_points?: Record<'top' | 'right' | 'bottom' | 'left', [number, number][]>
  calibration_source?: 'floor_mask' | 'partial_edges_radial_distortion' | null
  camera_matrix?: number[][] | null
  distortion_coefficients?: number[] | null
  straightness_rms_pixels?: number | null
}

export interface RegistrationMorphologyEdit {
  id: string
  operation: 'opening' | 'closing'
  kernel_size: number
  pixels_added: number
  pixels_removed: number
  delta_file: string
}

export interface ProjectRegistration {
  version: number
  method: string
  target_size: number
  canvas_width?: number | null
  canvas_height?: number | null
  canvas_offset?: [number, number] | null
  warp_mode?: 'bounded_full_frame_mesh' | 'affine_full_frame' | null
  status: 'labeling' | 'complete'
  videos: Record<string, VideoRegistration>
}

export interface PointAnnotation {
  x: number  // normalized 0-1
  y: number  // normalized 0-1
  label: 1 | 0  // 1=positive, 0=negative
}

export interface PoseAnnotation {
  x?: number
  y?: number
  visible: boolean
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

export interface AgentHistoryTurn {
  role: 'user' | 'assistant'
  content: string
}

export type AgentUiAction =
  | { action: 'select_video'; video_id: string }
  | { action: 'goto_frame'; video_id: string; frame_idx: number }
  | { action: 'refresh_project' }
  | { action: 'set_masks'; video_id: string; frame_idx: number; masks: MaskData }
  | { action: 'select_object'; object_id: string; video_id?: string }
  | { action: 'set_points'; video_id: string; object_id: string; frame_idx: number; points: [number, number][]; labels: number[] }
  | { action: 'start_propagation'; video_id: string; start_frame?: number; end_frame?: number; use_all_anchors?: boolean }
  | { action: 'set_anchor_phase'; video_id?: string; frames: number[]; current_index: number; committed_frame?: number }
  | { action: string; [key: string]: unknown }

export interface AgentChatEvent {
  event: string
  data: Record<string, unknown>
}
