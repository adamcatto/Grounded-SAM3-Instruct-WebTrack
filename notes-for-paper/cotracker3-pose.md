# CoTracker3 pose-tracking mode

Pose projects are distinct from segmentation projects through
`tracking_mode: pose_tracking`. Each video stores top-level biological objects
(for example, `mouse`) containing independently colored landmark parts (for
example, `snout`, `midback`, and `start-tail`). The user selects a query/start
frame, clicks one location for every part on that frame, and requests
prediction over the next configurable number of frames.

The backend uses Meta's scaled offline CoTracker3 checkpoint
(`facebook/cotracker3`, `scaled_offline.pth`). Sparse queries have shape
`(query_frame, x, y)`. Long requested ranges are processed in clips of at most
60 frames, using the final prediction of one clip as the query for the next.
Predicted coordinates and visibility are saved as normalized, frame-indexed
JSON in each video directory. This representation is independent of display
resolution and keeps the top-level object/part hierarchy.

The checkpoint can be installed with:

```bash
python scripts/download_cotracker3.py
```

The default runtime path is `/opt/models/cotracker3/scaled_offline.pth`;
`COTRACKER3_CHECKPOINT` overrides it. If neither exists, the backend downloads
the checkpoint through `huggingface_hub`. PyTorch Hub obtains Meta's official
model implementation on first load.

Pose parts can also be labeled on regularly spaced anchor frames throughout a
video. Completing an anchor caches the normalized convolutional feature map
from CoTracker3's own encoder in the video's `pose_feature_cache` directory.
When prediction begins on a frame without direct labels, the application
correlates that frame against all fully labeled memory frames. If a prediction
for the immediately preceding frame is available, its feature descriptor has
weight 0.6 and the anchor-memory descriptors share the remaining weight 0.4;
without a preceding prediction, anchor weights sum to 1.0. The peak of the
weighted correlation map initializes each landmark query. A direct annotation
on the requested frame is a hard constraint and supersedes memory matching.

This is an inference-time memory layer around the released CoTracker3 model,
not a modification of its trained transformer. CoTracker3's internal
correlation pyramid does not expose a supported SAM-style external
cross-attention weight. After memory initialization, its standard joint
point-tracking transformer predicts the requested forward interval.

## Candidate methodological contributions

The following are the pose system's potentially novel ML/AI contributions and
should be described as system contributions unless a literature review
establishes stronger novelty:

- **Anchor-memory initialization for CoTracker3.** Sparse human annotations
  distributed through a video are converted into reusable CoTracker3 encoder
  features. This adds long-range, user-correctable visual memory to an
  otherwise query-driven inference workflow without retraining the model.
- **Recency-weighted multi-anchor correspondence.** A landmark's query
  position is inferred from a weighted feature-correlation map: the preceding
  prediction receives weight 0.6, while all labeled anchor memories share
  weight 0.4. This combines short-term temporal continuity with long-range
  recovery from appearance drift or tracking failure.
- **Visibility-aware partial anchor memory.** Reviewers can explicitly mark an
  anatomical part as occluded instead of inventing a coordinate. That part is
  omitted from the anchor's descriptor correlation while visible parts on the
  same image still contribute. Anchor weights are normalized independently per
  landmark over the memories in which it is visible, preserving the intended
  recency-versus-anchor weighting despite heterogeneous occlusion.
- **Human constraints override model memory.** A point placed directly on the
  query frame is treated as a hard constraint. Thus manual corrections,
  anchor-derived proposals, and CoTracker3 predictions form an explicit
  precedence hierarchy rather than being blended ambiguously.
- **Hierarchical semantic pose representation.** Independently colored
  landmarks are nested beneath biological objects, retaining the association
  between an animal and its anatomical parts while still supplying the sparse
  point queries expected by CoTracker3.

## Candidate system contributions

- **Interactive anchor queue with feature caching.** Regularly spaced frames
  are presented as a labeling queue, and each completed frame is immediately
  encoded and persisted. Cached features can be reused across later prediction
  requests instead of repeatedly running the image encoder. Frames may also be
  deliberately skipped; this decision persists in the queue and prevents the
  frame from entering the correspondence memory bank.
- **Progressive prediction publication.** Long videos are processed in bounded
  clips, with each completed chunk atomically written to the project state.
  The annotation viewer polls that state, so newly predicted frames become
  inspectable and correctable before the entire requested range finishes.
- **Resolution-independent, incrementally editable output.** Normalized
  coordinates and visibility values are stored by object, part, and frame.
  This supports different display resolutions and permits annotations and
  predictions to coexist in the same interactive project.
- **Reversible human correction layer.** Pose annotations can be placed,
  repositioned, removed individually, or cleared as a frame-level operation.
  Each edit is persisted immediately but represented as a reversible command,
  allowing frame-local undo/redo while retaining the underlying model
  predictions. Manual annotations are rendered above predictions and therefore
  act as visible, editable corrections rather than destructively rewriting the
  predicted track file.

These notes must remain synchronized with implementation changes. Any future
algorithmic or system-level novelty should be added to the relevant document
under `notes-for-paper/` as part of the same change.
