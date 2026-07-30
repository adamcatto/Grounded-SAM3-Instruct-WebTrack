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
