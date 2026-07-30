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

This initial implementation labels a single query frame at a time. CoTracker3
predictions are editable only by choosing a new query frame and running another
prediction; multi-frame correction constraints are future work.
