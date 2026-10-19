# CrickVision JSONL schema (version 1)

CrickVision writes newline-delimited JSON: each non-empty line is one independent JSON object. Records from one video share `video_id` (currently the input filename stem) and `schema_version: 1`. Frame indices refer to decoded source frames, not the number of frames sampled for inference. Timestamps are milliseconds from the video decoder when available, with a frame-index/FPS fallback.

## Record types

### `person_detection`

| Field | Meaning |
|---|---|
| `video_id`, `shot_id`, `frame_index`, `timestamp_ms` | Source/video context |
| `class` | Always `person` in the current detector |
| `bbox_xyxy` | Pixel box `[x1,y1,x2,y2]` |
| `confidence` | Generic COCO detector score, not a calibrated probability |
| `track_id`, `track_age` | Present when IoU tracking is enabled and a detection is assigned |
| `team`, `team_colour_fraction` | Present when user-configured HSV ranges are enabled; heuristic only |
| `ground_xy` | Optional footpoint projected through a user-provided planar homography |

### `shot_boundary`

Contains `shot_id`, source frame/timestamp, and `score` from the frame-difference heuristic. This can miss cuts or flag motion/flash transitions.

### `ball_candidate`

Contains `center_xy`, `radius`, `candidate_score`, `ball_track_id`, `track_age`, `candidate_path_xy`, and `status: "unverified_candidate"`. This is a Hough-circle proposal, not a verified cricket ball or a physical trajectory.

### `calibration`

If calibration is enabled, the first line describes the accepted manual homography, point count, reprojection error, and matrix. It is metadata, not a per-frame detection.

## Detection benchmark ground truth

Ground truth must explicitly list every reviewed frame, including frames with no people. One row per ground-truth box:

```json
{"video_id":"match-A","frame_index":12,"bbox_xyxy":[20,30,80,210]}
{"video_id":"match-A","frame_index":12,"bbox_xyxy":[220,30,280,210]}
{"video_id":"match-A","frame_index":13,"empty_frame":true}
```

If a frame has one or more boxes, list one row per box. Do not mix `empty_frame: true` with box rows for the same frame. Ground-truth rows without a `video_id` use the legacy key `default`, so set `video_id` explicitly for multi-video datasets.

Run:

```bash
crickvision-benchmark --predictions runs/detections.jsonl --ground-truth datasets/annotations/test.jsonl --iou-threshold 0.5 --output runs/benchmark.json
```

The benchmark matches boxes greedily one-to-one by IoU. It reports frame-level precision, recall, and F1 over explicitly annotated frames only. It is not COCO AP/mAP, does not score unannotated frames, and does not measure tracking, ball, calibration, or event quality. Keep test matches held out from threshold selection and model development.

## CSV export

```bash
crickvision-export --input runs/detections.jsonl --output runs/detections.csv
```

CSV is a convenience view; JSONL is the source-of-record format. Nested values such as candidate histories are serialized as JSON-like cell values. Keep the original JSONL for exact provenance.
