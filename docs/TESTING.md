# Testing and release verification

## Automated checks

Create a Python 3.10+ virtual environment, then run from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
python -m compileall -q crickvision tests
python -m pytest -q
ruff check crickvision tests
```

The GitHub Actions workflow runs compilation and pytest on Python 3.10, 3.11, and 3.12. A workflow file is not proof that a run passed; inspect the latest Actions run before declaring CI green.

## CLI smoke test

Use a video file you are permitted to process:

```bash
python -m crickvision.cli --input path/to/clip.mp4 --output runs/annotated.mp4 --detections runs/detections.jsonl --every-n-frames 2
python -m crickvision.manifest datasets/manifests/manifest.jsonl
```

Check that:
- the process exits successfully and reports frames read/analysed;
- the MP4 opens and its duration/resolution are plausible;
- JSONL parses one object per non-empty line;
- frame indices are monotonic and timestamps are non-negative;
- shot-boundary records are inspected visually, not assumed correct;
- sampled frames only contain detections because skipped frames are not interpolated;
- track IDs are treated as temporary local IDs, not player identities.

The detector may download Torchvision weights on first use. Test once with network access, then test offline only after weights are cached. CUDA should only be tested on a compatible PyTorch/CUDA installation.

## Dashboard smoke test

```bash
python -m pip install -e ".[dashboard]"
streamlit run crickvision/dashboard.py
```

Upload a short, licensed video, run analysis, and verify both download buttons. This dashboard is for a single trusted local user. Do not deploy it publicly: it has no authentication, quotas, per-user isolation, or hardened upload controls. Large uploads can exhaust memory, disk, or CPU.

## Data and benchmark validation

- Keep dataset metadata in JSONL manifests; do not commit broadcast clips or extracted frames without explicit redistribution rights.
- Split data by whole match/source to prevent adjacent-frame leakage.
- Preserve a provenance record for each clip, annotation version, checkpoint, and evaluation configuration.
- Evaluate on held-out matches. Report dataset size, conditions, model/checkpoint hash, precision/recall/F1 or suitable task metrics, hardware, and known limitations.
- The current `crickvision.metrics.detection_metrics` helper is frame-level greedy IoU matching. It is not COCO AP/mAP or a tracking metric such as HOTA/IDF1.

## Release gates

Do not describe the project as production-ready until all of these are complete:

1. CI is green on all supported Python versions.
2. A reproducible end-to-end run on a legally usable cricket clip is verified.
3. Cricket-specific detection and ball tracking are trained/evaluated on documented data, or clearly remain out of scope.
4. Tracking, team classification, shot segmentation, field calibration, and event inference have task-appropriate held-out evaluation.
5. Dependency, checkpoint, dataset, and footage rights have been reviewed.
6. The original-code MIT licence is represented consistently and third-party/model/data terms are audited.
7. Security/privacy review, bounded processing, error handling, and documentation match the actual release surface.

Passing unit tests alone does not establish cricket accuracy or production readiness.

## Reproducible detection benchmark

Ground-truth JSONL contains one row per labelled person box, using original source frame indices:

```json
{"video_id":"match-001","frame_index":120,"bbox_xyxy":[100,50,180,320]}
```

Create predictions with the CLI, then evaluate:

```bash
crickvision-benchmark --predictions runs/detections.jsonl --ground-truth datasets/annotations/heldout.jsonl --iou-threshold 0.5 --output runs/benchmark.json
crickvision-export --input runs/detections.jsonl --output runs/detections.csv
```

Rows without `video_id` use the shared ID `default`; ensure you use distinct `video_id` values if combining multiple videos. Every annotated ground-truth box should be a separate row. Frames with no visible people can be represented with an explicit row containing video_id, frame_index, and empty_frame=true. Only include frames that were exhaustively reviewed. The evaluator reports aggregate greedy IoU precision/recall/F1 and per-frame results. It is not AP/mAP and should not be compared to published COCO scores.

## Fine-tune a person detector

Create a reviewed image manifest with one row per image. Paths are relative to the manifest. Each row must contain match_id, a split, and a list of person boxes; an empty list means a reviewed image with no visible target people. The trainer rejects match-level split leakage.

```json
{"image_path":"frames/train-001.jpg","match_id":"match-001","split":"train","boxes_xyxy":[[40,25,95,240]]}
{"image_path":"frames/validation-001.jpg","match_id":"match-002","split":"validation","boxes_xyxy":[]}
```

Keep all frames from one match/source in a single split. The trainer requires both train and validation rows:

```bash
crickvision-train-detector --manifest datasets/annotations/person-boxes.jsonl --output models/person-finetuned.pt --epochs 5 --batch-size 2
python -m crickvision.cli --input path/to/clip.mp4 --output runs/fine-tuned.mp4 --weights models/person-finetuned.pt
```

The training command fine-tunes Faster R-CNN to a single target class, person. It does not train a ball detector or infer cricket roles. The checkpoint is selected by validation loss, which is only a training diagnostic; choose final models using a separate held-out match and the benchmark workflow. The default initial weights may download from Torchvision. Ensure your dataset and the pretrained weights' terms permit your intended use.

## Manual ground-plane calibration

Use the example config in configs/calibration.example.json as a schema only; replace every point with measured correspondences from the actual camera view. All ground points must lie on the same planar surface and use consistent units.

```bash
python -m crickvision.cli --input path/to/clip.mp4 --output runs/calibrated.mp4 --calibration configs/calibration.json
```

The CLI records the homography and reprojection error and adds ground_xy to each person detection. The projected point is the bounding-box bottom-centre. A valid homography does not guarantee accurate field positions if the landmarks are wrong, the camera moves, the point is not on the calibrated plane, or the image is a close-up/replay. This does not provide 3D coordinates or airborne ball trajectories.

## Experimental ball-candidate mode

```bash
python -m crickvision.cli --input path/to/clip.mp4 --output runs/candidates.mp4 --ball-candidates --ball-min-radius 1 --ball-max-radius 8
```

This opt-in mode proposes small circular regions using OpenCV HoughCircles and links nearby proposals into short tracklets. JSONL rows are labelled type=ball_candidate and status=unverified_candidate. The local-contrast score is a ranking heuristic, not a probability. Bright objects, graphics, highlights, and noise can all trigger proposals. Candidate track IDs are not proof of ball identity; this is not a validated ball detector, trajectory estimator, or delivery-event system.
