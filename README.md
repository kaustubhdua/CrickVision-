# 🏏 CrickVision

**Open-source computer vision for cricket video intelligence.**  
*Turn broadcast pixels into structured cricket insights.*

CrickVision is a local-first project to detect and track cricket players, classify teams, map field positions, and progressively analyse the ball and delivery events from video. The long-term goal is a reproducible, well-tested toolkit for developers, researchers, coaches, and cricket analysts.

> **Current status: early prototype.** The repository currently starts with a general-purpose pretrained person detector baseline. It is **not** a cricket-trained model, does not yet provide persistent tracking/team classification/ball tracking, and has no claimed accuracy benchmark. The roadmap below is aspirational until each item is implemented and evaluated.

## Project goals

- Process local cricket video without hosted AI APIs.
- Detect visible people with a reproducible baseline.
- Build cricket-specific tracking, team classification, field calibration, and ball analysis in stages.
- Keep model outputs, timestamps, confidence values, and failure states explicit.
- Support CPU inference first; evaluate optional acceleration later.
- Make evaluation reproducible and keep training/test matches separate to avoid data leakage.
- Respect the licences and redistribution terms of code, pretrained weights, datasets, and source footage.

## What works now

- A baseline person detector using Torchvision's pretrained Faster R-CNN ResNet-50 FPN model.
- A CLI to annotate an input video with person bounding boxes and confidence scores.
- Configurable confidence threshold and frame sampling interval.
- JSONL detections with frame index, timestamp, bounding box, and confidence.
- Local video processing; no API key is required. The first run may download pretrained weights unless they are already cached.

The baseline detects the generic COCO `person` class. Cricket-specific accuracy has not been established.

## Quick start

### 1. Requirements

- Python 3.10 or 3.11 recommended
- FFmpeg is useful for converting unusual video formats (optional for OpenCV-readable files)
- A CPU works; inference may be slow on long or high-resolution videos
- Internet access may be needed on first run to download Torchvision weights

### 2. Create an environment

```bash
git clone https://github.com/kaustubhdua/CrickVision-.git
cd CrickVision-
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the baseline

```bash
python -m crickvision.cli --input path/to/match.mp4 --output runs/match_annotated.mp4
```

Optional arguments:

```bash
python -m crickvision.cli --input match.mp4 --output runs/annotated.mp4 \
  --detections runs/detections.jsonl --threshold 0.65 --every-n-frames 2
```

- `--threshold`: minimum person confidence from 0 to 1 (default: 0.60).
- `--every-n-frames`: run detection on every Nth frame (default: 1). Skipped frames are not interpolated.
- `--detections`: JSON Lines output path (default: alongside the annotated video as `<output>.jsonl`).
- `--device`: `auto`, `cpu`, or `cuda` (default: `auto`; CUDA requires a compatible PyTorch installation).

The output video shows detections, not identity tracking. A box appearing in consecutive frames is not proof that it belongs to the same player.

## Architecture roadmap

### Phase 0 — Foundations and reproducibility
- [x] Create repository structure and project scope.
- [x] Add generic pretrained person-detection baseline and video CLI.
- [x] Export timestamped detections as JSONL.
- [ ] Add automated unit/integration tests and CI.
- [ ] Add dependency lockfile, model provenance, and software/data licence inventory.
- [ ] Establish a small, legally usable, manually labelled cricket benchmark.

### Phase 1 — Cricket video baseline
- [ ] Audit cricket-specific datasets and annotation formats.
- [ ] Build a video manifest tool (source, match/clip ID, timestamps, rights, resolution, split).
- [ ] Evaluate person recall separately for wide shots, pitch views, close-ups, day/night, and small distant players.
- [ ] Add shot-boundary detection and shot/view labels.
- [ ] Add model configuration and structured logs.
- **Exit gate:** a reproducible benchmark report on held-out matches; no performance claims without measured results.

### Phase 2 — Multi-object player tracking
- [ ] Add tracker adapter and baseline such as ByteTrack or BoT-SORT after licence/dependency review.
- [ ] Track players within a shot and reset or isolate track IDs at camera cuts.
- [ ] Record track age, confidence, missing detections, and fragmentation.
- [ ] Evaluate HOTA, IDF1, ID switches, and fragmentation against labelled clips.
- [ ] Add track visualisation and CSV/JSON exports.
- **Exit gate:** documented tracking results across multiple broadcast views.

### Phase 3 — Team and official classification
- [ ] Extract torso crops and robust colour features (e.g. HSV).
- [ ] Support user-configured team colours and Team A / Team B / Official / Unknown labels.
- [ ] Aggregate predictions over tracklets rather than single frames.
- [ ] Evaluate similar kits, white-clothing Test cricket, helmets/pads, lighting changes, and officials.
- [ ] Add confidence thresholds and manual correction.
- **Exit gate:** publish accuracy, sample size, confusion matrix, and unknown rate for each supported condition.

### Phase 4 — Cricket field geometry
- [ ] Define cricket-specific landmarks: pitch corners, creases, wickets, boundary cues.
- [ ] Annotate and evaluate landmark detection.
- [ ] Estimate planar homographies for suitable ground views and track calibration quality.
- [ ] Project player footpoints to a field map only when calibration is valid.
- [ ] Mark measurements unavailable when geometry is uncertain; do not infer 3D positions from a planar homography.
- **Exit gate:** measured landmark and ground-position error on held-out calibrated shots.

### Phase 5 — Ball detection and trajectory research
- [ ] Build a ball visibility/annotation protocol; label only visually identifiable ball positions.
- [ ] Benchmark small-object detectors on full frames and high-resolution crops.
- [ ] Add temporal association, uncertainty, missing-observation handling, and outlier rejection.
- [ ] Evaluate trajectory coverage and position error by camera view, resolution, blur, and occlusion.
- [ ] Investigate release, bounce, bat contact, and boundary-crossing candidates.
- **Exit gate:** demonstrate trajectory coverage and error on a labelled dataset before claiming delivery-level accuracy.

### Phase 6 — Cricket event intelligence
- [ ] Build temporal event records with source timestamps and confidence.
- [ ] Infer possible bowler/batter/wicketkeeper roles using temporal context and spatial evidence.
- [ ] Investigate delivery phases, shot direction/zones, fielding actions, and candidate wickets.
- [ ] Separate observed facts from inferred events; allow unknown/ambiguous states.
- [ ] Validate event precision, recall, F1, and timestamp error against independent annotations.
- **Exit gate:** event-level benchmark; no automatic scoring claims until independently validated.

### Phase 7 — Analytics application
- [ ] Build a local dashboard for upload, playback, detections, tracks, field map, and event timeline.
- [ ] Add side-by-side original/annotated video and track inspection.
- [ ] Export JSON, CSV, annotated MP4, and benchmark reports.
- [ ] Add job cancellation, resumable processing, caching, and bounded memory use.
- [ ] Keep inference separate from UI and provide a command-line workflow.
- **Exit gate:** end-to-end local workflow with tests and documented limitations.

### Phase 8 — Performance, release, and community
- [ ] Profile CPU throughput and peak memory on documented hardware.
- [ ] Evaluate optional ONNX Runtime/GPU acceleration without changing output contracts.
- [ ] Add CI, linting, type checks, tests, issue templates, contribution guide, and release notes.
- [ ] Publish model/data cards, provenance, hashes, and third-party notices.
- [ ] Add reproducible benchmark scripts and regression fixtures with redistribution rights.
- [ ] Review security, privacy, and all licences before public model/data distribution.

## Target architecture

```text
Video input
  └─ Decode + timestamps
      └─ Shot boundaries / camera-view classification
          ├─ Person detector ──> Player tracker ──> Team/role inference
          ├─ Ball detector ────> Trajectory estimation ──> Event hypotheses
          └─ Pitch landmarks ──> Calibration ──> Field coordinates
                  └─ Temporal fusion + confidence
                      └─ Annotated video / JSONL / CSV / dashboard
```

Modules should use explicit data contracts and keep image coordinates, ground coordinates, and timestamps distinct. A calibration or tracking failure should be surfaced rather than silently converted into a confident statistic.

## Suggested repository layout

```text
crickvision/
├── crickvision/
│   ├── __init__.py
│   ├── detector.py
│   └── cli.py
├── tests/
├── configs/
├── datasets/
│   ├── manifests/
│   └── annotation_guidelines/
├── benchmarks/
├── runs/                 # generated outputs; not committed
├── requirements.txt
├── .gitignore
├── README.md
└── THIRD_PARTY_NOTICES.md
```

## Dataset and research principles

Potential starting points include cricket broadcast image datasets such as [Cric-360](https://huggingface.co/datasets/sarimshahzad/Cric-360), subject to auditing the exact dataset subset, annotations, access conditions, underlying footage rights, and redistribution terms. Still frames alone do not provide persistent identities or ball trajectories.

Recommended annotation workflow:
1. Define classes and occlusion/visibility rules before labelling.
2. Maintain a source manifest and provenance for every clip.
3. Split by entire match or broadcast source, not neighbouring frames.
4. Keep real and synthetic data results separate.
5. Use active learning to prioritise hard examples.
6. Publish dataset cards and never commit footage without redistribution permission.

## Evaluation policy

| Capability | Metrics |
|---|---|
| Detection | Precision, recall, AP/mAP, size-specific recall |
| Tracking | HOTA, IDF1, ID switches, fragmentation |
| Team classification | Accuracy, confusion matrix, unknown rate |
| Calibration | Landmark error and ground-position error |
| Ball trajectory | Detection recall, trajectory coverage, position error |
| Event detection | Precision, recall, F1, timestamp error |
| Runtime | FPS, latency, peak memory, hardware/software versions |

All results should include dataset size, match-level splits, model/checkpoint hash, configuration, hardware, and limitations. Do not publish target values as achieved results.

## Licensing and model provenance

- CrickVision's own licence has not yet been selected. Choose it after reviewing the complete dependency and distribution plan.
- Torchvision's code licence and the selected pretrained weights' terms are distinct questions; review the applicable PyTorch/Torchvision and COCO asset terms before redistribution.
- Every future detector/tracker, checkpoint, dataset subset, and third-party asset needs its own licence/provenance record.
- Do not commit copyrighted broadcast footage or model weights unless their terms permit it.
- This repository does not bundle training footage or claim a custom-trained cricket model.

## Contributing

Contributions are welcome. Start with issues for dataset auditing, baseline evaluation, tests, tracking, team classification, calibration, and documentation. Please include a reproducible example, environment details, expected behaviour, and measured results for model changes. Never upload private or unlicensed footage.

## Disclaimer

CrickVision is an experimental research/development project. Outputs may be incorrect, incomplete, or unavailable on difficult broadcast views. It is not an official scoring system or a substitute for human match officials.

## Acknowledgements

CrickVision builds on the open-source computer-vision ecosystem. All third-party projects, datasets, and model assets must be credited according to their individual terms.


## Deep research and product specification

The expanded research document defines the product scope, module requirements, architecture, result schemas, dataset strategy, evaluation metrics, licensing risks, and implementation priorities:

- [Read the research and product specification](docs/research-and-product-spec.md)

It distinguishes the current generic person-detection baseline from planned features and sets release gates so we do not claim tracking, team classification, ball trajectories, or accuracy before those capabilities are implemented and benchmarked.
