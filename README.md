# 🏏 CrickVision

**A local-first computer-vision toolkit for cricket video intelligence.**  
Version 0.3.0 · [Changelog](CHANGELOG.md) · [Release checklist](docs/RELEASE_CHECKLIST.md)  
*Turn broadcast pixels into structured cricket insights.*

CrickVision is a local-first project to detect and track cricket players, classify teams, map field positions, and progressively analyse the ball and delivery events from video. The long-term goal is a reproducible, well-tested toolkit for developers, researchers, coaches, and cricket analysts.

> **Current status: experimental prototype (v0.3).** The default pipeline runs a general-purpose pretrained person detector, heuristic shot-cut detection, and optional IoU tracking. An optional HSV kit-colour hint can be enabled with user-provided ranges, but it is not a learned classifier and has not been validated on match footage. There is no cricket-trained detector, ball tracking, calibrated field mapping, event intelligence, or measured cricket benchmark. Do not treat outputs as official match statistics.

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
- Configurable confidence threshold, frame sampling, device, and shot-cut threshold.
- JSONL records for person detections and heuristic shot boundaries; optional track IDs are temporary and may switch.
- A local Streamlit dashboard for a single trusted user.
- JSONL dataset-manifest validation, frame-level IoU precision/recall/F1 benchmarking, and CSV export.
- Local inference with no hosted AI API key; the first run may download pretrained weights unless cached.
- A fine-tuning starter workflow for person boxes, custom checkpoint loading, benchmark reports, and CSV exports.

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

The output video shows detections with experimental, temporary IoU track IDs unless `--no-tracking` is passed. IDs can switch under occlusion, camera movement, and missed detections; they are not player identities.

Optional team-colour hints can be enabled with a user-created JSON config. HSV ranges must be tuned for the source video and are **not** a reliable learned team classifier:

```json
{
  "teams": [
    {"name": "Team A", "hsv_lower": [90, 70, 40], "hsv_upper": [130, 255, 255]},
    {"name": "Team B", "hsv_lower": [0, 80, 60], "hsv_upper": [12, 255, 255]}
  ]
}
```

Run with `--team-colors configs/team-colors.json`. Each detection JSONL record then includes `team` and `team_colour_fraction`; ambiguous or weak colour evidence is labelled `Unknown`. The HSV hue channel uses OpenCV's 0–179 range. Do not use these labels as official team identity without manual review.


### Fine-tune a person detector

Once you have a rights-cleared, manually labelled image dataset, prepare a JSONL manifest with one row per image:

```json
{"image_path":"frames/train-001.jpg","split":"train","boxes_xyxy":[[40,25,95,240]]}
{"image_path":"frames/validation-001.jpg","split":"validation","boxes_xyxy":[]}
```

Paths are relative to the manifest. Keep entire matches/sources in one split. Then fine-tune the person detector and use the resulting local checkpoint:

```bash
crickvision-train-detector --manifest datasets/annotations/person-boxes.jsonl --output models/person-finetuned.pt --epochs 5
python -m crickvision.cli --input match.mp4 --output runs/fine-tuned.mp4 --weights models/person-finetuned.pt
```

This training workflow learns the single person class; it does not train a ball detector or infer cricket events. Validation loss is only a training diagnostic. Evaluate the selected checkpoint on a separate held-out match before reporting performance. The initial Torchvision weights may be downloaded when training starts.

## Architecture roadmap

### Phase 0 — Foundations and reproducibility
- [x] Create repository structure and project scope.
- [x] Add generic pretrained person-detection baseline and video CLI.
- [x] Export timestamped detections as JSONL.
- [x] Add automated unit tests and CI configuration (execution still needs a green Actions run).
- [ ] Add dependency lockfile, model provenance, and software/data licence inventory.
- [ ] Establish a small, legally usable, manually labelled cricket benchmark.

### Phase 1 — Cricket video baseline
- [ ] Audit cricket-specific datasets and annotation formats.
- [x] Build a basic video manifest validator (source, match/clip ID, rights status, split); timestamp/resolution enrichment remains future work.
- [ ] Evaluate person recall separately for wide shots, pitch views, close-ups, day/night, and small distant players.
- [x] Add a heuristic shot-boundary detector; shot/view labels and benchmark validation remain open.
- [x] Add a baseline person-detector fine-tuning command and checkpoint loading.
- [ ] Add model configuration and structured logs.
- **Exit gate:** a reproducible benchmark report on held-out matches; no performance claims without measured results.

### Phase 2 — Multi-object player tracking
- [x] Add a lightweight greedy IoU tracking baseline; stronger MOT adapters remain open after licence/dependency review.
- [x] Reset the baseline track state when the shot-cut heuristic flags a camera cut; validate cut/ID behaviour on labelled footage.
- [ ] Record track age, confidence, missing detections, and fragmentation.
- [ ] Evaluate HOTA, IDF1, ID switches, and fragmentation against labelled clips.
- [x] Export JSONL detections and CSV summaries; track visualisation and track metrics remain open.
- **Exit gate:** documented tracking results across multiple broadcast views.

### Phase 3 — Team and official classification
- [x] Add a configurable HSV torso-colour heuristic; track-level aggregation and evaluation remain open.
- [x] Support user-configured HSV team-colour hints and Unknown results; Official/player roles remain unimplemented.
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
- [x] Add an initial local dashboard for video upload, processing, and downloads; playback, field map, event timeline and robust job management remain open.
- [ ] Add side-by-side original/annotated video and track inspection.
- [x] Export JSONL observations, CSV tables, annotated MP4, and JSON benchmark reports.
- [ ] Add job cancellation, resumable processing, caching, and bounded memory use.
- [ ] Keep inference separate from UI and provide a command-line workflow.
- **Exit gate:** end-to-end local workflow with tests and documented limitations.

### Phase 8 — Performance, release, and community
- [ ] Profile CPU throughput and peak memory on documented hardware.
- [ ] Evaluate optional ONNX Runtime/GPU acceleration without changing output contracts.
- [x] Add initial CI and unit tests; linting, type checks, issue templates, contribution guide, and release process remain open.
- [x] Publish a baseline model card, data policy, annotation guidelines, and third-party notices; trained-checkpoint hashes and full SBOM remain pending.
- [x] Add a benchmark runner and synthetic regression fixtures; a rights-cleared real-cricket benchmark remains pending.
- [ ] Complete security/privacy review and audit all third-party, model, dataset, and footage terms before distributing models or data.

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

- The root LICENSE applies MIT to original CrickVision code and project-authored documentation. Third-party code, pretrained weights, datasets, and broadcast footage retain their own terms.
- Torchvision's code licence and the selected pretrained weights' terms are distinct questions; review the applicable PyTorch/Torchvision and COCO asset terms before redistribution.
- Every future detector/tracker, checkpoint, dataset subset, and third-party asset needs its own licence/provenance record.
- Do not commit copyrighted broadcast footage or model weights unless their terms permit it.
- This repository does not bundle training footage or claim a custom-trained cricket model.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and contribution rules. Never upload private or unlicensed footage. Security reporting guidance is in [.github/SECURITY.md](.github/SECURITY.md).

## Disclaimer

CrickVision is an experimental research/development project. Outputs may be incorrect, incomplete, or unavailable on difficult broadcast views. It is not an official scoring system or a substitute for human match officials.

## Acknowledgements

CrickVision builds on the open-source computer-vision ecosystem. All third-party projects, datasets, and model assets must be credited according to their individual terms.


## Deep research and product specification

The expanded research document defines the product scope, module requirements, architecture, result schemas, dataset strategy, evaluation metrics, licensing risks, and implementation priorities. The repository also includes a [model card](docs/model-card.md), [data policy](docs/data-policy.md), and [annotation guidelines](datasets/annotation_guidelines/README.md):

- [Read the research and product specification](docs/research-and-product-spec.md)

It distinguishes the current generic person-detection baseline from planned features and sets release gates so we do not claim tracking, team classification, ball trajectories, or accuracy before those capabilities are implemented and benchmarked.


## Experimental analysis modules (v0.3)

The repository now also includes reusable, explicitly experimental modules:

- `crickvision.tracker.IoUTracker`: greedy intersection-over-union association with track ageing and explicit reset at shot boundaries. IDs are not reliable across occlusion or camera cuts; this is a baseline, not a production MOT algorithm.
- `crickvision.shot.ShotBoundaryDetector`: frame-difference cut heuristic. Fast motion, flashes and replays can cause false positives.
- `crickvision.team.classify_team_colour`: configurable HSV kit-colour hint from a torso crop. Requires user-defined HSV ranges and can return `Unknown`; it is not a learned classifier.

These APIs are intentionally modular and are not yet presented as a validated end-to-end match analytics system. The current default video CLI combines person detection, a shot-cut heuristic, and experimental IoU tracking. Team-colour hints can be integrated with `--team-colors`; the output remains heuristic and unvalidated. See tests for deterministic synthetic examples.

### Development and tests

```bash
python -m pip install -e ".[dev]"
python -m compileall -q crickvision tests
python -m pytest -q
```

GitHub Actions runs the test suite on Python 3.10, 3.11 and 3.12. CI results must be checked on the repository before treating the build as verified.

### Current implementation boundary

Implemented: generic person detector, video annotation CLI, JSONL detections, shot-aware IoU tracking, shot-cut heuristic, configurable HSV team-colour hint, local upload dashboard, manifest validation, frame-level detection metrics, tests and CI configuration.

Still not complete/validated: green CI run and full real-video smoke test; cricket-specific model training on rights-cleared labelled data; ball detection/trajectory; calibrated field mapping; role/event inference; track-level team classification; a legally usable labelled benchmark with measured results; complete third-party licence audit and software bill of materials. No accuracy or match-scoring claims are made.


## Integrated shot-aware analysis

The video CLI now combines person detection with the experimental IoU tracker and frame-difference shot-boundary heuristic. Track IDs are reset when a cut is flagged. The JSONL stream includes `person_detection` and `shot_boundary` record types. Since cut detection is heuristic, inspect results manually and tune `--shot-threshold` for the video source.

```bash
python -m crickvision.cli --input match.mp4 --output runs/annotated.mp4 --every-n-frames 2
```

Use `--no-tracking` to disable tracking or `--shot-threshold 0.35` to adjust the cut heuristic. Track IDs are temporary within a shot and are not player identities.

## Optional local dashboard

```bash
python -m pip install -e ".[dashboard]"
streamlit run crickvision/dashboard.py
```

The dashboard runs processing as a local subprocess. It is intended for a single trusted user, not public multi-user hosting. Large video uploads can consume substantial RAM, disk and compute.

For the current testing instructions and release gates, see [docs/TESTING.md](docs/TESTING.md).


## Dataset governance and evaluation utilities

Before building a benchmark, define a JSONL manifest and validate it:

```bash
crickvision-manifest datasets/manifests/manifest.jsonl
# or: python -m crickvision.manifest datasets/manifests/manifest.jsonl
```

The validator checks required provenance fields, duplicate clip IDs, supported split labels and rights-status warnings. It cannot grant permission to use a dataset; verify source terms independently. See [manifest guidance](datasets/manifests/README.md).

A small IoU-matching precision/recall/F1 helper is available in `crickvision.metrics` for labelled frame-level detections. It is not a substitute for COCO AP/mAP or full MOT metrics. Do not report benchmark scores until labels, match-level splits, and evaluation procedures are independently reviewed.
