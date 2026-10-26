# 🏏 CrickVision

**A local-first computer-vision toolkit for cricket video intelligence.**  
Version 0.3.0 · [Changelog](CHANGELOG.md) · [Release checklist](docs/RELEASE_CHECKLIST.md) · [Production completion plan](docs/PRODUCTION_COMPLETION_PLAN.md)  
*Turn broadcast pixels into structured cricket insights.*

CrickVision is a local-first project to detect and track cricket players, classify teams, map field positions, and progressively analyse the ball and delivery events from video. The long-term goal is a reproducible, well-tested toolkit for developers, researchers, coaches, and cricket analysts.

> **Current status: experimental prototype (v0.3).** The default pipeline runs a general-purpose pretrained person detector, heuristic shot-cut detection, and optional IoU tracking. An optional HSV kit-colour hint can be enabled with user-provided ranges, but it is not a learned classifier and has not been validated on match footage. There is no cricket-trained checkpoint, reliable/validated ball tracking, automatic landmark detection, validated field mapping, event intelligence, or measured cricket benchmark. An optional Hough-circle candidate proposal mode exists for manual review. Manual planar homography projection is available when the user supplies valid point correspondences. Do not treat outputs as official match statistics.

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
- JSONL dataset-manifest validation, frame-level IoU precision/recall/F1 benchmarking, CSV export, a provenance-aware video frame sampler for annotation preparation, and a sampled-frame manifest validator.
- Optional manual planar homography calibration to project person-box footpoints into user-defined ground coordinates.
- Opt-in Hough-circle ball-like candidate proposals and nearest-neighbour candidate tracklets, explicitly marked unverified.
- Local inference with no hosted AI API key; the first run may download pretrained weights unless cached.
- A fine-tuning starter workflow for person boxes, custom checkpoint loading, benchmark reports, and CSV exports.
- Public metadata discovery for candidate cricket datasets, models and reference repositories with no API key.
- A scheduled GitHub Actions workflow that creates a resource discovery report artifact every Monday and can also be run manually.
- A synthetic end-to-end pipeline test that verifies annotated MP4 and JSONL output without downloading model weights or using match footage.

The baseline detects the generic COCO `person` class. Cricket-specific accuracy has not been established.

## Discover open-source resources

Run the metadata scout:

```bash
crickvision-research
crickvision-research --limit 20 --json > resource-scout.json
```

The local Streamlit dashboard also offers a **Discover open-source datasets, models and reference projects** panel. The scout queries public Hugging Face and GitHub metadata and continues if one source is unavailable. It does not download or execute third-party assets. Review licence terms, dataset labels, model provenance, and underlying footage rights before using any result. Details: [Online resource discovery](docs/ONLINE_RESOURCE_DISCOVERY.md).

The scheduled report is available from the GitHub Actions run's artifact section: [Resource discovery workflow](.github/workflows/resource-discovery.yml).

## Prepare frames for manual annotation

Extract timestamped frames and a SHA-256 provenance manifest from footage you are authorised to use:

```bash
crickvision-sample-frames --input path/to/match.mp4 --output-dir datasets/working/match-001 --every-seconds 1 --video-id match-001 --rights-status licensed
```

The output directory must be empty. The sampler writes JPEGs and `frames.jsonl`; the rights field is a declaration supplied by the user, not an automated legal determination. Validate the sample manifest and review the underlying terms before training.

## Integrated shot-aware analysis

The video CLI combines person detection with the experimental IoU tracker and frame-difference shot-boundary heuristic. Track IDs are reset when a cut is flagged. The JSONL stream includes `person_detection` and `shot_boundary` record types. Since cut detection is heuristic, inspect results manually and tune `--shot-threshold` for the video source.

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

For the current testing instructions and release gates, see [docs/TESTING.md](docs/TESTING.md). JSONL record contracts and benchmark annotation examples are documented in [docs/SCHEMA.md](docs/SCHEMA.md); dataset source suitability and licence gates are in [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md).

## Dataset governance and evaluation utilities

Before building a benchmark, define a JSONL manifest and validate it:

```bash
crickvision-manifest datasets/manifests/manifest.jsonl
```

The validator checks required provenance fields, duplicate clip IDs, supported split labels and rights-status warnings. It cannot grant permission to use a dataset; verify source terms independently. See [manifest guidance](datasets/manifests/README.md).

A small IoU-matching precision/recall/F1 helper is available for labelled frame-level detections. It is not a substitute for COCO AP/mAP or full MOT metrics. Do not report benchmark scores until labels, match-level splits, and evaluation procedures are independently reviewed.

## Product limitations

This is a local research toolkit, not yet a production-grade hosted website or validated cricket analytics service. The default detector is generic, ball candidates are experimental, and tracking IDs can switch. A complete accuracy claim requires a compatible rights-cleared labelled dataset, a cricket-specific trained model, standard held-out evaluation, and real-video acceptance evidence. See [release checklist](docs/RELEASE_CHECKLIST.md) and [status report](docs/STATUS.md).
