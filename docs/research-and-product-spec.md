# CrickVision: Research Findings and Product Specification

**Research snapshot:** 19 October 2026  
**Status:** Engineering plan, not a claim that every component has been implemented or benchmarked.

This document translates the research into a practical product definition: what CrickVision should do, how modules fit together, what data is required, how to evaluate results, and which decisions should be made before training specialised models.

## 1. Product definition

CrickVision should be a local-first, open-source cricket video understanding toolkit. Its first job is not to predict a score or replace an umpire. It should turn a video into inspectable, timestamped visual observations and then build cricket-specific interpretations on top.

### Product modes

1. **Video analysis:** process an uploaded local video, render detections/tracks, inspect uncertain moments, export structured data.
2. **Dataset lab:** manage clip manifests, annotations, train/validation/test splits, model runs and failure cases.
3. **Benchmark mode:** evaluate a model/tracker against labelled ground truth and produce a reproducible report.
4. **Research mode:** experiment with ball trajectories, pitch geometry and event hypotheses without representing them as production-ready features.

### Design principles

- Local-first; no hosted AI API is required for the core pipeline.
- Modular model interfaces; detectors and trackers can be swapped.
- Every output retains source timestamp, confidence, model/config version and coordinate system.
- Unknown is a valid result. Never force a team, role, ball position or event when evidence is weak.
- Separate observation from inference. A visible bounding box is an observation; “bowler released the ball” is a temporal hypothesis.
- Do not claim accuracy or real-time performance without a published benchmark.
- Do not bundle or redistribute footage, datasets or weights without rights and licence review.

## 2. Scope and release boundaries

### MVP: make this reliable first

- Read local MP4/MOV files supported by the installed decoder.
- Decode frames once and preserve source timestamps.
- Detect people with a baseline model.
- Render annotated video and export JSONL/CSV detections.
- Add shot-boundary-aware track IDs and baseline multi-object tracking.
- Add a review UI or CLI summaries.
- Include tests, sample metadata (not copyrighted footage), a reproducible environment and benchmark protocol.

### Next release

- Cricket-specific player fine-tuning.
- Team A / Team B / Official / Unknown classification using temporal aggregation.
- Shot types and camera-view classification.
- Field calibration on selected wide shots.
- Track review and corrections.
- Evaluation reports across held-out matches.

### Research milestones (not MVP promises)

- Small-ball detection and continuous trajectory estimation.
- Delivery release/bounce/contact hypotheses.
- Batter/bowler/wicketkeeper role inference.
- Shot direction, fielding events, catch/run-out candidates.
- Physical distances/speeds, only where calibration and assumptions justify them.
- Live-stream processing and GPU acceleration.

## 3. Recommended architecture

    Input video + metadata
      -> Decoder / timestamp normaliser
      -> Shot boundary detector + view classifier
      -> Perception branches
           People detector -> Multi-object tracker -> Tracklet appearance / team labels
           Ball detector   -> Candidate association  -> Trajectory + uncertainty
           Pitch/landmarks -> Calibration quality     -> Ground-plane coordinates
      -> Temporal fusion / event hypotheses
      -> Results store
      -> Annotated video + JSONL/CSV + review UI + benchmark report

### Coordinate and time contracts

Keep these concepts separate:

- **Image coordinates:** pixel-space boxes and points at original source resolution.
- **Ground-plane coordinates:** calibrated 2D points on the playable surface, valid only when camera/landmark calibration is valid.
- **World/3D estimates:** not inferred from a planar homography alone.
- **Time:** source timestamp in milliseconds plus frame index; do not use processed-frame count as the only clock.
- **Shot ID:** a broadcast segment with its own camera state and local track IDs.
- **Track ID:** stable within the supported continuity window; identity across cuts must be a separate, confidence-scored re-identification decision.

All records should include schema version and enough provenance to reproduce the result.

### Suggested result records

Detection fields:
- schema_version, video_id, frame_index, timestamp_ms, shot_id
- class_name, bbox_xyxy, confidence, model_id

Track observation:
- detection reference, track_id, track_confidence, is_interpolated, association_method

Team/role hypothesis:
- label, confidence, evidence_window, status (observed/inferred/unknown)

Calibration:
- camera_view, landmarks, homography, reprojection_error, valid, reason

Event hypothesis:
- event_type, start_ms, end_ms, confidence, evidence_refs, review_status

## 4. Module-by-module requirements

### A. Video ingestion and shot boundaries

**Why:** broadcast cricket is edited. Wide field shots, bowler run-ups, delivery views, close-ups, replays and crowd shots are not a single continuous camera.

Build:
- robust open/read error handling;
- frame dimensions, FPS and timestamps;
- frame sampling with original timestamp preservation;
- cut detection and manual correction;
- optional view categories: wide field, pitch-centred, bowler run-up, batter close-up, replay, crowd/other;
- output and processing manifests.

Use [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) as a candidate baseline, then evaluate it on cricket broadcasts. Shot boundaries should prevent accidental tracker continuity; cross-shot identity matching is a separate later stage.

### B. Player detection

The current baseline uses Torchvision Faster R-CNN pretrained on COCO and detects the generic person class. It is a useful plumbing baseline, not a cricket specialist.

Evaluate at least two candidate detector approaches on the same held-out frames:
- Torchvision Faster R-CNN baseline;
- a compact detector with documented licence/weights (for example, an RF-DETR distribution whose exact variant and checkpoint terms have been verified);
- optionally an Ultralytics detector only if the project explicitly accepts its AGPL obligations or obtains appropriate licensing.

Measure recall by object size and view. The most important cricket-specific failure is missing distant fielders in wide shots. Also measure false positives from staff, spectators, shadows and advertising boards.

Do not pick a model based only on generic COCO mAP. Evaluate actual cricket frames, CPU throughput, peak memory, ease of training, checkpoint availability and licence constraints.

### C. Multi-object tracking

Start with a detector-agnostic tracker interface. Benchmark [ByteTrack](https://github.com/FoundationVision/ByteTrack) and/or [BoT-SORT](https://github.com/NirAharon/BoT-SORT), after reviewing dependencies and implementation details.

Requirements:
- per-shot IDs;
- explicit track birth/death and missed detections;
- camera-motion compensation where useful;
- appearance/ReID as an optional later component;
- no automatic ID continuity across hard cuts;
- tracklet stitching only when evidence is sufficient;
- export identity switches and track fragmentation for evaluation.

Use HOTA alongside IDF1 and ID-switch/fragmentation counts. HOTA is useful because it jointly reflects detection, localisation and association rather than rewarding only one aspect.

### D. Team, official and role classification

Team classification should use multiple observations per tracklet, not a single torso crop.

Initial method:
1. extract torso crops;
2. mask or down-weight background pixels;
3. compute robust colour features (HSV/Lab are candidates);
4. cluster likely team groups within a compatible wide shot;
5. aggregate across the tracklet;
6. allow user-supplied kit colours and manual correction;
7. output Team A, Team B, Official or Unknown with confidence.

Limitations to test: Test cricket whites, similar team colours, helmets and pads, coloured training/alternate kits, lighting, shadows, partial occlusion, close-ups and officials.

Roles such as batter, bowler and wicketkeeper are a different problem from team classification. Infer roles from pitch-relative position, temporal movement, equipment and event context; do not create a hard role label from one frame unless it is visibly unambiguous.

### E. Pitch and field geometry

Use [Cric-360](https://huggingface.co/datasets/sarimshahzad/Cric-360) as a candidate resource for ground/pitch segmentation and calibration experiments. Audit its exact subset, annotation coverage, access conditions and rights before use. Its dataset card distinguishes real broadcast frames from a synthetic game subset with different terms; do not assume all subsets share a licence.

Build:
- pitch/ground segmentation baseline;
- landmark annotation guidelines for pitch corners, creases and wickets;
- landmark detector and calibration-quality scoring;
- robust homography estimation for planar ground points;
- player footpoint projection only on valid calibrated views;
- uncertainty and invalidation on camera movement/cuts.

A homography maps one plane to another. It does not recover the 3D path of an airborne ball, and a pitch-only calibration does not automatically produce accurate coordinates across the whole oval. Keep image-space results available when calibration fails.

### F. Ball detection and trajectory

Treat ball tracking as a separate specialist research track. The ball may be only a few pixels wide, blurred, occluded or absent from the frame. A generic object detector and a tracker cannot guarantee continuity when visual evidence is missing.

Build in this order:
1. annotate visible ball centres/boxes and visibility/occlusion states;
2. measure ball visibility by camera view and resolution;
3. benchmark high-resolution crops and full-frame small-object detection;
4. associate candidates using time, motion and uncertainty;
5. preserve gaps and multiple candidate paths rather than fabricating observations;
6. investigate release, bounce, bat contact and boundary crossing as separate event hypotheses;
7. validate trajectory coverage and position error on held-out deliveries.

Physical speed, bounce point and 3D trajectory require calibrated geometry and defensible assumptions. Do not infer them from pixel movement alone.

### G. Event intelligence

Use a temporal state machine or sequence model, but allow events to be missing, ambiguous or hidden by editing. Potential stages include pre-delivery, approach, possible release, flight, possible bounce/contact, fielding action and dead-ball interval.

Every event should include evidence references and confidence. Begin with human-reviewable proposals; do not claim automatic scoring, wicket decisions or official ball-by-ball truth without an event-level benchmark.

### H. User interface and exports

First UI can be a local Streamlit app or a CLI-first workflow. The UI should not own the model logic.

Minimum panels:
- source video and processing configuration;
- original/annotated video;
- detection/track inspector;
- timeline and shot boundaries;
- calibration status and field map;
- uncertain/failed frames review queue;
- export and benchmark report.

Exports: JSONL for frame-level records, CSV for tabular summaries, annotated MP4 for visual review, and a machine-readable run manifest.

## 5. Dataset plan

### Candidate resources

- **Cric-360 real broadcast frames:** [dataset card](https://huggingface.co/datasets/sarimshahzad/Cric-360) and [project repository](https://github.com/cogniqube/Cric360-Dataset). Useful for ground/pitch segmentation and scene/calibration work; verify labels and terms for each subset.
- **CVAT:** [repository](https://github.com/cvat-ai/cvat), a candidate for video bounding boxes, track IDs, masks and review workflows.
- **Custom short clips:** necessary for persistent player identities, camera-cut transitions, ball trajectories and event labels unless an existing dataset explicitly supplies those annotations under usable terms.

### Annotation protocol

Define before annotation:
- classes and what counts as visible;
- bounding-box convention;
- occlusion/truncation rules;
- unknown/ignore labels;
- track identity policy across cuts;
- ball centre/box and visibility;
- pitch landmark definitions;
- event timestamps and evidence requirements;
- reviewer disagreement resolution.

### Split policy

Split by entire match/broadcast source, not random adjacent frames. Keep train, validation and test matches separate. Keep synthetic and real results separate. Record source URL/owner, acquisition date, permission/licence, clip timestamps, hash, annotation version, and split in a manifest.

Do not commit raw broadcast footage or restricted dataset subsets to the repository. Store download instructions and manifests instead.

## 6. Evaluation and release gates

| Capability | Metrics / slices |
|---|---|
| Person detector | precision, recall, AP/mAP; small/medium/large objects; wide/pitch/close-up |
| Tracker | HOTA, IDF1, ID switches, fragmentation, track coverage |
| Team labels | accuracy, confusion matrix, unknown rate, per-kit/view performance |
| Landmark/calibration | landmark error, reprojection error, ground-position error where ground truth exists |
| Ball | visible-ball recall, trajectory coverage, position error, false trajectory rate |
| Events | precision, recall, F1, start/end timestamp error, abstention rate |
| Runtime | FPS, latency, peak RAM/VRAM, resolution, hardware, software versions |

Each benchmark report must contain dataset version, number of matches/clips/frames, match-level split policy, model and checkpoint hash, configuration, hardware, and known limitations. Targets are not results. No accuracy badges until the CI/release process can reproduce the metric.

## 7. Licence and data-rights strategy

This project has not yet committed to a top-level licence. Decide after reviewing dependencies and the intended distribution model.

- Torchvision code, pretrained checkpoints, and underlying training data may have distinct terms; review the exact model/weights page and notices.
- Ultralytics states its default AGPL-3.0/Enterprise licensing distinction for its code and models. Do not add it casually if the project's intended licence or deployment model is undecided.
- ByteTrack, BoT-SORT, CVAT, MMTracking and other components need exact version/dependency review, not just a glance at the top-level licence.
- Dataset licence labels do not necessarily grant rights to redistribute underlying broadcast content. Verify source footage rights and subset-specific terms.
- Add a dependency SBOM, model card, data card, checkpoint hashes and third-party notices before release.

This is an engineering checklist, not legal advice.

## 8. Implementation priority

### P0 — Make the current baseline trustworthy
1. Run the CLI on a short, permission-cleared clip.
2. Fix all dependency and runtime issues.
3. Add unit tests for confidence filtering, coordinates, JSONL schema and invalid video handling.
4. Add integration test with a tiny generated fixture video; do not commit copyrighted sample footage.
5. Add CI, linting, type checking and pinned/compatible dependency ranges.
6. Publish the known limitations and model provenance.

### P1 — Create the first benchmark
1. Choose and document a small set of legally usable cricket clips/frames.
2. Label person boxes and shot boundaries.
3. Create match-level splits.
4. Compare the baseline with at least one alternative.
5. Publish recall by view/object size and runtime.

### P2 — Add tracking
1. Define the tracker interface.
2. Integrate ByteTrack or BoT-SORT behind the interface.
3. Reset IDs at shot boundaries.
4. Add HOTA/IDF1/ID-switch evaluation.
5. Review track fragments visually.

### P3 — Add team classification
1. Implement torso crop extraction and colour baseline.
2. Add tracklet-level temporal voting.
3. Add unknown/official categories and manual correction.
4. Benchmark multiple kit types and Test cricket whites.

### P4 — Geometry and ball research
1. Audit Cric-360 subset annotations/terms.
2. Add pitch/ground segmentation and landmarks.
3. Validate calibration only on suitable views.
4. Create a ball visibility dataset.
5. Build and benchmark ball candidate detection before event inference.

### P5 — Productise
Add local dashboard, job management, exports, packaging, contributor documentation, release process and model/data cards after the inference pipeline and benchmark are reproducible.

## 9. Recommended repository growth

    crickvision/
      video/          decoding, timestamps, shot boundaries
      detection/      detector interface, person, ball, pitch
      tracking/       tracker adapters, tracklet stitching
      teams/          kit features, team clustering, unknown handling
      calibration/    landmarks, homography, calibration quality
      events/         temporal hypotheses and event schemas
      analytics/      field maps and validated metrics
      rendering/      annotations and video output
      schemas/        versioned result contracts
      evaluation/     metrics, benchmark runner, reports
      app/            local review UI
    configs/
    datasets/
      manifests/
      annotation_guidelines/
    benchmarks/
    tests/
      unit/
      integration/
      regression/
    docs/
      research-and-product-spec.md
      data-policy.md
      model-card.md
      annotation-guide.md

This is a target structure. Move modules into it when the corresponding feature exists; avoid creating empty abstractions that do not have a working use case.

## 10. Primary references

- [Cric-360 dataset card](https://huggingface.co/datasets/sarimshahzad/Cric-360)
- [Cric360-Dataset repository](https://github.com/cogniqube/Cric360-Dataset)
- [CVAT Community](https://github.com/cvat-ai/cvat)
- [ByteTrack](https://github.com/FoundationVision/ByteTrack)
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
- [MMTracking](https://github.com/open-mmlab/mmtracking)
- [OpenCV homography tutorial](https://docs.opencv.org/5.0/tutorials/features/homography/homography.html)
- [HOTA tracking metric paper](https://doi.org/10.1007/s11263-020-01375-2)
- [Torchvision Faster R-CNN documentation](https://docs.pytorch.org/vision/stable/models/faster_rcnn)
- [Ultralytics licensing](https://www.ultralytics.com/license)
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)

## Final recommendation

CrickVision's strongest first deliverable is a **reproducible player detection + shot-aware tracking + team classification benchmark**, not a broad promise of automatic cricket analysis. It establishes the video/data/evaluation foundation needed by every later feature. Ball trajectories, calibrated field metrics and event intelligence should be added only when their own datasets and release gates are met.
