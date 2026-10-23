# CrickVision implementation status

Last reviewed: 23 October 2026  
Package version: 0.3.0

## Implemented in the repository

- Torchvision COCO-pretrained Faster R-CNN person-detection baseline.
- CLI for local video annotation, timestamped JSONL, heuristic shot boundaries, greedy IoU tracklets, optional HSV team-colour hints, and optional manual planar homography projection.
- Fine-tuning starter for a single person class from user-labelled image boxes; custom checkpoint inference.
- Dataset-manifest validation including duplicate clip IDs, clip-time checks, and match-level split leakage checks.
- Provenance-aware video frame sampler: configurable time-interval extraction, JPEG frames, source SHA-256, decoder timestamp/frame index, dimensions, user-supplied rights status, and JSONL manifest. The tool does not verify legal rights or label objects.
- Sample-manifest validator: checks schema, source hashes, duplicate frames/paths, timestamps, path traversal, image existence/decodability/dimensions, and rights metadata; optional source-video hash verification.
- Training-manifest preflight audit (`crickvision-audit-dataset`): validates JSONL structure, safe image paths, duplicate images, match-disjoint splits, decoded image bounds, box geometry, and rights metadata warnings/errors.
- JSONL frame-level detection benchmark with reviewed empty-frame support and CSV export.
- Local single-user Streamlit dashboard.
- Unit tests, synthetic CLI integration test, CI matrix, Ruff checks, contributor/security templates, model card, data policy, annotation guide, release checklist, and MIT licence for original source code.
- Release workflow builds wheel/source distributions, smoke-tests installed CLIs, and generates a CycloneDX environment SBOM as a release artifact. SBOM generation does not replace human review of licences or dependency advisories.

## Not yet implemented or verified

- No cricket-specific checkpoint has been trained because no rights-cleared labelled cricket dataset was supplied or added.
- No trained or validated ball detector or trajectory reconstruction. An opt-in Hough-circle candidate proposer and nearest-neighbour candidate tracklets are available as an annotation aid only.
- No delivery/event inference, automatic pitch/landmark detector, role classifier, or cross-shot identity re-identification.
- No track-level team aggregation or manual correction UI.
- No measured benchmark on real cricket footage; synthetic tests only test software behavior.
- CI passed on Python 3.10, 3.11, and 3.12 for the training-manifest audit PR ([run 40026125937](https://github.com/kaustubhdua/CrickVision-/actions/runs/40026125937)); the release build, installed-wheel CLI checks, and CycloneDX SBOM generation passed ([run 40026125916](https://github.com/kaustubhdua/CrickVision-/actions/runs/40026125916)). The post-merge main-branch CI run is queued and must be verified separately.
- No full end-to-end run has been verified against a rights-cleared real cricket clip.
- Third-party model/data/footage licence audit and human review of the generated software bill of materials remain release gates.

## Recommended next actions

1. Verify the post-merge CI run on main and keep CI green for subsequent changes.
2. Run a short, rights-cleared cricket clip through the CLI and dashboard.
3. Use the frame sampler and manifest validator to prepare a small, reviewed, match-disjoint dataset and run the detection benchmark.
4. Fine-tune a person detector only after confirming data and pretrained-weight terms.
5. Create a separate ball annotation/benchmark effort before adding a ball model.
6. Add tracking, calibration, team and event metrics only when corresponding ground truth exists.
7. Review the generated CycloneDX SBOM and complete the model/data/footage licence audit before release.

Do not represent the repository as a production cricket analytics system until the release checklist has evidence for the relevant features.
