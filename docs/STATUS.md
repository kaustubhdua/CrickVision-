# CrickVision implementation status

Last reviewed: 20 October 2026  
Package version: 0.3.0

## Implemented in the repository

- Torchvision COCO-pretrained Faster R-CNN person-detection baseline.
- CLI for local video annotation, timestamped JSONL, heuristic shot boundaries, greedy IoU tracklets, optional HSV team-colour hints, and optional manual planar homography projection.
- Fine-tuning starter for a single person class from user-labelled image boxes; custom checkpoint inference.
- Dataset-manifest validation including duplicate clip IDs, clip-time checks, and match-level split leakage checks.
- JSONL frame-level detection benchmark with reviewed empty-frame support and CSV export.
- Local single-user Streamlit dashboard.
- Unit tests, synthetic CLI integration test, CI matrix, Ruff checks, contributor/security templates, model card, data policy, annotation guide, release checklist, and MIT licence for original source code.

## Not yet implemented or verified

- No cricket-specific checkpoint has been trained because no rights-cleared labelled cricket dataset was supplied or added.
- No trained or validated ball detector or trajectory reconstruction. An opt-in Hough-circle candidate proposer and nearest-neighbour candidate tracklets are available as an annotation aid only.
- No delivery/event inference, automatic pitch/landmark detector, role classifier, or cross-shot identity re-identification.
- No track-level team aggregation or manual correction UI.
- No measured benchmark on real cricket footage; synthetic tests only test software behavior.
- No confirmed green CI run. The connected GitHub status response returned no check statuses/workflow runs, so the current branch's tests must be verified in GitHub Actions.
- No full end-to-end run has been verified against a rights-cleared real cricket clip.
- Third-party model/data/footage licence audit and a software bill of materials remain release gates.

## Recommended next actions

1. Run GitHub Actions on main and fix any lint/test/build failures.
2. Run a short, rights-cleared cricket clip through the CLI and dashboard.
3. Label a small match-disjoint dataset and run the benchmark tool.
4. Fine-tune a person detector only after confirming data and pretrained-weight terms.
5. Create a separate ball annotation/benchmark effort before adding a ball model.
6. Add tracking, calibration, team and event metrics only when corresponding ground truth exists.

Do not represent the repository as a production cricket analytics system until the release checklist has evidence for the relevant features.
