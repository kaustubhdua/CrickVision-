# Changelog

Notable project changes are recorded here. This is not a claim that each feature has passed a real-match benchmark.

## Unreleased

- Added timestamp-based video frame sampling with source SHA-256 provenance and a JSONL manifest for annotation preparation.
- Added sampled-frame manifest validation for path safety, image integrity, dimensions, timestamps, source hashes, and rights-status warnings.
- Added CI smoke checks for the new command-line tools and configured the release workflow to validate build artifacts on pull requests.

## 0.3.0 — 2026-10-20

- Added JSONL detection benchmark with frame-level greedy IoU precision/recall/F1 and explicit reviewed-empty-frame support.
- Added JSONL-to-CSV export.
- Added single-class Faster R-CNN fine-tuning workflow, custom checkpoint loading, and tests for the custom checkpoint path.
- Added manual planar homography calibration with overdetermined residual checks and ground-plane footpoint projection.
- Added opt-in Hough-circle ball-like proposals and nearest-neighbour candidate tracklets for manual review; these are not validated ball detections.
- Added schema version, source video ID, and shot ID to detection and shot-boundary records.
- Strengthened manifest validation, including match-level split leakage checks and clip time ranges.
- Added synthetic end-to-end CLI regression test, benchmark/export/training tests, and expanded CI smoke checks.
- Added original-code MIT licence, model card, data policy, annotation guidelines, release checklist, contributor/security guidance, and issue templates.
- The detector, tracking, team-colour, and shot-boundary outputs remain experimental. No cricket-specific model training or real-match accuracy benchmark is included.
