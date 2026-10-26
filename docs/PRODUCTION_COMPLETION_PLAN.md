# Production completion plan

This file is the release contract for CrickVision. It deliberately separates implemented behavior from capabilities that still require evidence. A green CI run is necessary but is not proof of cricket-analysis accuracy.

## Current verified scope

The current application is a local, single-user research prototype. The default detector is a general-purpose pretrained person detector. Optional IoU tracking and shot-cut detection are heuristic. Public resource discovery retrieves metadata only; it does not grant dataset rights, download gated data, train a model, or validate model quality.

Do not market this release as a production cricket analytics service, and do not publish benchmark numbers unless they are reproducible from reviewed labels and held-out match footage.

## Release levels

### Level 1 — dependable local research tool

- [ ] A clean install works on supported Python versions and documented operating systems.
- [ ] Invalid, truncated, unsupported, and oversized media fail with clear user-facing messages.
- [ ] A user can process a real video they are authorised to use and download a playable annotated MP4 and parseable JSONL.
- [ ] CSV export is tested against actual output records.
- [ ] Output metadata records the source-video hash, tool version, settings, timestamps, and relevant model/checkpoint identifier.
- [ ] The app explains the limitations of generic person detection, heuristic cuts, and temporary track IDs.
- [ ] No real match footage, restricted frames, private user data, or unapproved model weights are committed.
- [ ] A release SBOM and third-party licence review are attached to the release record.

### Level 2 — cricket-validated analysis

- [ ] Obtain written permission or a clearly applicable licence for source footage and annotation.
- [ ] Publish dataset provenance, annotation instructions, exclusions, and permitted uses.
- [ ] Split training, validation, and test sets by match/source video, not by adjacent frames.
- [ ] Train and version a cricket-specific player detector; report held-out precision, recall, and suitable AP metrics with per-scenario breakdowns.
- [ ] Evaluate tracking against reviewed ground truth using established metrics such as HOTA and IDF1; include camera cuts, occlusion, zoom, and distant players.
- [ ] Annotate visible ball positions and visibility/occlusion states before reporting ball recall, localization error, or trajectory continuity.
- [ ] Compare every claimed improvement against the current baseline using the same held-out clips.
- [ ] Report uncertainty and failure cases. Do not infer speed, spin, delivery type, identity, or official statistics without suitable measurements and validation.

### Level 3 — public multi-user website

- [ ] Select and document a real hosting target and deployed HTTPS URL.
- [ ] Separate web frontend, API, background worker, job queue, and persistent object storage where required by expected workload.
- [ ] Implement authentication, per-user job ownership, upload validation, size/duration limits, quotas, cancellation, retry, retention, and cleanup.
- [ ] Keep credentials server-side; do not put secrets in frontend build variables.
- [ ] Ensure one user's uploads, outputs, logs, and job identifiers cannot be accessed by another user.
- [ ] Add operational health checks, structured logs, error reporting, resource limits, backups where applicable, and a rollback procedure.
- [ ] Run browser-based end-to-end tests against the deployed site, not just local unit tests.
- [ ] Test accessibility, responsive layouts, upload failures, long-running jobs, and concurrent jobs.
- [ ] Publish privacy/data-retention terms that match actual behavior.

## Real-video acceptance record

For each release candidate, preserve a record containing:

1. Permission basis and source URL or private provenance reference (never expose restricted source media).
2. SHA-256 of the source video and its dimensions, frame rate, duration, and codec.
3. CrickVision commit, Python/dependency environment, model/checkpoint hash, and full analysis settings.
4. Command or UI path used, wall-clock runtime, peak memory if measured, and any warnings.
5. Output hashes, MP4 decode/playback result, JSONL parse result, CSV export result, and reviewed sample frames.
6. Human-reviewed ground truth and metric calculation scripts for any accuracy claim.
7. Known failures and the release decision signed off by a reviewer.

Synthetic fixtures may be used for deterministic software regression tests only. They must never be described as real cricket footage, training data, or evidence of cricket accuracy.

## Release decision

A release is not complete while any applicable checkbox above is unchecked. The local prototype can be useful before Levels 2 and 3 are met, but its product page and README must describe the actual local scope and must not imply public hosting or validated cricket intelligence.
