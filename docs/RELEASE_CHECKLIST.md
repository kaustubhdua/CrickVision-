# Release checklist

A release may be tagged only when every applicable gate below has evidence attached to the release record.

## Build and quality
- [ ] GitHub Actions passes on Python 3.10, 3.11 and 3.12.
- [ ] Ruff, compilation, pytest, and all installed CLI help commands pass.
- [ ] End-to-end smoke test succeeds on a rights-cleared short video.
- [ ] Annotated output video opens; JSONL is valid; CSV export round-trips.
- [ ] Test dependencies and runtime dependencies are compatible and documented.
- [ ] Create a software bill of materials for the release.

## Data and models
- [ ] Source video rights are recorded and permit the planned use.
- [ ] Dataset manifest validates; train/validation/test splits are match-disjoint.
- [ ] Benchmark annotations were reviewed and held-out test matches were not used for tuning.
- [ ] Model/checkpoint hash, configuration, environment, and training manifest are recorded.
- [ ] Report detection metrics and known failure modes; do not substitute validation loss for held-out accuracy.
- [ ] No raw video, extracted restricted frames, or unapproved checkpoint is in the release bundle.

## Product and security
- [ ] Dashboard limitations and local-only deployment warning are visible.
- [ ] Resource use and invalid input handling have been checked.
- [ ] User documentation matches implemented behavior.
- [ ] Original-code MIT licence and all third-party notices are consistent.
- [ ] No official scoring, player identity, trajectory, or event-accuracy claims exceed the evidence.
