# Video manifest

Store metadata only; do not commit copyrighted broadcast footage or extracted frames.
Use one JSON object per line. Required fields:

```json
{"clip_id":"clip-0001","source_id":"source-01","match_id":"match-01","video_path":"private-data/clip.mp4","rights_status":"licensed","split":"train","notes":"wide camera"}
```

- `clip_id` must be unique.
- `source_id` and `match_id` let evaluators group related clips.
- `rights_status` is one of `owned`, `licensed`, `research-only`, `unknown`, or `prohibited`.
- `split` is `train`, `validation`, `test`, or `unassigned`.
- Split by whole match/source where possible, never neighbouring frames from the same match across train and test.
- Review dataset terms separately for every subset. A valid manifest is not legal clearance.

Validate with `python -m crickvision.manifest datasets/manifests/manifest.jsonl`.
