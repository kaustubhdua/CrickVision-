# Frame sampling for dataset preparation

CrickVision includes a local frame sampler to help prepare footage for manual annotation. It extracts timestamped JPEG frames and writes a provenance-bearing `frames.jsonl` manifest. It does **not** label objects, validate legal rights, or make the source video redistributable.

## Example

```bash
crickvision-sample-frames \
  --input /path/to/rights-cleared-match.mp4 \
  --output-dir datasets/working/match-001-frames \
  --every-seconds 1.0 \
  --video-id match-001 \
  --rights-status licensed
```

Use `--max-frames 100` for a small pilot. The output directory must be empty to avoid accidental overwrite. The default interval is two seconds. Sampling is based on decoder timestamps where available and falls back to frame-index/FPS timestamps.

## Output

Each JSONL record contains the source video ID and filename, source SHA-256, decoded frame index, timestamp in milliseconds, relative image filename, dimensions, sampling interval and the rights status supplied by the user. The rights status is metadata only: `unknown` is the safe default, and selecting `licensed` does not verify a licence.

The manifest is intentionally distinct from the detector training annotation manifest. Annotate the extracted frames with a video annotation tool, then export/convert reviewed boxes into the training JSONL schema documented in [SCHEMA.md](SCHEMA.md). Keep match IDs and split assignments in the dataset-level manifest; split by match, not by frame.

## Recommended safeguards

- Only process footage you are authorised to use.
- Keep an untouched copy of the source video and its SHA-256.
- Review the extracted sample for cuts, replays, close-ups, lighting changes, distant players and occlusion.
- Do not place private or restricted footage in public Git commits.
- Store annotation instructions, reviewer, review date and annotation version alongside the dataset.
- Use separate match-disjoint train, validation and test splits. Never tune on the test split.

## Limitations

- Frames are sampled for annotation efficiency; unsampled frames are not labelled.
- A decoder's timestamps can be imperfect for variable-frame-rate footage.
- The sampler does not detect shot boundaries, identify players or interpolate skipped frames.
- The generated manifest does not prove ownership or licensing.
