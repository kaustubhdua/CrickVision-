# Training-manifest preflight audit

Run this before fine-tuning a person detector:

```bash
crickvision-audit-dataset datasets/annotations/person-boxes.jsonl
```

The command emits a JSON report and exits non-zero if it finds errors. It checks JSONL syntax, required row fields, match-disjoint train/validation/test splits, duplicate image paths, relative paths staying inside the manifest directory, image existence/decodability, finite positive-area boxes, and box bounds against actual image dimensions. An empty `boxes_xyxy: []` is valid for a manually reviewed frame with no visible people.

Optional `rights_status` values are metadata only. `prohibited` fails validation; `unknown` and `research-only` produce warnings that require human review. The tool cannot establish ownership, licence validity, annotation quality, whether empty frames were truly reviewed, or whether two different IDs refer to the same match. Review the report and source permissions before training.

`--skip-image-check` skips file existence, decoding and coordinate-bound checks, but still checks manifest structure and split integrity. Do not use this option as release evidence.
