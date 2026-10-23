# Validate sampled-frame manifests

After extracting frames, validate the manifest and image files before sending them for annotation or converting annotations into training data.

```bash
crickvision-validate-samples datasets/working/match-001/frames.jsonl
```

Optionally verify that the manifest's source hash matches the original video:

```bash
crickvision-validate-samples datasets/working/match-001/frames.jsonl --source-video /path/to/match.mp4
```

The validator checks JSONL structure, schema version, SHA-256 shape and consistency, duplicate frame indices and image paths, finite timestamps, positive dimensions, allowed rights-status values, path traversal, image existence and decodability, and recorded-versus-actual image dimensions. It can skip image decoding for faster path-only checks using `--skip-image-decode`.

A successful validation is a data-integrity check, not legal advice, annotation-quality certification, or proof that the footage can be redistributed. Unknown rights produce a warning; prohibited rights fail validation. Research-only restrictions must be reviewed for the specific intended use.
