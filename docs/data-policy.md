# Data policy and provenance

CrickVision does not bundle cricket broadcast footage or claim rights to third-party datasets.

## Before acquiring data

- Record source owner, URL or agreement reference, acquisition date, permitted purposes, retention/redistribution terms, and contact for rights questions.
- Review rights for the exact dataset subset and underlying source footage. A dataset card alone may not grant rights to redistribute broadcast content.
- Do not commit raw videos, extracted frames, private data, or restricted model weights to Git.
- Use local paths or controlled storage and keep manifests free of secrets and personal information.

## Manifest

Each JSONL clip record should include clip_id, source_id, match_id, video_path, rights_status, and split. Recommended additional fields: source_url, rights_evidence, acquired_at, clip_start_ms, clip_end_ms, sha256, annotation_version, review_status, width, height, and fps.

Allowed split values are train, validation, test, and unassigned. Split by match or broadcast source to prevent adjacent-frame leakage. Keep synthetic game footage and real broadcast footage in separate benchmark reports.

## Annotation and evaluation

Use the annotation guidelines in datasets/annotation_guidelines/README.md. Keep reviewer instructions and annotation versions with the dataset. Do not publish metrics from test clips used to tune thresholds. Mark empty frames explicitly in benchmark ground truth with empty_frame: true after review.

## Retention and deletion

The repository stores code and metadata only. Users remain responsible for securing their local video assets, observing applicable retention obligations, and deleting data when their rights or purpose require it.
