# Dataset audit: cricket video and image sources

Last reviewed: 20 October 2026. This is a source-screening note, not legal advice. Re-check each dataset card and licence before acquisition or redistribution.

## Cric-360 v2 — candidate for scene and field research

Source: [Hugging Face dataset card](https://huggingface.co/datasets/sarimshahzad/Cric-360) · [maintainer repository](https://github.com/cogniqube/Cric360-Dataset)

The current card describes 13,545 real broadcast ground frames (v1 + v2), with a separate 6,418-image synthetic subset from the Real Cricket game available on request. The real `grounds` subset is labelled Apache-2.0 by the card; the synthetic `games` subset is explicitly excluded from that licence and restricted to research use. The card also says users must agree to share contact information to access hosted files and notes that underlying broadcast footage remains the property of its rights-holders.

**Recommended disposition**
- Treat the real `grounds` subset as a candidate for ground/scene segmentation, camera-view research, and calibration experiments, subject to reviewing the current terms and provenance for the exact files.
- Do not download or commit the synthetic `games` subset into this repository. It is not Apache-2.0 and has separate research-use terms.
- Do not treat the dataset card alone as legal clearance for every downstream use of broadcast-derived images. Record the exact version, terms, access agreement, file hashes, intended use, and redistribution permissions.
- The described data and metadata are not a ready-made player bounding-box or ball-trajectory dataset. They cannot directly train the current person detector without suitable reviewed annotations.
- Do not assume the published frame-level train/validation/test labels guarantee match-disjoint splits. Inspect source/match provenance and prove split independence before evaluating.

Optional metadata-only inspection after accepting the dataset's access conditions:

```python
from datasets import load_dataset

grounds = load_dataset("sarimshahzad/Cric-360", "grounds", split="train")
print(grounds.features)
print(grounds[0].keys())
```

This snippet is not run by CrickVision, does not bypass access controls, and does not download files into the repository automatically. The `datasets` package is not a core dependency.

## Public cricket-ball detection repositories

A search found repositories describing single-camera ball detection/tracking, including [cricket-ball-detection](https://github.com/Manish9198/cricket-ball-detection). The public README describes a YOLO/Kalman-filter workflow, but the source inspected during this audit did not establish a clear licence grant for all code, weights, or sample assets.

**Disposition:** research reference only. Do not copy its code, checkpoint, or assets into CrickVision until a compatible explicit licence and weight/data provenance have been confirmed. A repository being public is not, by itself, permission to redistribute its contents.

## What CrickVision still needs

1. A rights-cleared dataset of cricket frames with player boxes and visibility/occlusion labels.
2. A separate ball-centre/box annotation protocol that marks unobservable frames explicitly.
3. Match/source-level split identifiers to prevent leakage from adjacent frames or repeated broadcast footage.
4. A held-out, manually reviewed test set and a published benchmark configuration.
5. Source licence, model-weight terms, file hashes, annotation version, and data retention notes for every dataset release.

No images, video, pretrained weights, or third-party repository code were copied into CrickVision as part of this audit. Until the above items are complete, the project remains an experimental baseline rather than a validated cricket analytics system.
