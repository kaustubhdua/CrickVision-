# Real cricket data: source and rights review

**Review date:** 4 November 2026  
**Purpose:** identify real, non-synthetic cricket footage/frames that may support CrickVision research. This is a technical source review, not legal advice or a substitute for checking the actual terms and obtaining permission where required.

## Decision summary

No external media is copied into this repository by this review. No dataset is automatically approved for commercial redistribution, public model-weight release, or inclusion in a public benchmark simply because a repository or dataset card displays an open licence.

### Candidate A — Cric-360, real `grounds` subset

- Dataset card: https://huggingface.co/datasets/sarimshahzad/Cric-360
- Maintainer documentation: https://github.com/Cogni-Qube/Cric360-Dataset
- The current card describes 13,545 real broadcast frames in the `grounds` subset (v1 + v2) and a separate 6,418-image `games` subset.
- The `grounds` subset is described as real TV-broadcast frames and is labelled Apache-2.0 in the card. It requires accepting Hugging Face access conditions before files are available.
- The `games` subset is explicitly synthetic Real Cricket game imagery, research-use only, and excluded from Apache-2.0. **CrickVision's real-footage training/evaluation pipeline must exclude it.**
- The dataset documentation says underlying broadcast footage remains the property of respective rights-holders. The maintainer's Apache-2.0 statement and this underlying-rights caveat must be reconciled for the exact intended use with the dataset maintainer and, where needed, rights-holders.
- The card lists recommended train/validation/test partitions (10,505 / 1,497 / 1,543 frames). These are frame counts, not proof that the partitions are match-disjoint. Inspect the metadata and grouping before using them; do not assume no match/source leakage.
- Labels and tasks described include ground segmentation, venue/ground metadata and scene context. Do not assume the card supplies complete person bounding boxes, player identity, ball coordinates or tracking IDs suitable for CrickVision's detector/tracker benchmarks.

**Status: candidate for a controlled review, not yet approved for model training or redistribution.** First inspect access terms, exact files, metadata, label schema, provenance and source grouping. Seek written clarification from the maintainer about commercial model training, checkpoint distribution and the broadcast-rights caveat. Keep the `games` subset out.

### Candidate B — UVY / UVY-Track

- Zenodo record: https://zenodo.org/records/21303900
- DOI: 10.5281/zenodo.21303900
- The Zenodo record describes 20 user-generated sports videos across seven sports, including cricket, with 31,961 annotated frames, 128,258 boxes and 378 tracks.
- The record says source videos were collected under Creative Commons (CC BY) licences and annotations were manually reviewed after a semi-automated detection pipeline.
- The archive is approximately 3.3 GB.
- The Zenodo page's visible Rights/License section did not expose a specific licence value during this review. The source videos' CC BY statement does not by itself establish the licence for the compiled annotations, archive, scripts, or every embedded item.

**Status: promising real-video tracking evaluation candidate, but hold pending licence and per-video provenance verification.** Inspect the archive manifest and every video's original YouTube source, uploader, exact CC licence/version, attribution requirements, modifications, and annotation licence. Confirm whether model training, derived weights, redistribution, and commercial use are permitted.

### Candidate C — IPL Player Detection dataset (IITB PML)

- Dataset card: https://huggingface.co/datasets/goyaljai/IPL-Player-Detection-IITB-PML
- The card describes 1,005 IPL broadcast images, 8×8 grid team annotations and player counts.
- It may support exploratory team-presence/counting research, but the visible card does not establish a complete player bounding-box benchmark or a clear licence in the reviewed summary.

**Status: not approved for training or redistribution until licence and annotation suitability are confirmed.** Grid-level team labels/player counts are not a substitute for instance boxes or track identities.

## Dataset acceptance procedure

Before importing any source:

1. Save the exact source URL, access date, version/revision, archive checksum and dataset-card snapshot.
2. Record the licence for the dataset container, annotations, source media and any third-party content separately.
3. Obtain written clarification for ambiguous rights, especially broadcast-derived imagery and model-weight redistribution.
4. Inspect actual annotation fields and a sample of labels; document classes, coordinate conventions, frame rates, missing labels and known annotation errors.
5. Recover source video / match / event identifiers and construct match-disjoint train, validation and test splits. Do not split adjacent frames randomly.
6. Run duplicate and near-duplicate checks across splits.
7. Store only permitted data. Keep restricted source footage and extracted frames outside this public repository.
8. Record exclusions and uncertainty. If rights or provenance cannot be verified, do not train on or redistribute the material.

## What is still needed for CrickVision

The candidate sources do not currently establish all required labels. A real cricket player detector needs reviewed player boxes across wide, close, low-light, occluded and boundary views. A tracker needs persistent object IDs over sequences. A ball model needs ball centre/box labels plus explicit visible/occluded/out-of-frame states across complete deliveries. A field map needs pitch/ground landmarks or segmentation masks and calibration ground truth.

Do not infer any of these labels from dataset marketing descriptions. Verify the actual archive schema and annotation content first.

## Current release consequence

No cricket-specific accuracy metric, tracking score, ball trajectory claim or model checkpoint is approved by this review. The generic COCO person detector remains the baseline until a rights-cleared, match-disjoint, manually reviewed benchmark is established and evaluated. The source review can be updated when exact permissions and archive contents have been verified.
