# CrickVision baseline model card

## Model identity

- Model: Torchvision Faster R-CNN ResNet-50 FPN, COCO-pretrained weights.
- Intended use: generic visible-person detection as an engineering baseline on local video.
- Not a cricket-specific model; no cricket fine-tuning or accuracy benchmark is claimed.
- Output: person boxes in source-frame pixel coordinates and detector confidence scores.

## Intended and out-of-scope use

Use for development, qualitative inspection, and preparing labelled cricket-video experiments. Do not use as official scoring, umpiring, player identification, betting, disciplinary evidence, or a guarantee that all players are detected.

## Known limitations

- Distant fielders may be missed in wide shots.
- Crowds, officials, advertising, motion blur, occlusion, night lighting, camera cuts, and non-standard views may degrade results.
- The generic COCO person class does not distinguish player, umpire, coach, or spectator.
- IoU tracking IDs are temporary tracklet labels, not identity.
- Team-colour hints are hand-configured HSV rules and can be wrong under lighting changes or similar kits.
- Shot boundaries are a frame-difference heuristic and can produce false positives.
- No ball trajectory, calibrated ground coordinates, delivery events, or official match statistics are provided.

## Data and evaluation

The repository does not include a legally redistributable labelled cricket benchmark or custom-trained cricket checkpoint. Evaluation tooling supports frame-level greedy IoU precision/recall/F1 only. Publish benchmark results only with documented rights, match-level split, annotation protocol, checkpoint hash, and evaluation configuration. This metric is not COCO AP/mAP or a tracking metric.

## Reproducibility

Record Python, PyTorch, Torchvision, OpenCV, model-weight version/hash, device, video source hash, thresholds, frame sampling, and source frame indices for each run. The default pretrained weights may be downloaded on first use; cache them before an offline run.

## Licence and provenance

The root MIT licence applies to original CrickVision code/documentation only. It does not grant rights to the Torchvision checkpoint, COCO source data, input footage, or third-party components. Audit each separately before redistribution.
