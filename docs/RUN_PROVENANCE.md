# Reproducible run provenance

After a successful local analysis, create a sidecar record that fingerprints the exact source and output artifacts:

```bash
crickvision-record-run \
  --input path/to/authorised-match-clip.mp4 \
  --output-video runs/annotated.mp4 \
  --detections runs/detections.jsonl \
  --settings runs/settings.json \
  --output runs/provenance.json
```

The optional settings file is a JSON object. Include the confidence threshold, frame sampling interval, device, shot threshold, tracking toggle, team-colour configuration identifier, calibration configuration identifier and any other parameters needed to reproduce the run. Do not include secrets.

When a local fine-tuned checkpoint was used, pass `--checkpoint models/person-finetuned.pt`; the tool records its SHA-256. Without that argument, the record identifies the default Torchvision COCO-pretrained model family but explicitly does not claim to know the hash of a cached checkpoint.

The record includes:
- source video SHA-256 and decoder-reported dimensions, FPS and frame count;
- annotated video and JSONL SHA-256 hashes and file sizes;
- Python, platform, OpenCV, PyTorch, Torchvision and CrickVision versions where available;
- optional user-supplied run settings;
- explicit `rights_status: not_assessed` and `accuracy_status: not_evaluated_by_this_record` fields.

This is a provenance aid, not proof of permission, label quality, model identity when the default checkpoint is cached, or cricket accuracy. Store it with the run outputs. Keep restricted source footage and extracted frames private.
