import csv
import json

import pytest

from crickvision.export import export_csv


def test_export_detection_and_shot_records(tmp_path):
    source = tmp_path / "events.jsonl"
    destination = tmp_path / "events.csv"
    source.write_text("\n".join([
        json.dumps({"type": "person_detection", "frame_index": 4,
                    "timestamp_ms": 120, "bbox_xyxy": [1, 2, 11, 22],
                    "confidence": 0.9, "track_id": 7, "team": "Blue"}),
        json.dumps({"type": "shot_boundary", "frame_index": 5,
                    "timestamp_ms": 160, "score": 0.8}),
    ]) + "\n", encoding="utf-8")
    assert export_csv(source, destination) == 2
    with destination.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["x1"] == "1"
    assert rows[0]["track_id"] == "7"
    assert rows[0]["team"] == "Blue"
    assert rows[1]["shot_score"] == "0.8"


def test_export_rejects_same_path(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="different files"):
        export_csv(source, source)
