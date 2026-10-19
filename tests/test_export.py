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
                    "confidence": 0.9, "track_id": 7, "team": "Blue",
                    "ground_xy": [5.5, 12.0]}),
        json.dumps({"type": "shot_boundary", "frame_index": 5,
                    "timestamp_ms": 160, "score": 0.8}),
        json.dumps({"type": "ball_candidate", "frame_index": 6, "center_xy": [20.5, 30.5],
                    "radius": 3.0, "candidate_score": 0.7, "ball_track_id": 2,
                    "candidate_path_xy": [[18, 28], [20.5, 30.5]],
                    "status": "unverified_candidate"}),
    ]) + "\n", encoding="utf-8")
    assert export_csv(source, destination) == 3
    with destination.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["x1"] == "1"
    assert rows[0]["track_id"] == "7"
    assert rows[0]["team"] == "Blue"
    assert rows[0]["ground_x"] == "5.5"
    assert rows[0]["ground_y"] == "12.0"
    assert rows[1]["shot_score"] == "0.8"
    assert rows[2]["center_x"] == "20.5"
    assert rows[2]["center_y"] == "30.5"
    assert rows[2]["ball_track_id"] == "2"
    assert "20.5" in rows[2]["candidate_path_xy"]
    assert rows[2]["status"] == "unverified_candidate"


def test_export_rejects_same_path(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="different files"):
        export_csv(source, source)
