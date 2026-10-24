"""Synthetic end-to-end smoke test for video annotation without model downloads."""
import json
import sys

import cv2
import numpy as np
import pytest

from crickvision import cli
from crickvision.detector import PersonDetection


class DeterministicPersonDetector:
    """Replace network-backed pretrained weights with deterministic test detections."""

    def __init__(self, device="auto", threshold=0.60, checkpoint=None):
        self.threshold = threshold

    def predict(self, image):
        width, height = image.size
        return [PersonDetection(
            x1=width * 0.2, y1=height * 0.1,
            x2=width * 0.5, y2=height * 0.9,
            confidence=0.97,
        )]


def test_cli_video_to_annotated_mp4_and_jsonl(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.mp4"
    output = tmp_path / "annotated.mp4"
    detections = tmp_path / "detections.jsonl"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120)
    )
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV MP4 encoder is unavailable in this environment")
    for index in range(5):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (32 + index, 12), (80 + index, 108), (0, 180, 30), -1)
        writer.write(frame)
    writer.release()

    monkeypatch.setattr(cli, "PersonDetector", DeterministicPersonDetector)
    monkeypatch.setattr(sys, "argv", [
        "crickvision", "--input", str(source), "--output", str(output),
        "--detections", str(detections), "--device", "cpu",
    ])
    cli.main()

    assert output.is_file() and output.stat().st_size > 0
    assert detections.is_file()
    records = [json.loads(line) for line in detections.read_text(encoding="utf-8").splitlines()]
    person_records = [row for row in records if row.get("type") == "person_detection"]
    assert len(person_records) == 5
    assert all(row["confidence"] == pytest.approx(0.97) for row in person_records)
    assert all(len(row["bbox_xyxy"]) == 4 for row in person_records)

    cap = cv2.VideoCapture(str(output))
    try:
        assert cap.isOpened()
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        assert decoded == 5
    finally:
        cap.release()
