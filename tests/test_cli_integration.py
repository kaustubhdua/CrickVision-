import json
import sys

import cv2
import numpy as np

from crickvision.cli import main
from crickvision.detector import PersonDetection


class FakeCapture:
    def __init__(self, _path):
        self.index = 0
        self.frames = [
            np.zeros((30, 40, 3), dtype=np.uint8),
            np.full((30, 40, 3), 255, dtype=np.uint8),
        ]

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 40
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 30
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.index * 40.0
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        pass


class FakeWriter:
    def __init__(self, *_args):
        self.frames = []

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        pass


class FakeDetector:
    def predict(self, _image):
        return [PersonDetection(2, 2, 12, 22, 0.95)]


def test_cli_end_to_end_with_synthetic_frames(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.mp4"
    source.touch()
    output = tmp_path / "annotated.mp4"
    detections = tmp_path / "detections.jsonl"
    writer = FakeWriter()

    monkeypatch.setattr(sys, "argv", [
        "crickvision", "--input", str(source), "--output", str(output),
        "--detections", str(detections),
    ])
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *_args: 0)
    monkeypatch.setattr("crickvision.cli.PersonDetector",
                        lambda **_kwargs: FakeDetector())

    main()

    records = [json.loads(line) for line in detections.read_text(encoding="utf-8").splitlines()]
    people = [record for record in records if record["type"] == "person_detection"]
    cuts = [record for record in records if record["type"] == "shot_boundary"]
    assert len(writer.frames) == 2
    assert len(people) == 2
    assert len(cuts) == 1
    assert people[0]["video_id"] == "synthetic"
    assert people[0]["shot_id"] == 0
    assert people[1]["shot_id"] == 1
    assert all(record["schema_version"] == 1 for record in people + cuts)
