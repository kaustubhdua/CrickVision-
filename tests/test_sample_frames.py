import json

import numpy as np
import pytest

from crickvision import sample_frames


class FakeCapture:
    def __init__(self, frames, fps=10.0):
        self.frames = frames
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == sample_frames.cv2.CAP_PROP_FPS:
            return self.fps
        if prop == sample_frames.cv2.CAP_PROP_POS_MSEC:
            return max(0, self.index - 1) * 100.0
        return 0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


def setup_video(monkeypatch, tmp_path, frame_count=21):
    source = tmp_path / "match.mp4"
    source.write_bytes(b"test-video-bytes")
    frames = [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(frame_count)]
    capture = FakeCapture(frames)
    monkeypatch.setattr(sample_frames.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(sample_frames.cv2, "imwrite", lambda path, frame, params: True)
    return source, frames, capture


def test_samples_at_interval_and_writes_provenance_manifest(monkeypatch, tmp_path):
    source, _, capture = setup_video(monkeypatch, tmp_path)
    out = tmp_path / "frames"
    result = sample_frames.sample_video(source, out, every_seconds=0.5, rights_status="licensed")
    records = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert result["frames_sampled"] == 5
    assert [r["frame_index"] for r in records] == [0, 5, 10, 15, 20]
    assert all(r["source_sha256"] == sample_frames.sha256_file(source) for r in records)
    assert all(r["rights_status"] == "licensed" for r in records)
    assert records[0]["image_path"].endswith(".jpg")
    assert capture.released


def test_respects_max_frames(monkeypatch, tmp_path):
    source, _, _ = setup_video(monkeypatch, tmp_path)
    result = sample_frames.sample_video(source, tmp_path / "frames", every_seconds=0.1, max_frames=3)
    assert result["frames_sampled"] == 3


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_interval(monkeypatch, tmp_path, interval):
    source, _, _ = setup_video(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="every_seconds"):
        sample_frames.sample_video(source, tmp_path / "frames", every_seconds=interval)


def test_refuses_nonempty_output_directory(monkeypatch, tmp_path):
    source, _, _ = setup_video(monkeypatch, tmp_path)
    out = tmp_path / "frames"
    out.mkdir()
    (out / "keep.txt").write_text("do not overwrite")
    with pytest.raises(ValueError, match="not empty"):
        sample_frames.sample_video(source, out)
    assert (out / "keep.txt").read_text() == "do not overwrite"


def test_rejects_unknown_rights_status(monkeypatch, tmp_path):
    source, _, _ = setup_video(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="rights_status"):
        sample_frames.sample_video(source, tmp_path / "frames", rights_status="cleared")
