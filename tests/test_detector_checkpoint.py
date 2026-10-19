import torch
import pytest

import crickvision.detector as detector_module
from crickvision.detector import PersonDetector


class DummyModel:
    def __init__(self):
        self.loaded = None
        self.device = None
        self.evaluated = False

    def load_state_dict(self, state):
        self.loaded = state

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, images):
        return [{
            "boxes": torch.tensor([[1.0, 2.0, 11.0, 22.0], [0.0, 0.0, 5.0, 5.0]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.95, 0.99]),
        }]


def test_custom_checkpoint_builds_two_class_detector(tmp_path, monkeypatch):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.touch()
    model = DummyModel()
    calls = {}

    def make_model(**kwargs):
        calls.update(kwargs)
        return model

    monkeypatch.setattr(detector_module, "fasterrcnn_resnet50_fpn", make_model)
    monkeypatch.setattr(
        detector_module.torch, "load",
        lambda *args, **kwargs: {"state_dict": {"weight": torch.tensor([1.0])}},
    )
    detector = PersonDetector(device="cpu", checkpoint=checkpoint)
    assert calls["num_classes"] == 2
    assert model.loaded["weight"].item() == 1.0
    assert model.evaluated
    assert detector.checkpoint == checkpoint


def test_custom_checkpoint_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        PersonDetector(device="cpu", checkpoint=tmp_path / "missing.pt")

def test_custom_detector_filters_non_person_labels(tmp_path, monkeypatch):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.touch()
    model = DummyModel()
    monkeypatch.setattr(detector_module, "fasterrcnn_resnet50_fpn", lambda **kwargs: model)
    monkeypatch.setattr(
        detector_module.torch, "load",
        lambda *args, **kwargs: {"state_dict": {"weight": torch.tensor([1.0])}},
    )
    detector = PersonDetector(device="cpu", threshold=0.5, checkpoint=checkpoint)
    from PIL import Image
    detections = detector.predict(Image.new("RGB", (16, 16)))
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.95)
    assert (detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2) == (1, 2, 11, 22)
