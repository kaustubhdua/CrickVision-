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
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        PersonDetector(device="cpu", checkpoint=tmp_path / "missing.pt")
