"""Generic person-detection baseline for CrickVision.

This is not a cricket-trained model. It uses Torchvision's COCO-pretrained
Faster R-CNN model and filters detections to the COCO "person" class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)


@dataclass(frozen=True)
class PersonDetection:
    """A person bounding box in xyxy pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class PersonDetector:
    """COCO-pretrained person detector with a small, stable API."""

    # Torchvision COCO category ID for "person".
    PERSON_LABEL = 1

    def __init__(self, device: str = "auto", threshold: float = 0.60) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

        self.device = torch.device(device)
        self.threshold = threshold
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = self.weights.transforms()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[PersonDetection]:
        """Detect visible people in an RGB PIL image."""
        rgb = image.convert("RGB")
        tensor = self.transforms(rgb).to(self.device)
        output: dict[str, Any] = self.model([tensor])[0]

        detections: list[PersonDetection] = []
        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu()
        scores = output["scores"].detach().cpu()

        for box, label, score in zip(boxes, labels, scores):
            confidence = float(score.item())
            if int(label.item()) != self.PERSON_LABEL:
                continue
            if confidence < self.threshold:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.tolist())
            detections.append(
                PersonDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)
            )

        return detections
