"""Frame-difference shot boundary heuristic; validate on target footage."""
from __future__ import annotations
import cv2
import numpy as np

class ShotBoundaryDetector:
    def __init__(self, threshold: float = 0.42, resize_width: int = 160) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if resize_width < 16:
            raise ValueError("resize_width must be >= 16")
        self.threshold, self.resize_width = threshold, resize_width
        self._previous = None

    def reset(self) -> None:
        self._previous = None

    def update(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame must be a non-empty image")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        th = max(1, round(h * self.resize_width / max(1, w)))
        small = cv2.resize(gray, (self.resize_width, th), interpolation=cv2.INTER_AREA)
        if self._previous is None:
            self._previous = small
            return False, 0.0
        if small.shape != self._previous.shape:
            self._previous = small
            return True, 1.0
        score = float(np.mean(cv2.absdiff(small, self._previous)) / 255.0)
        self._previous = small
        return score >= self.threshold, score
