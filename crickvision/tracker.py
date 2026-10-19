"""Dependency-light IoU tracker for short, single-camera cricket shots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from crickvision.detector import PersonDetection


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    detection: PersonDetection
    age: int
    missed_frames: int = 0


@dataclass
class _Track:
    track_id: int
    detection: PersonDetection
    age: int = 1
    missed: int = 0


def iou(a: PersonDetection, b: PersonDetection) -> float:
    """Compute intersection-over-union for two xyxy boxes."""
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class IoUTracker:
    """Greedy IoU association; reset at every camera cut."""

    def __init__(self, iou_threshold: float = 0.30, max_missed: int = 8) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_missed < 0:
            raise ValueError("max_missed must be >= 0")
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def reset(self, *, reset_ids: bool = False) -> None:
        self._tracks.clear()
        if reset_ids:
            self._next_id = 1

    def update(self, detections: Iterable[PersonDetection]) -> list[TrackedDetection]:
        items = list(detections)
        candidates = sorted(
            (
                (iou(track.detection, detection), track_id, index)
                for track_id, track in self._tracks.items()
                for index, detection in enumerate(items)
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        assignment: dict[int, int] = {}
        for score, track_id, index in candidates:
            if score < self.iou_threshold:
                break
            if track_id in used_tracks or index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(index)
            assignment[index] = track_id

        for track_id, track in list(self._tracks.items()):
            if track_id not in used_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self._tracks[track_id]

        output = []
        for index, detection in enumerate(items):
            track_id = assignment.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id, detection)
            else:
                track = self._tracks[track_id]
                track.detection = detection
                track.age += 1
                track.missed = 0
            track = self._tracks[track_id]
            output.append(TrackedDetection(track_id, detection, track.age, track.missed))
        return output
