"""Experimental Hough-circle proposals for tiny ball-like image regions.

These are visual candidates, not cricket-ball detections. Broadcast footage often
contains many false positives; review candidate overlays before using them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class BallCandidate:
    x: float
    y: float
    radius: float
    score: float


@dataclass(frozen=True)
class TrackedBallCandidate:
    track_id: int
    candidate: BallCandidate
    age: int
    history_xy: tuple[tuple[float, float], ...]


@dataclass
class _BallTrack:
    track_id: int
    candidate: BallCandidate
    age: int = 1
    missed: int = 0
    history_xy: list[tuple[float, float]] | None = None


class BallCandidateDetector:
    """Propose small circular regions using HoughCircles and local contrast."""

    def __init__(
        self,
        min_radius: int = 1,
        max_radius: int = 8,
        *,
        param1: float = 100.0,
        param2: float = 12.0,
        min_distance: float = 8.0,
    ) -> None:
        if min_radius < 1 or max_radius < min_radius:
            raise ValueError("radius bounds must satisfy 1 <= min_radius <= max_radius")
        if param1 <= 0 or param2 <= 0 or min_distance <= 0:
            raise ValueError("Hough parameters must be positive")
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.param1 = float(param1)
        self.param2 = float(param2)
        self.min_distance = float(min_distance)

    def detect(self, frame_bgr: np.ndarray) -> list[BallCandidate]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame must be a non-empty image")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.min_distance,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        if circles is None:
            return []
        height, width = gray.shape
        candidates = []
        for x, y, radius in circles[0]:
            x_i, y_i, r_i = int(round(x)), int(round(y)), max(1, int(round(radius)))
            patch_radius = r_i + 3
            if (x_i - patch_radius < 0 or y_i - patch_radius < 0
                    or x_i + patch_radius >= width or y_i + patch_radius >= height):
                continue
            patch = gray[
                y_i-patch_radius:y_i+patch_radius+1,
                x_i-patch_radius:x_i+patch_radius+1,
            ]
            yy, xx = np.ogrid[-patch_radius:patch_radius+1, -patch_radius:patch_radius+1]
            distance_squared = xx * xx + yy * yy
            inner_mask = distance_squared <= max(1, r_i - 1) ** 2
            ring_mask = (distance_squared >= (r_i + 1) ** 2) & (
                distance_squared <= (r_i + 3) ** 2
            )
            # Small local contrast score for ranking proposals; not a probability.
            inner_values = patch[inner_mask]
            ring_values = patch[ring_mask]
            if inner_values.size == 0 or ring_values.size == 0:
                score = 0.0
            else:
                score = abs(float(inner_values.mean()) - float(ring_values.mean())) / 255.0
            candidates.append(BallCandidate(float(x), float(y), float(radius), score))
        return sorted(candidates, key=lambda item: item.score, reverse=True)


class BallCandidateTracker:
    """Greedy nearest-neighbour linking for short candidate tracks, not ball identity."""

    def __init__(self, max_distance: float = 60.0, max_missed: int = 2) -> None:
        if not math.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("max_distance must be finite and positive")
        if max_missed < 0:
            raise ValueError("max_missed must be >= 0")
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _BallTrack] = {}

    def reset(self, *, reset_ids: bool = False) -> None:
        self._tracks.clear()
        if reset_ids:
            self._next_id = 1

    def update(self, candidates: list[BallCandidate]) -> list[TrackedBallCandidate]:
        pairs = sorted(
            (
                (math.hypot(track.candidate.x - candidate.x,
                            track.candidate.y - candidate.y), track_id, index)
                for track_id, track in self._tracks.items()
                for index, candidate in enumerate(candidates)
            ),
            key=lambda item: item[0],
        )
        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        assignments: dict[int, int] = {}
        for distance, track_id, index in pairs:
            if distance > self.max_distance:
                break
            if track_id in used_tracks or index in used_candidates:
                continue
            used_tracks.add(track_id)
            used_candidates.add(index)
            assignments[index] = track_id

        for track_id, track in list(self._tracks.items()):
            if track_id not in used_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self._tracks[track_id]

        output = []
        for index, candidate in enumerate(candidates):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _BallTrack(
                    track_id, candidate, history_xy=[(candidate.x, candidate.y)]
                )
            else:
                track = self._tracks[track_id]
                track.candidate = candidate
                track.age += 1
                track.missed = 0
                if track.history_xy is None:
                    track.history_xy = []
                track.history_xy.append((candidate.x, candidate.y))
            track = self._tracks[track_id]
            history = tuple(track.history_xy or [(candidate.x, candidate.y)])
            output.append(TrackedBallCandidate(track_id, candidate, track.age, history))
        return output
