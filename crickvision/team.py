"""HSV kit-colour heuristic; returns Unknown when visual evidence is weak."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class TeamColourConfig:
    name: str
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]

def classify_team_colour(frame_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float],
                         teams: tuple[TeamColourConfig, ...], *, min_pixels: int = 20,
                         min_fraction: float = 0.12, margin: float = 0.04) -> tuple[str, float]:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("frame must be a non-empty image")
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    left, right = max(0, min(w, int(x1+(x2-x1)*.2))), max(0, min(w, int(x1+(x2-x1)*.8)))
    top, bottom = max(0, min(h, int(y1+(y2-y1)*.2))), max(0, min(h, int(y1+(y2-y1)*.62)))
    if right <= left or bottom <= top or not teams:
        return "Unknown", 0.0
    hsv = cv2.cvtColor(frame_bgr[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    area = max(1, hsv.shape[0]*hsv.shape[1])
    counts = []
    for team in teams:
        mask = cv2.inRange(hsv, np.array(team.hsv_lower, np.uint8), np.array(team.hsv_upper, np.uint8))
        counts.append((team.name, cv2.countNonZero(mask)/area))
    counts.sort(key=lambda item: item[1], reverse=True)
    if counts[0][1] < min_fraction or counts[0][1]*area < min_pixels:
        return "Unknown", counts[0][1]
    if len(counts) > 1 and counts[0][1]-counts[1][1] < margin:
        return "Unknown", counts[0][1]
    return counts[0]
