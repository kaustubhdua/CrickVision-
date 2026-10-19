"""Manual planar homography calibration for field-ground point projection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    matrix: np.ndarray
    reprojection_error: float
    valid: bool
    point_count: int


def estimate_homography(
    image_points_px: list[list[float]] | np.ndarray,
    ground_points: list[list[float]] | np.ndarray,
    *,
    max_reprojection_error: float = 0.5,
) -> CalibrationResult:
    """Fit image-pixel to user-defined planar ground coordinates.

    The error is measured in the same units as ground_points (for example metres).
    Use at least five well-distributed, coplanar correspondences so reprojection error can provide a residual check.
    """
    if max_reprojection_error <= 0 or not np.isfinite(max_reprojection_error):
        raise ValueError("max_reprojection_error must be a finite positive number")
    image_points = np.asarray(image_points_px, dtype=np.float64)
    ground = np.asarray(ground_points, dtype=np.float64)
    if image_points.ndim != 2 or ground.ndim != 2 or image_points.shape != ground.shape:
        raise ValueError("image_points_px and ground_points must have matching Nx2 shapes")
    if image_points.shape[1] != 2 or image_points.shape[0] < 5:
        raise ValueError("at least five matching 2D point pairs are required")
    if not np.isfinite(image_points).all() or not np.isfinite(ground).all():
        raise ValueError("calibration points must be finite numbers")

    matrix, _ = cv2.findHomography(image_points, ground, method=0)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("homography estimation failed; points may be degenerate")
    projected = cv2.perspectiveTransform(image_points.reshape(1, -1, 2), matrix)[0]
    errors = np.linalg.norm(projected - ground, axis=1)
    error = float(np.sqrt(np.mean(np.square(errors))))
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("estimated homography is singular")
    return CalibrationResult(
        matrix=matrix,
        reprojection_error=error,
        valid=error <= max_reprojection_error,
        point_count=int(image_points.shape[0]),
    )


def project_point(result: CalibrationResult, point_xy_px: tuple[float, float]) -> tuple[float, float]:
    """Project an image-space point into the calibrated ground plane."""
    if not result.valid:
        raise ValueError("calibration is invalid: reprojection error exceeds the configured limit")
    point = np.asarray(point_xy_px, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("point_xy_px must contain two finite numbers")
    projected = cv2.perspectiveTransform(point.reshape(1, 1, 2), result.matrix)[0, 0]
    if not np.isfinite(projected).all():
        raise ValueError("point projection produced non-finite coordinates")
    return float(projected[0]), float(projected[1])


def project_bbox_footpoint(
    result: CalibrationResult,
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Project the bottom-centre of an xyxy person box to the ground plane."""
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have positive area")
    return project_point(result, ((x1 + x2) / 2.0, y2))


def load_calibration_config(path: Path) -> CalibrationResult:
    """Load manual point correspondences from a JSON calibration config."""
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read calibration config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("calibration config must be a JSON object")
    image_points = payload.get("image_points_px")
    ground_points = payload.get("ground_points")
    threshold = payload.get("max_reprojection_error", 0.5)
    if not isinstance(image_points, list) or not isinstance(ground_points, list):
        raise ValueError("calibration config needs image_points_px and ground_points arrays")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("max_reprojection_error must be a positive number")
    result = estimate_homography(
        image_points, ground_points, max_reprojection_error=float(threshold)
    )
    if not result.valid:
        raise ValueError(
            f"calibration reprojection error {result.reprojection_error:.6g} exceeds "
            f"limit {threshold}; verify point order, coordinate units, and landmark quality"
        )
    return result
