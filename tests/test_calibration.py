import json

import numpy as np
import pytest

from crickvision.calibration import (
    estimate_homography,
    load_calibration_config,
    project_bbox_footpoint,
    project_point,
)


def test_homography_projects_synthetic_ground_points():
    image_points = [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]]
    ground_points = [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]]
    result = estimate_homography(image_points, ground_points, max_reprojection_error=0.01)
    assert result.valid
    assert result.reprojection_error < 1e-6
    x, y = project_point(result, (50, 50))
    assert x == pytest.approx(5.0)
    assert y == pytest.approx(5.0)


def test_project_bbox_uses_bottom_centre():
    result = estimate_homography(
        [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]],
        [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]],
    )
    assert project_bbox_footpoint(result, (40, 20, 60, 80)) == pytest.approx((5, 8))


def test_invalid_calibration_rejects_projection():
    result = estimate_homography(
        [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]],
        [[0, 0], [10, 0], [10, 10], [0, 10], [5, 20]],
        max_reprojection_error=0.00001,
    )
    assert not result.valid
    with pytest.raises(ValueError, match="calibration is invalid"):
        project_point(result, (50, 50))


def test_calibration_config_round_trip(tmp_path):
    config = tmp_path / "calibration.json"
    config.write_text(json.dumps({
        "image_points_px": [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]],
        "ground_points": [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]],
        "max_reprojection_error": 0.01,
    }), encoding="utf-8")
    result = load_calibration_config(config)
    assert result.valid
    assert result.point_count == 5


def test_rejects_too_few_calibration_points():
    with pytest.raises(ValueError, match="at least five"):
        estimate_homography([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
