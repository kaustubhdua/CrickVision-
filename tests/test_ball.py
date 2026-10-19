import cv2
import numpy as np
import pytest

from crickvision.ball import BallCandidate, BallCandidateDetector, BallCandidateTracker


def test_ball_candidate_detector_finds_synthetic_circle():
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(frame, (64, 64), 5, (255, 255, 255), -1)
    candidates = BallCandidateDetector(min_radius=2, max_radius=8).detect(frame)
    assert candidates
    best = candidates[0]
    assert best.x == pytest.approx(64, abs=2)
    assert best.y == pytest.approx(64, abs=2)
    assert best.radius == pytest.approx(5, abs=2)


def test_ball_candidate_detector_returns_empty_for_blank_frame():
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    assert BallCandidateDetector().detect(frame) == []


def test_ball_candidate_tracker_links_nearby_candidates_and_resets():
    tracker = BallCandidateTracker(max_distance=10)
    first = tracker.update([BallCandidate(10, 10, 3, 0.8)])[0]
    second = tracker.update([BallCandidate(12, 11, 3, 0.7)])[0]
    assert first.track_id == second.track_id
    assert second.age == 2
    tracker.reset()
    third = tracker.update([BallCandidate(12, 11, 3, 0.7)])[0]
    assert third.track_id != second.track_id


def test_ball_candidate_tracker_rejects_invalid_distance():
    with pytest.raises(ValueError, match="finite and positive"):
        BallCandidateTracker(max_distance=float("inf"))
