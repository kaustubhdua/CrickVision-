import json

import pytest

from crickvision.benchmark import evaluate_jsonl


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_benchmark_aggregates_frame_level_metrics(tmp_path):
    pred = tmp_path / "pred.jsonl"
    gt = tmp_path / "gt.jsonl"
    write_jsonl(pred, [
        {"type": "person_detection", "frame_index": 0, "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9},
        {"type": "shot_boundary", "frame_index": 1, "score": 0.8},
    ])
    write_jsonl(gt, [
        {"frame_index": 0, "bbox_xyxy": [0, 0, 10, 10]},
        {"frame_index": 1, "bbox_xyxy": [5, 5, 15, 15]},
    ])
    report = evaluate_jsonl(pred, gt)
    assert report["true_positives"] == 1
    assert report["false_positives"] == 0
    assert report["false_negatives"] == 1
    assert report["precision"] == 1.0
    assert report["recall"] == 0.5
    assert report["f1"] == pytest.approx(2 / 3)


def test_benchmark_keeps_video_frames_separate(tmp_path):
    pred = tmp_path / "pred.jsonl"
    gt = tmp_path / "gt.jsonl"
    write_jsonl(pred, [{"video_id": "a", "frame_index": 0, "bbox_xyxy": [0, 0, 5, 5]}])
    write_jsonl(gt, [
        {"video_id": "a", "frame_index": 0, "bbox_xyxy": [0, 0, 5, 5]},
        {"video_id": "b", "frame_index": 0, "bbox_xyxy": [0, 0, 5, 5]},
    ])
    report = evaluate_jsonl(pred, gt)
    assert report["true_positives"] == 1
    assert report["false_negatives"] == 1


def test_benchmark_rejects_missing_ground_truth_boxes(tmp_path):
    pred = tmp_path / "pred.jsonl"
    gt = tmp_path / "gt.jsonl"
    write_jsonl(pred, [])
    write_jsonl(gt, [{"frame_index": 0, "bbox_xyxy": [1, 1, 0, 0]}])
    with pytest.raises(ValueError, match="positive area"):
        evaluate_jsonl(pred, gt)

def test_benchmark_counts_false_positives_on_empty_annotated_frame(tmp_path):
    pred = tmp_path / "pred.jsonl"
    gt = tmp_path / "gt.jsonl"
    write_jsonl(pred, [
        {"type": "person_detection", "frame_index": 2, "bbox_xyxy": [0, 0, 10, 10]}
    ])
    write_jsonl(gt, [{"frame_index": 2, "empty_frame": True}])
    report = evaluate_jsonl(pred, gt)
    assert report["annotated_frames"] == 1
    assert report["false_positives"] == 1
    assert report["false_negatives"] == 0
    assert report["precision"] == 0.0

def test_benchmark_ignores_unannotated_prediction_frames(tmp_path):
    pred = tmp_path / "pred.jsonl"
    gt = tmp_path / "gt.jsonl"
    write_jsonl(pred, [
        {"frame_index": 0, "bbox_xyxy": [0, 0, 10, 10]},
        {"frame_index": 1, "bbox_xyxy": [20, 20, 30, 30]},
    ])
    write_jsonl(gt, [{"frame_index": 0, "bbox_xyxy": [0, 0, 10, 10]}])
    report = evaluate_jsonl(pred, gt)
    assert report["evaluated_frames"] == 1
    assert report["true_positives"] == 1
    assert report["false_positives"] == 0

def test_benchmark_rejects_mixed_empty_frame_and_box_rows(tmp_path):
    pred = tmp_path / "pred.jsonl"
    gt = tmp_path / "gt.jsonl"
    write_jsonl(pred, [])
    write_jsonl(gt, [
        {"video_id": "a", "frame_index": 0, "empty_frame": True},
        {"video_id": "a", "frame_index": 0, "bbox_xyxy": [0, 0, 10, 10]},
    ])
    with pytest.raises(ValueError, match="mixes empty_frame and box rows"):
        evaluate_jsonl(pred, gt)


def test_benchmark_rejects_non_finite_box_coordinates(tmp_path):
    pred = tmp_path / "pred.jsonl"
    gt = tmp_path / "gt.jsonl"
    write_jsonl(pred, [])
    write_jsonl(gt, [{"frame_index": 0, "bbox_xyxy": [0, 0, float("inf"), 10]}])
    with pytest.raises(ValueError, match="finite"):
        evaluate_jsonl(pred, gt)
