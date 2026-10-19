"""Small detection metrics for matched box annotations."""
from __future__ import annotations

from dataclasses import dataclass

from crickvision.detector import PersonDetection
from crickvision.tracker import iou


@dataclass(frozen=True)
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def detection_metrics(
    predictions: list[PersonDetection],
    ground_truth: list[PersonDetection],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Greedy one-to-one IoU matching; use only on labelled, held-out frames."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    candidates = sorted(
        (
            (iou(prediction, truth), pred_index, truth_index)
            for pred_index, prediction in enumerate(predictions)
            for truth_index, truth in enumerate(ground_truth)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    for score, pred_index, truth_index in candidates:
        if score < iou_threshold:
            break
        if pred_index in used_predictions or truth_index in used_ground_truth:
            continue
        used_predictions.add(pred_index)
        used_ground_truth.add(truth_index)

    true_positives = len(used_predictions)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else (1.0 if not ground_truth else 0.0)
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else (1.0 if not predictions else 0.0)
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return DetectionMetrics(
        true_positives, false_positives, false_negatives, precision, recall, f1
    )
