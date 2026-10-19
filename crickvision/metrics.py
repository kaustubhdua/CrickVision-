"""Small, dependency-light detection metrics for matched box annotations."""
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

def detection_metrics(predictions: list[PersonDetection], ground_truth: list[PersonDetection],
                      iou_threshold: float = 0.5) -> DetectionMetrics:
    """Greedy one-to-one IoU matching; use only on labelled, held-out frames."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    candidates = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(predictions)
                         for gi, g in enumerate(ground_truth)), reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    for score, pi, gi in candidates:
        if score < iou_threshold:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
    tp = len(used_p)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else (1.0 if not ground_truth else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not predictions else 0.0)
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return DetectionMetrics(tp, fp, fn, precision, recall, f1)
