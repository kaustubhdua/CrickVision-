"""Evaluate JSONL person detections against frame-level ground-truth boxes."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from crickvision.detector import PersonDetection
from crickvision.metrics import detection_metrics


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
        records.append(value)
    return records


def _box(record: dict[str, Any], path: Path, row: int) -> PersonDetection:
    values = record.get("bbox_xyxy")
    if (not isinstance(values, list) or len(values) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)):
        raise ValueError(f"{path}: row {row}: bbox_xyxy must contain four numbers")
    x1, y1, x2, y2 = (float(v) for v in values)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"{path}: row {row}: bbox_xyxy values must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{path}: row {row}: bbox_xyxy must have positive area")
    confidence = record.get("confidence", 1.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f"{path}: row {row}: confidence must be between 0 and 1")
    return PersonDetection(x1, y1, x2, y2, float(confidence))


def _frame_key(record: dict[str, Any], path: Path, row: int) -> tuple[str, int]:
    frame = record.get("frame_index")
    if not isinstance(frame, int) or frame < 0:
        raise ValueError(f"{path}: row {row}: frame_index must be a non-negative integer")
    video = record.get("video_id", "default")
    if not isinstance(video, str) or not video:
        raise ValueError(f"{path}: row {row}: video_id must be a non-empty string")
    return video, frame


def evaluate_jsonl(predictions_path: Path, ground_truth_path: Path,
                   iou_threshold: float = 0.5) -> dict[str, Any]:
    """Evaluate one-to-one IoU detection metrics across all annotated frames."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be between 0 and 1")
    predictions: dict[tuple[str, int], list[PersonDetection]] = defaultdict(list)
    ground_truth: dict[tuple[str, int], list[PersonDetection]] = defaultdict(list)
    annotated_frames: set[tuple[str, int]] = set()
    frames_with_boxes: set[tuple[str, int]] = set()
    explicitly_empty_frames: set[tuple[str, int]] = set()

    for row, record in enumerate(_load_jsonl(predictions_path), 1):
        if record.get("type", "person_detection") != "person_detection":
            continue
        key = _frame_key(record, predictions_path, row)
        predictions[key].append(_box(record, predictions_path, row))

    for row, record in enumerate(_load_jsonl(ground_truth_path), 1):
        key = _frame_key(record, ground_truth_path, row)
        annotated_frames.add(key)
        if record.get("empty_frame") is True:
            explicitly_empty_frames.add(key)
            ground_truth.setdefault(key, [])
            continue
        frames_with_boxes.add(key)
        ground_truth[key].append(_box(record, ground_truth_path, row))

    if not annotated_frames:
        raise ValueError("ground-truth file contains no annotated frames")
    conflicting_empty_frames = frames_with_boxes & explicitly_empty_frames
    if conflicting_empty_frames:
        video_id, frame_index = sorted(conflicting_empty_frames)[0]
        raise ValueError(
            f"ground-truth frame {video_id!r}:{frame_index} mixes empty_frame and box rows"
        )

    keys = annotated_frames
    tp = fp = fn = 0
    frame_reports = []
    for video_id, frame_index in sorted(keys):
        result = detection_metrics(predictions[(video_id, frame_index)],
                                   ground_truth[(video_id, frame_index)], iou_threshold)
        tp += result.true_positives
        fp += result.false_positives
        fn += result.false_negatives
        frame_reports.append({
            "video_id": video_id, "frame_index": frame_index,
            "true_positives": result.true_positives,
            "false_positives": result.false_positives,
            "false_negatives": result.false_negatives,
            "precision": result.precision, "recall": result.recall, "f1": result.f1,
        })

    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "schema_version": 1,
        "metric": "frame-level greedy IoU precision/recall/F1 (not COCO AP/mAP)",
        "iou_threshold": iou_threshold,
        "annotated_frames": len(annotated_frames),
        "evaluated_frames": len(annotated_frames),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "frames": frame_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON report path (otherwise print to stdout)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        report = evaluate_jsonl(args.predictions, args.ground_truth, args.iou_threshold)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Evaluation report written to {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
