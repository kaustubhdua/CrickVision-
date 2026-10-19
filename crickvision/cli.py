"""Run CrickVision's generic person-detection baseline on a local video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

from crickvision.detector import PersonDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument("--output", required=True, type=Path, help="Annotated MP4 path")
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="JSONL path (default: <output>.jsonl)",
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--every-n-frames", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input video not found: {args.input}")
    if args.every_n_frames < 1:
        raise SystemExit("--every-n-frames must be >= 1")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    detections_path = args.detections or Path(str(args.output) + ".jsonl")
    detections_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise SystemExit("Video has invalid frame dimensions")

    # MP4V is widely available in OpenCV builds; re-encode with FFmpeg if needed.
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create output video: {args.output}")

    detector = PersonDetector(device=args.device, threshold=args.threshold)
    frame_index = 0
    processed = 0

    try:
        with detections_path.open("w", encoding="utf-8") as jsonl:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                frame_detections = []
                if frame_index % args.every_n_frames == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    people = detector.predict(Image.fromarray(rgb))
                    processed += 1
                    for item in people:
                        x1, y1, x2, y2 = (
                            int(round(item.x1)),
                            int(round(item.y1)),
                            int(round(item.x2)),
                            int(round(item.y2)),
                        )
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 90), 2)
                        label = f"person {item.confidence:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (40, 220, 90),
                            2,
                            cv2.LINE_AA,
                        )
                        record = {
                            "frame_index": frame_index,
                            "timestamp_ms": round(timestamp_ms, 3),
                            "class": "person",
                            "bbox_xyxy": [item.x1, item.y1, item.x2, item.y2],
                            "confidence": item.confidence,
                        }
                        frame_detections.append(record)
                    for record in frame_detections:
                        jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

                writer.write(frame)
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(f"Frames read: {frame_index}")
    print(f"Frames analysed: {processed}")
    print(f"Annotated video: {args.output}")
    print(f"Detections JSONL: {detections_path}")


if __name__ == "__main__":
    main()
