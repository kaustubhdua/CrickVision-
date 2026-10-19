"""Run local person detection, shot-aware IoU tracking and video annotation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

from crickvision.detector import PersonDetector
from crickvision.shot import ShotBoundaryDetector
from crickvision.tracker import IoUTracker
from crickvision.team import TeamColourConfig, classify_team_colour


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument("--output", required=True, type=Path, help="Annotated MP4 path")
    parser.add_argument(
        "--detections", type=Path, default=None,
        help="JSONL output (default: <output>.jsonl)",
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--every-n-frames", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable experimental IoU tracking")
    parser.add_argument("--shot-threshold", type=float, default=0.42,
                        help="Frame-difference cut threshold")
    parser.add_argument("--team-colors", type=Path, default=None,
                        help="Optional JSON config with team names and HSV lower/upper ranges")
    return parser.parse_args()


def validate_paths(input_path: Path, output_path: Path, detections_path: Path) -> None:
    """Prevent accidental overwrites and ambiguous output destinations."""
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    detections_resolved = detections_path.resolve()
    if input_resolved == output_resolved:
        raise SystemExit("Input and annotated output must be different files")
    if input_resolved == detections_resolved:
        raise SystemExit("Input video and JSONL detections must be different files")
    if output_resolved == detections_resolved:
        raise SystemExit("Annotated video and JSONL detections must be different files")



def load_team_colors(path: Path | None) -> tuple[TeamColourConfig, ...]:
    """Load optional user-defined HSV kit ranges from a JSON file."""
    if path is None:
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read team colour config {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("teams"), list):
        raise SystemExit("Team colour config must be a JSON object with a 'teams' array")
    teams = []
    names = set()
    for index, item in enumerate(payload["teams"]):
        if not isinstance(item, dict):
            raise SystemExit(f"Team colour entry {index} must be an object")
        name, lower, upper = item.get("name"), item.get("hsv_lower"), item.get("hsv_upper")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise SystemExit(f"Team colour entry {index} needs a unique non-empty name")
        if not isinstance(lower, list) or not isinstance(upper, list) or len(lower) != 3 or len(upper) != 3:
            raise SystemExit(f"Team colour entry {index} needs 3-value hsv_lower and hsv_upper arrays")
        if any(not isinstance(v, int) for v in lower + upper):
            raise SystemExit(f"Team colour entry {index} HSV values must be integers")
        if any(v < 0 or v > 179 for v in (lower[0], upper[0])):
            raise SystemExit(f"Team colour entry {index} hue values must be from 0 to 179")
        if any(v < 0 or v > 255 for v in (lower[1], lower[2], upper[1], upper[2])):
            raise SystemExit(f"Team colour entry {index} saturation/value values must be from 0 to 255")
        if any(a > b for a, b in zip(lower, upper)):
            raise SystemExit(f"Team colour entry {index} has a lower HSV value above its upper value")
        names.add(name)
        teams.append(TeamColourConfig(name.strip(), tuple(lower), tuple(upper)))
    if not teams:
        raise SystemExit("Team colour config must contain at least one team")
    return tuple(teams)


def _timestamp_ms(capture: cv2.VideoCapture, frame_index: int, fps: float) -> float:
    """Use decoder time when available, otherwise derive time from frame index."""
    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
    if frame_index > 0 and (timestamp <= 0 or not timestamp == timestamp):
        timestamp = frame_index * 1000.0 / fps
    return max(0.0, float(timestamp))


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input video not found: {args.input}")
    if args.every_n_frames < 1:
        raise SystemExit("--every-n-frames must be >= 1")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    if not 0 < args.shot_threshold <= 1:
        raise SystemExit("--shot-threshold must be in (0, 1]")

    detections_path = args.detections or Path(str(args.output) + ".jsonl")
    validate_paths(args.input, args.output, detections_path)
    team_colors = load_team_colors(args.team_colors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detections_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the model before opening output resources, so model/download failures
    # cannot leave a half-created output file or an unreleased VideoWriter.
    detector = PersonDetector(device=args.device, threshold=args.threshold)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"Could not open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps > 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise SystemExit("Video has invalid frame dimensions")

    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise SystemExit(f"Could not create output video: {args.output}")

    tracker = IoUTracker()
    shot_detector = ShotBoundaryDetector(threshold=args.shot_threshold)
    frame_index = processed = cuts = 0
    try:
        with detections_path.open("w", encoding="utf-8") as output:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ms = _timestamp_ms(capture, frame_index, fps)
                is_cut, cut_score = shot_detector.update(frame)
                if is_cut:
                    tracker.reset()
                    cuts += 1
                    output.write(json.dumps({
                        "type": "shot_boundary",
                        "frame_index": frame_index,
                        "timestamp_ms": round(timestamp_ms, 3),
                        "score": round(cut_score, 6),
                    }) + "\n")
                    cv2.putText(frame, "SHOT CUT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 165, 255), 2, cv2.LINE_AA)

                if frame_index % args.every_n_frames == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = detector.predict(Image.fromarray(rgb))
                    processed += 1
                    tracked = tracker.update(detections) if not args.no_tracking else []
                    track_by_detection = {id(item.detection): item for item in tracked}
                    for detection in detections:
                        x1, y1, x2, y2 = (
                            int(round(value)) for value in
                            (detection.x1, detection.y1, detection.x2, detection.y2)
                        )
                        track = track_by_detection.get(id(detection))
                        label = f"person {detection.confidence:.2f}"
                        record = {
                            "type": "person_detection",
                            "frame_index": frame_index,
                            "timestamp_ms": round(timestamp_ms, 3),
                            "class": "person",
                            "bbox_xyxy": [
                                detection.x1, detection.y1, detection.x2, detection.y2
                            ],
                            "confidence": detection.confidence,
                        }
                        if team_colors:
                            team_name, team_score = classify_team_colour(
                                frame, (detection.x1, detection.y1, detection.x2, detection.y2),
                                team_colors,
                            )
                            record.update({
                                "team": team_name,
                                "team_colour_fraction": round(team_score, 6),
                            })
                            if team_name != "Unknown":
                                label = f"{team_name} {label}"
                        if track is not None:
                            label = f"ID {track.track_id} {detection.confidence:.2f}"
                            record.update({"track_id": track.track_id, "track_age": track.age})
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 90), 2)
                        cv2.putText(frame, label, (x1, max(18, y1 - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (40, 220, 90), 2, cv2.LINE_AA)
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.write(frame)
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(f"Frames read: {frame_index}")
    print(f"Frames analysed: {processed}")
    print(f"Shot boundaries flagged: {cuts}")
    print(f"Annotated video: {args.output}")
    print(f"Detections JSONL: {detections_path}")


if __name__ == "__main__":
    main()
