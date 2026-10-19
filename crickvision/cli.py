"""Run local person detection, shot-aware IoU tracking and video annotation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from crickvision.detector import PersonDetector
from crickvision.shot import ShotBoundaryDetector
from crickvision.tracker import IoUTracker
from crickvision.team import TeamColourConfig, classify_team_colour
from crickvision.calibration import load_calibration_config, project_bbox_footpoint
from crickvision.ball import BallCandidateDetector, BallCandidateTracker


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
    parser.add_argument("--weights", type=Path, default=None,
                        help="Optional local checkpoint from crickvision-train-detector")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable experimental IoU tracking")
    parser.add_argument("--shot-threshold", type=float, default=0.42,
                        help="Frame-difference cut threshold")
    parser.add_argument("--team-colors", type=Path, default=None,
                        help="Optional JSON config with team names and HSV lower/upper ranges")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Optional manual planar homography config for ground coordinates")
    parser.add_argument("--ball-candidates", action="store_true",
                        help="Enable experimental Hough-circle ball-like candidates")
    parser.add_argument("--ball-min-radius", type=int, default=1)
    parser.add_argument("--ball-max-radius", type=int, default=8)
    parser.add_argument("--ball-max-distance", type=float, default=60.0,
                        help="Maximum pixel displacement for candidate tracklet association")
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
    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if not math.isfinite(timestamp) or timestamp < 0 or (frame_index > 0 and timestamp == 0):
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
    try:
        calibration = load_calibration_config(args.calibration) if args.calibration else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detections_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ball_detector = (
            BallCandidateDetector(args.ball_min_radius, args.ball_max_radius)
            if args.ball_candidates else None
        )
        ball_tracker = (
            BallCandidateTracker(max_distance=args.ball_max_distance)
            if args.ball_candidates else None
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Load the model before opening output resources, so model/download failures
    # cannot leave a half-created output file or an unreleased VideoWriter.
    detector = PersonDetector(device=args.device, threshold=args.threshold, checkpoint=args.weights)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"Could not open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
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
    shot_id = 0
    try:
        with detections_path.open("w", encoding="utf-8") as output:
            if calibration is not None:
                output.write(json.dumps({
                    "schema_version": 1,
                    "video_id": args.input.stem,
                    "type": "calibration",
                    "valid": calibration.valid,
                    "reprojection_error": calibration.reprojection_error,
                    "point_count": calibration.point_count,
                    "ground_coordinate_system": "user_defined_planar",
                    "image_to_ground_homography": calibration.matrix.tolist(),
                }) + "\n")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ms = _timestamp_ms(capture, frame_index, fps)
                is_cut, cut_score = shot_detector.update(frame)
                ball_candidates_for_frame = []
                if ball_detector is not None and frame_index % args.every_n_frames == 0:
                    ball_candidates_for_frame = ball_detector.detect(frame)
                if is_cut:
                    tracker.reset()
                    if ball_tracker is not None:
                        ball_tracker.reset()
                    cuts += 1
                    shot_id += 1
                    output.write(json.dumps({
                        "schema_version": 1,
                        "video_id": args.input.stem,
                        "type": "shot_boundary",
                        "shot_id": shot_id,
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
                            "schema_version": 1,
                            "video_id": args.input.stem,
                            "type": "person_detection",
                            "shot_id": shot_id,
                            "frame_index": frame_index,
                            "timestamp_ms": round(timestamp_ms, 3),
                            "class": "person",
                            "bbox_xyxy": [
                                detection.x1, detection.y1, detection.x2, detection.y2
                            ],
                            "confidence": detection.confidence,
                        }
                        if calibration is not None:
                            ground_x, ground_y = project_bbox_footpoint(
                                calibration,
                                (detection.x1, detection.y1, detection.x2, detection.y2),
                            )
                            record["ground_xy"] = [ground_x, ground_y]
                            record["ground_coordinate_system"] = "user_defined_planar"
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
                            team_prefix = f"{record['team']} " if record.get("team") not in (None, "Unknown") else ""
                            label = f"ID {track.track_id} {team_prefix}{detection.confidence:.2f}"
                            record.update({"track_id": track.track_id, "track_age": track.age})
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 90), 2)
                        cv2.putText(frame, label, (x1, max(18, y1 - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (40, 220, 90), 2, cv2.LINE_AA)
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if ball_tracker is not None:
                        for ball_track in ball_tracker.update(ball_candidates_for_frame):
                            candidate = ball_track.candidate
                            centre = (int(round(candidate.x)), int(round(candidate.y)))
                            radius = max(1, int(round(candidate.radius)))
                            cv2.circle(frame, centre, radius, (255, 80, 20), 2, cv2.LINE_AA)
                            if len(ball_track.history_xy) > 1:
                                trail = [
                                    (int(round(x)), int(round(y)))
                                    for x, y in ball_track.history_xy
                                ]
                                cv2.polylines(
                                    frame, [np.asarray(trail, dtype=np.int32)],
                                    False, (255, 80, 20), 2, cv2.LINE_AA,
                                )
                            output.write(json.dumps({
                                "schema_version": 1,
                                "video_id": args.input.stem,
                                "type": "ball_candidate",
                                "shot_id": shot_id,
                                "frame_index": frame_index,
                                "timestamp_ms": round(timestamp_ms, 3),
                                "center_xy": [candidate.x, candidate.y],
                                "radius": candidate.radius,
                                "candidate_score": candidate.score,
                                "ball_track_id": ball_track.track_id,
                                "track_age": ball_track.age,
                                "candidate_path_xy": [list(point) for point in ball_track.history_xy],
                                "status": "unverified_candidate",
                            }, ensure_ascii=False) + "\n")
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
