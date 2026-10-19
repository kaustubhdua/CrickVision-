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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Input video path")
    p.add_argument("--output", required=True, type=Path, help="Annotated MP4 path")
    p.add_argument("--detections", type=Path, default=None, help="JSONL output (default: <output>.jsonl)")
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--every-n-frames", type=int, default=1)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--no-tracking", action="store_true", help="Disable experimental IoU tracking")
    p.add_argument("--shot-threshold", type=float, default=0.42, help="Frame-difference cut threshold")
    return p.parse_args()

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    detections_path = args.detections or Path(str(args.output) + ".jsonl")
    detections_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise SystemExit("Video has invalid frame dimensions")
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create output video: {args.output}")

    detector = PersonDetector(device=args.device, threshold=args.threshold)
    tracker = IoUTracker()
    shot_detector = ShotBoundaryDetector(threshold=args.shot_threshold)
    frame_index = processed = cuts = 0
    try:
        with detections_path.open("w", encoding="utf-8") as out:
            while True:
                ok, frame = cap.read()
                if not ok: break
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                is_cut, cut_score = shot_detector.update(frame)
                if is_cut:
                    tracker.reset()
                    cuts += 1
                    out.write(json.dumps({"type":"shot_boundary", "frame_index":frame_index,
                        "timestamp_ms":round(timestamp_ms,3), "score":round(cut_score,6)}) + "\n")
                    cv2.putText(frame, "SHOT CUT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 165, 255), 2, cv2.LINE_AA)
                if frame_index % args.every_n_frames == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = detector.predict(Image.fromarray(rgb))
                    processed += 1
                    tracked = tracker.update(detections) if not args.no_tracking else []
                    track_by_detection = {id(t.detection): t for t in tracked}
                    for item in detections:
                        x1,y1,x2,y2 = (int(round(v)) for v in (item.x1,item.y1,item.x2,item.y2))
                        track = track_by_detection.get(id(item))
                        label = f"person {item.confidence:.2f}"
                        record = {"type":"person_detection", "frame_index":frame_index,
                            "timestamp_ms":round(timestamp_ms,3), "class":"person",
                            "bbox_xyxy":[item.x1,item.y1,item.x2,item.y2],
                            "confidence":item.confidence}
                        if track is not None:
                            label = f"ID {track.track_id} {item.confidence:.2f}"
                            record.update({"track_id":track.track_id,"track_age":track.age})
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(40,220,90),2)
                        cv2.putText(frame,label,(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,(40,220,90),2,cv2.LINE_AA)
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.write(frame)
                frame_index += 1
    finally:
        cap.release()
        writer.release()
    print(f"Frames read: {frame_index}")
    print(f"Frames analysed: {processed}")
    print(f"Shot boundaries flagged: {cuts}")
    print(f"Annotated video: {args.output}")
    print(f"Detections JSONL: {detections_path}")

if __name__ == "__main__":
    main()
