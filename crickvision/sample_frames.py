"""Sample timestamped frames from a local video for manual annotation.

This utility creates image files plus a provenance-bearing JSONL manifest. It
does not infer labels and does not grant rights to the source footage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_timestamp_ms(capture: Any, frame_index: int, fps: float) -> float:
    """Read decoder time when available; otherwise use frame index and FPS."""
    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if not math.isfinite(timestamp) or timestamp < 0 or (frame_index > 0 and timestamp == 0):
        timestamp = frame_index * 1000.0 / fps
    return max(0.0, timestamp)


def sample_video(
    input_path: Path,
    output_dir: Path,
    every_seconds: float = 2.0,
    max_frames: int | None = None,
    jpeg_quality: int = 95,
    video_id: str | None = None,
    rights_status: str = "unknown",
) -> dict[str, Any]:
    """Extract frames at a time interval and write a JSONL sampling manifest."""
    if not input_path.is_file():
        raise ValueError(f"input video does not exist: {input_path}")
    if not math.isfinite(every_seconds) or every_seconds <= 0:
        raise ValueError("every_seconds must be a finite number greater than zero")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if rights_status not in {"owned", "licensed", "research-only", "unknown", "prohibited"}:
        raise ValueError("unsupported rights_status")

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frames.jsonl"
    if manifest_path.exists():
        raise ValueError(f"manifest already exists: {manifest_path}")

    source_hash = sha256_file(input_path)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    actual_video_id = video_id or input_path.stem
    interval_ms = every_seconds * 1000.0
    next_sample_ms = 0.0
    frame_index = 0
    sampled = 0
    records: list[dict[str, Any]] = []
    try:
        while max_frames is None or sampled < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = frame_timestamp_ms(capture, frame_index, fps)
            if timestamp_ms + 0.5 >= next_sample_ms:
                image_name = f"frame_{sampled:06d}_f{frame_index:09d}.jpg"
                image_path = output_dir / image_name
                written = cv2.imwrite(
                    str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
                if not written:
                    raise ValueError(f"could not write sampled frame: {image_path}")
                records.append({
                    "schema_version": 1,
                    "type": "sampled_frame",
                    "video_id": actual_video_id,
                    "source_video_name": input_path.name,
                    "source_sha256": source_hash,
                    "frame_index": frame_index,
                    "timestamp_ms": round(timestamp_ms, 3),
                    "image_path": image_name,
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "rights_status": rights_status,
                    "sampling_interval_ms": round(interval_ms, 3),
                })
                sampled += 1
                next_sample_ms += interval_ms
            frame_index += 1
    finally:
        capture.release()

    if not records:
        raise ValueError(f"video contained no decodable frames: {input_path}")

    temporary_manifest = output_dir / "frames.jsonl.tmp"
    try:
        with temporary_manifest.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_manifest.replace(manifest_path)
    except OSError:
        temporary_manifest.unlink(missing_ok=True)
        raise

    return {
        "video_id": actual_video_id,
        "source_sha256": source_hash,
        "frames_read": frame_index,
        "frames_sampled": sampled,
        "manifest_path": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source video path")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--every-seconds", type=float, default=2.0, help="Sampling interval (default: 2)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of extracted frames")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality 1–100 (default: 95)")
    parser.add_argument("--video-id", default=None, help="Stable dataset video ID (default: input filename stem)")
    parser.add_argument(
        "--rights-status", choices=["owned", "licensed", "research-only", "unknown", "prohibited"],
        default="unknown", help="Recorded rights status; this is not a legal rights check",
    )
    args = parser.parse_args()
    try:
        report = sample_video(
            args.input, args.output_dir, args.every_seconds, args.max_frames,
            args.jpeg_quality, args.video_id, args.rights_status,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
