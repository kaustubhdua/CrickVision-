"""Create a reproducibility sidecar for a completed local CrickVision run.

The sidecar records hashes and decoder/environment metadata; it does not certify
footage rights, annotation quality, or model accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the whole media file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_run_record(
    input_video: Path,
    output_video: Path,
    detections: Path,
    *,
    checkpoint: Path | None = None,
    settings_path: Path | None = None,
) -> dict[str, Any]:
    """Build a provenance record for already-created run artifacts."""
    for label, path in (
        ("input video", input_video),
        ("output video", output_video),
        ("detections JSONL", detections),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist or is not a file: {path}")
    if checkpoint is not None and not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist or is not a file: {checkpoint}")

    capture = cv2.VideoCapture(str(input_video))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot decode input video: {input_video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames < 1 or width < 1 or height < 1:
            raise ValueError("input video has invalid frame count or dimensions")
    finally:
        capture.release()

    settings: dict[str, Any] | None = None
    if settings_path is not None:
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read settings JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("settings JSON must contain an object")
        settings = loaded

    try:
        package_version = version("crickvision")
    except PackageNotFoundError:
        package_version = "source-checkout (package metadata unavailable)"

    record: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "crickvision_run_provenance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": {"name": "CrickVision", "version": package_version},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
        },
        "input_video": {
            "name": input_video.name,
            "sha256": sha256_file(input_video),
            "width": width,
            "height": height,
            "fps_reported_by_decoder": fps if fps > 0 else None,
            "frame_count_reported_by_decoder": frames,
            "duration_seconds_estimate": frames / fps if fps > 0 else None,
        },
        "outputs": {
            "annotated_video": {
                "name": output_video.name,
                "sha256": sha256_file(output_video),
                "bytes": output_video.stat().st_size,
            },
            "detections_jsonl": {
                "name": detections.name,
                "sha256": sha256_file(detections),
                "bytes": detections.stat().st_size,
            },
        },
        "checkpoint": (
            {"name": checkpoint.name, "sha256": sha256_file(checkpoint)}
            if checkpoint is not None else {
                "identifier": "Torchvision Faster R-CNN ResNet-50 FPN COCO default weights; "
                              "exact cached checkpoint hash not measured"
            }
        ),
        "settings": settings,
        "rights_status": "not_assessed",
        "accuracy_status": "not_evaluated_by_this_record",
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Original source video")
    parser.add_argument("--output-video", required=True, type=Path, help="Annotated MP4")
    parser.add_argument("--detections", required=True, type=Path, help="Run JSONL output")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional exact local model checkpoint used")
    parser.add_argument("--settings", type=Path, default=None,
                        help="Optional JSON object containing the CLI settings used")
    parser.add_argument("--output", required=True, type=Path, help="Provenance JSON output")
    args = parser.parse_args()
    try:
        record = build_run_record(
            args.input, args.output_video, args.detections,
            checkpoint=args.checkpoint, settings_path=args.settings,
        )
        if args.output.resolve() in {
            args.input.resolve(), args.output_video.resolve(), args.detections.resolve()
        }:
            raise ValueError("provenance output must not overwrite an input or run artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Run provenance written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
