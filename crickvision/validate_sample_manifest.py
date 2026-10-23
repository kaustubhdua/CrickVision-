"""Validate frame-sampler JSONL manifests and their extracted image files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2

RIGHTS = {"owned", "licensed", "research-only", "unknown", "prohibited"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample_manifest(
    manifest_path: Path,
    source_video: Path | None = None,
    verify_images: bool = True,
) -> dict[str, Any]:
    """Return structured validation errors and warnings for a sampler manifest."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"valid": False, "records": 0, "errors": [f"cannot read manifest: {exc}"], "warnings": []}

    base_dir = manifest_path.resolve().parent
    seen_frames: set[tuple[str, int]] = set()
    seen_images: set[str] = set()
    source_hashes: set[str] = set()
    video_ids: set[str] = set()
    records = 0

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(record, dict):
            errors.append(f"line {line_number}: expected a JSON object")
            continue
        records += 1
        prefix = f"line {line_number}:"
        if record.get("schema_version") != 1 or record.get("type") != "sampled_frame":
            errors.append(f"{prefix} expected schema_version 1 and type 'sampled_frame'")

        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            errors.append(f"{prefix} video_id must be a non-empty string")
            video_id = f"__invalid_line_{line_number}"

        source_hash = record.get("source_sha256")
        if not isinstance(source_hash, str) or not SHA256_RE.fullmatch(source_hash):
            errors.append(f"{prefix} source_sha256 must be a 64-character lowercase SHA-256")
        else:
            source_hashes.add(source_hash)

        frame_index = record.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            errors.append(f"{prefix} frame_index must be a non-negative integer")
        else:
            key = (video_id, frame_index)
            if key in seen_frames:
                errors.append(f"{prefix} duplicate frame index {frame_index} for video {video_id!r}")
            seen_frames.add(key)

        timestamp = record.get("timestamp_ms")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            errors.append(f"{prefix} timestamp_ms must be a finite non-negative number")

        for field in ("width", "height"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"{prefix} {field} must be a positive integer")

        interval = record.get("sampling_interval_ms")
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
            errors.append(f"{prefix} sampling_interval_ms must be a finite positive number")

        rights = record.get("rights_status")
        if not isinstance(rights, str) or rights not in RIGHTS:
            errors.append(f"{prefix} unsupported rights_status")
        elif rights == "prohibited":
            errors.append(f"{prefix} source is marked prohibited")
        elif rights == "unknown":
            warnings.append(f"{prefix} source rights are unknown; do not redistribute or train until reviewed")
        elif rights == "research-only":
            warnings.append(f"{prefix} research-only restrictions must be checked for the intended use")

        image_name = record.get("image_path")
        if not isinstance(image_name, str) or not image_name.strip():
            errors.append(f"{prefix} image_path must be a non-empty relative path")
            continue
        image_path = Path(image_name)
        if image_path.is_absolute():
            errors.append(f"{prefix} image_path must be relative")
            continue
        resolved_image = (base_dir / image_path).resolve()
        if not resolved_image.is_relative_to(base_dir):
            errors.append(f"{prefix} image_path escapes the manifest directory")
            continue
        normalized_name = resolved_image.relative_to(base_dir).as_posix()
        if normalized_name in seen_images:
            errors.append(f"{prefix} duplicate image_path {image_name!r}")
        seen_images.add(normalized_name)
        if verify_images:
            if not resolved_image.is_file():
                errors.append(f"{prefix} image file does not exist: {image_name}")
            else:
                image = cv2.imread(str(resolved_image), cv2.IMREAD_UNCHANGED)
                if image is None:
                    errors.append(f"{prefix} image cannot be decoded: {image_name}")
                else:
                    if isinstance(record.get("width"), int) and not isinstance(record.get("width"), bool) and image.shape[1] != record["width"]:
                        errors.append(f"{prefix} recorded width does not match image: {image_name}")
                    if isinstance(record.get("height"), int) and not isinstance(record.get("height"), bool) and image.shape[0] != record["height"]:
                        errors.append(f"{prefix} recorded height does not match image: {image_name}")

        video_ids.add(video_id)

    if records == 0:
        errors.append("manifest contains no records")
    if len(source_hashes) > 1:
        errors.append("manifest contains multiple source SHA-256 values")
    if source_video is not None:
        if not source_video.is_file():
            errors.append(f"source video does not exist: {source_video}")
        else:
            actual_hash = _sha256(source_video)
            if source_hashes and actual_hash not in source_hashes:
                errors.append("source video SHA-256 does not match the manifest")

    return {
        "valid": not errors,
        "records": records,
        "videos": len(video_ids),
        "source_sha256": next(iter(source_hashes)) if len(source_hashes) == 1 else None,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="frames.jsonl produced by crickvision-sample-frames")
    parser.add_argument("--source-video", type=Path, default=None, help="Optionally verify source-video SHA-256")
    parser.add_argument("--skip-image-decode", action="store_true", help="Check paths but do not decode every image")
    args = parser.parse_args()
    report = validate_sample_manifest(args.manifest, args.source_video, not args.skip_image_decode)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
