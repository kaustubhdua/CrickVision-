"""Audit a person-box training manifest before model training.

Each non-empty JSONL row must contain image_path, match_id, split and boxes_xyxy.
Image paths are resolved relative to the manifest and must remain inside its directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

SPLITS = {"train", "validation", "test", "unassigned"}
RIGHTS = {"owned", "licensed", "research-only", "unknown", "prohibited"}


def audit_training_manifest(manifest_path: Path, *, check_images: bool = True) -> dict[str, Any]:
    """Return a machine-readable audit report without changing the input data."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"valid": False, "records": 0, "images_checked": 0,
                "errors": [f"cannot read manifest: {exc}"], "warnings": []}

    base = manifest_path.resolve().parent
    seen_images: dict[str, int] = {}
    match_splits: dict[str, str] = {}
    image_split: dict[str, str] = {}
    records = 0
    images_checked = 0

    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_no}: expected a JSON object")
            continue
        records += 1
        prefix = f"line {line_no}:"

        image_value = row.get("image_path")
        if not isinstance(image_value, str) or not image_value.strip():
            errors.append(f"{prefix} image_path must be a non-empty string")
            continue
        raw_path = Path(image_value)
        if raw_path.is_absolute():
            errors.append(f"{prefix} image_path must be relative")
            continue
        image_path = (base / raw_path).resolve()
        if not image_path.is_relative_to(base):
            errors.append(f"{prefix} image_path escapes the manifest directory")
            continue
        key = image_path.as_posix()
        if key in seen_images:
            errors.append(f"{prefix} duplicate image_path also used on line {seen_images[key]}")
        else:
            seen_images[key] = line_no

        match_id = row.get("match_id")
        split = row.get("split")
        if not isinstance(match_id, str) or not match_id.strip():
            errors.append(f"{prefix} match_id must be a non-empty string")
        if not isinstance(split, str) or split not in SPLITS:
            errors.append(f"{prefix} split must be train, validation, test, or unassigned")
        elif isinstance(match_id, str) and match_id.strip() and split in {"train", "validation", "test"}:
            prior = match_splits.get(match_id)
            if prior is not None and prior != split:
                errors.append(f"{prefix} match_id {match_id!r} leaks across splits ({prior!r} and {split!r})")
            match_splits[match_id] = split
        if isinstance(split, str) and split in {"train", "validation", "test"}:
            prior_split = image_split.get(key)
            if prior_split is not None and prior_split != split:
                errors.append(f"{prefix} image_path leaks across splits ({prior_split!r} and {split!r})")
            image_split[key] = split

        rights = row.get("rights_status")
        if rights is None:
            errors.append(f"{prefix} rights_status is required; record owned, licensed, research-only, unknown, or prohibited")
        elif not isinstance(rights, str) or rights not in RIGHTS:
            errors.append(f"{prefix} unsupported rights_status")
        elif rights == "prohibited":
            errors.append(f"{prefix} source is marked prohibited")
        elif rights == "unknown":
            warnings.append(f"{prefix} source rights are unknown; do not train or redistribute until reviewed")
        elif rights == "research-only":
            warnings.append(f"{prefix} research-only terms must permit the intended training use")

        boxes = row.get("boxes_xyxy")
        checked_boxes: list[tuple[float, float, float, float]] = []
        if not isinstance(boxes, list):
            errors.append(f"{prefix} boxes_xyxy must be a list (use [] for reviewed empty frames)")
        else:
            for box_index, box in enumerate(boxes):
                if (not isinstance(box, list) or len(box) != 4
                        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box)):
                    errors.append(f"{prefix} box {box_index} must contain four numeric coordinates")
                    continue
                coords = tuple(float(v) for v in box)
                if not all(math.isfinite(v) for v in coords):
                    errors.append(f"{prefix} box {box_index} coordinates must be finite")
                    continue
                x1, y1, x2, y2 = coords
                if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                    errors.append(f"{prefix} box {box_index} must have non-negative, positive-area xyxy coordinates")
                    continue
                checked_boxes.append(coords)

        if check_images:
            if not image_path.is_file():
                errors.append(f"{prefix} image does not exist: {image_value}")
            else:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        width, height = image.size
                    images_checked += 1
                    for box_index, (x1, y1, x2, y2) in enumerate(checked_boxes):
                        if x2 > width or y2 > height:
                            errors.append(f"{prefix} box {box_index} exceeds image bounds {width}x{height}")
                except (OSError, ValueError) as exc:
                    errors.append(f"{prefix} image cannot be decoded: {image_value} ({exc})")

    if records == 0:
        errors.append("manifest contains no records")
    return {"valid": not errors, "records": records, "unique_images": len(seen_images),
            "images_checked": images_checked, "matches": len(match_splits),
            "errors": errors, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL person-box training manifest")
    parser.add_argument("--skip-image-check", action="store_true",
                        help="Skip existence, decoding and box-bound checks")
    args = parser.parse_args()
    report = audit_training_manifest(args.manifest, check_images=not args.skip_image_check)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
