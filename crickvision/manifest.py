"""Validate a JSONL video manifest before annotation or evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED = ("clip_id", "source_id", "match_id", "video_path", "rights_status", "split")
ALLOWED_SPLITS = {"train", "validation", "test", "unassigned"}
ALLOWED_RIGHTS = {"owned", "licensed", "research-only", "unknown", "prohibited"}


def validate_record(record: Any, line_number: int) -> list[str]:
    errors = []
    if not isinstance(record, dict):
        return [f"line {line_number}: expected a JSON object"]
    for key in REQUIRED:
        if not isinstance(record.get(key), str) or not record[key].strip():
            errors.append(f"line {line_number}: {key!r} must be a non-empty string")
    if isinstance(record.get("split"), str) and record["split"] not in ALLOWED_SPLITS:
        errors.append(f"line {line_number}: split must be train, validation, test, or unassigned")
    if isinstance(record.get("rights_status"), str) and record["rights_status"] not in ALLOWED_RIGHTS:
        errors.append(f"line {line_number}: unsupported rights_status")
    for key in ("clip_start_ms", "clip_end_ms"):
        value = record.get(key)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            errors.append(f"line {line_number}: {key} must be a non-negative number")
    start, end = record.get("clip_start_ms"), record.get("clip_end_ms")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end <= start:
        errors.append(f"line {line_number}: clip_end_ms must be greater than clip_start_ms")
    return errors


def validate_manifest(path: Path) -> list[str]:
    errors = []
    seen_clip_ids: set[str] = set()
    match_splits: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"cannot read manifest: {exc}"]

    nonempty_count = 0
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        nonempty_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON: {exc.msg}")
            continue
        errors.extend(validate_record(record, number))
        if not isinstance(record, dict):
            continue

        clip_id = record.get("clip_id")
        if isinstance(clip_id, str) and clip_id:
            if clip_id in seen_clip_ids:
                errors.append(f"line {number}: duplicate clip_id {clip_id!r}")
            seen_clip_ids.add(clip_id)

        match_id = record.get("match_id")
        split = record.get("split")
        if isinstance(match_id, str) and match_id and split in {"train", "validation", "test"}:
            prior_split = match_splits.get(match_id)
            if prior_split is not None and prior_split != split:
                errors.append(
                    f"line {number}: match_id {match_id!r} leaks across splits "
                    f"({prior_split!r} and {split!r})"
                )
            else:
                match_splits[match_id] = split

        if split == "test" and record.get("rights_status") in {"unknown", "prohibited"}:
            errors.append(f"line {number}: test asset rights must be reviewed before use")

    if nonempty_count == 0:
        errors.append("manifest is empty")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = validate_manifest(args.manifest)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Manifest valid: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
