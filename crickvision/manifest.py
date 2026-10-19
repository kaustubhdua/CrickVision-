"""Validate a JSONL video manifest before annotation or evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED = ("clip_id", "source_id", "match_id", "video_path", "rights_status", "split")

def validate_record(record: Any, line_number: int) -> list[str]:
    errors = []
    if not isinstance(record, dict):
        return [f"line {line_number}: expected a JSON object"]
    for key in REQUIRED:
        if not record.get(key):
            errors.append(f"line {line_number}: missing required field {key!r}")
    if record.get("split") not in {None, "train", "validation", "test", "unassigned"}:
        errors.append(f"line {line_number}: split must be train, validation, test, or unassigned")
    if record.get("rights_status") not in {None, "owned", "licensed", "research-only", "unknown", "prohibited"}:
        errors.append(f"line {line_number}: unsupported rights_status")
    return errors

def validate_manifest(path: Path) -> list[str]:
    errors = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"cannot read manifest: {exc}"]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON: {exc.msg}")
            continue
        errors.extend(validate_record(record, number))
        if isinstance(record, dict):
            clip_id = record.get("clip_id")
            if clip_id and clip_id in seen:
                errors.append(f"line {number}: duplicate clip_id {clip_id!r}")
            if clip_id:
                seen.add(clip_id)
            if record.get("split") == "test" and record.get("rights_status") in {"unknown", "prohibited"}:
                errors.append(f"line {number}: test asset rights must be reviewed before use")
    if not lines:
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
