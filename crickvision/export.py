"""Export CrickVision JSONL detection records to a flat CSV file."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "video_id", "shot_id", "frame_index", "timestamp_ms", "type", "class",
    "x1", "y1", "x2", "y2", "confidence", "track_id", "track_age",
    "team", "team_colour_fraction", "shot_score", "ground_x", "ground_y",
    "center_x", "center_y", "radius", "candidate_score", "ball_track_id", "candidate_path_xy", "status",
]


def export_csv(source: Path, destination: Path) -> int:
    """Convert detection/shot-boundary JSONL records into CSV rows."""
    if source.resolve() == destination.resolve():
        raise ValueError("input JSONL and output CSV must be different files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        input_file = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot open input JSONL: {exc}") from exc
    with input_file, destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            row = {key: record.get(key, "") for key in FIELDS}
            box = record.get("bbox_xyxy")
            if box is not None:
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError(f"line {line_number}: bbox_xyxy must have four values")
                for key, value in zip(("x1", "y1", "x2", "y2"), box):
                    row[key] = value
            if record.get("type") == "shot_boundary":
                row["shot_score"] = record.get("score", "")
            ground = record.get("ground_xy")
            if isinstance(ground, list) and len(ground) == 2:
                row["ground_x"], row["ground_y"] = ground
            centre = record.get("center_xy")
            if isinstance(centre, list) and len(centre) == 2:
                row["center_x"], row["center_y"] = centre
            writer.writerow(row)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV")
    args = parser.parse_args()
    try:
        count = export_csv(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Exported {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
