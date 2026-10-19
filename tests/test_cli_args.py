import json
import sys

import pytest

from crickvision.cli import (
    _timestamp_ms,
    load_team_colors,
    parse_args,
    validate_paths,
)


def test_parser_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "crickvision", "--input", "input.mp4", "--output", "output.mp4"
    ])
    args = parse_args()
    assert args.threshold == 0.60
    assert args.every_n_frames == 1
    assert args.device == "auto"
    assert args.no_tracking is False


def test_paths_must_not_overwrite_input(tmp_path):
    source = tmp_path / "match.mp4"
    source.touch()
    with pytest.raises(SystemExit, match="must be different"):
        validate_paths(source, source, tmp_path / "detections.jsonl")


def test_video_and_jsonl_outputs_must_differ(tmp_path):
    same = tmp_path / "result.mp4"
    with pytest.raises(SystemExit, match="different files"):
        validate_paths(tmp_path / "input.mp4", same, same)


def test_timestamp_falls_back_to_frame_rate():
    class Capture:
        @staticmethod
        def get(_property):
            return 0.0

    assert _timestamp_ms(Capture(), 25, 25.0) == 1000.0


def test_timestamp_uses_decoder_position_when_available():
    class Capture:
        @staticmethod
        def get(_property):
            return 480.0

    assert _timestamp_ms(Capture(), 12, 25.0) == 480.0

def test_load_team_colours(tmp_path):
    config = tmp_path / "teams.json"
    config.write_text(json.dumps({"teams": [
        {"name": "Blue", "hsv_lower": [90, 80, 40], "hsv_upper": [130, 255, 255]}
    ]}), encoding="utf-8")
    teams = load_team_colors(config)
    assert len(teams) == 1
    assert teams[0].name == "Blue"
    assert teams[0].hsv_lower == (90, 80, 40)


def test_reject_invalid_team_colour_range(tmp_path):
    config = tmp_path / "teams.json"
    config.write_text(json.dumps({"teams": [
        {"name": "Bad", "hsv_lower": [120, 0, 0], "hsv_upper": [90, 255, 255]}
    ]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="lower HSV"):
        load_team_colors(config)
