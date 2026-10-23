import json
from crickvision.manifest import validate_manifest

def test_valid_manifest(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text(json.dumps({"clip_id":"c1","source_id":"s1","match_id":"m1",
        "video_path":"data/clip.mp4","rights_status":"licensed","split":"train"}))
    assert validate_manifest(p) == []

def test_duplicate_ids_and_missing_fields(tmp_path):
    p = tmp_path / "manifest.jsonl"
    record = {"clip_id":"same","source_id":"s","match_id":"m","video_path":"x",
              "rights_status":"licensed","split":"train"}
    p.write_text(json.dumps(record) + "\n" + json.dumps(record))
    errors = validate_manifest(p)
    assert any("duplicate clip_id" in e for e in errors)

def test_rejects_unknown_split(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text(json.dumps({"clip_id":"x","source_id":"s","match_id":"m",
        "video_path":"x","rights_status":"licensed","split":"maybe"}))
    assert any("split must be" in e for e in validate_manifest(p))

def test_rejects_match_leakage_across_splits(tmp_path):
    p = tmp_path / "manifest.jsonl"
    base = {"source_id": "s", "match_id": "match-1", "video_path": "x",
            "rights_status": "licensed"}
    train = dict(base, clip_id="c1", split="train")
    test = dict(base, clip_id="c2", split="test")
    p.write_text(json.dumps(train) + "\n" + json.dumps(test), encoding="utf-8")
    assert any("leaks across splits" in error for error in validate_manifest(p))


def test_rejects_invalid_clip_time_range(tmp_path):
    p = tmp_path / "manifest.jsonl"
    record = {"clip_id": "c1", "source_id": "s", "match_id": "m",
              "video_path": "x", "rights_status": "licensed", "split": "train",
              "clip_start_ms": 1000, "clip_end_ms": 500}
    p.write_text(json.dumps(record), encoding="utf-8")
    assert any("clip_end_ms must be greater" in error for error in validate_manifest(p))


def test_blank_manifest_is_empty(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text("  \n\n", encoding="utf-8")
    assert "manifest is empty" in validate_manifest(p)


def test_rejects_prohibited_rights_in_any_split(tmp_path):
    p = tmp_path / "manifest.jsonl"
    record = {"clip_id": "c1", "source_id": "s", "match_id": "m",
              "video_path": "x", "rights_status": "prohibited", "split": "train"}
    p.write_text(json.dumps(record), encoding="utf-8")
    assert any("marked prohibited" in error for error in validate_manifest(p))


def test_rejects_non_finite_or_boolean_clip_times(tmp_path):
    p = tmp_path / "manifest.jsonl"
    base = {"clip_id": "c1", "source_id": "s", "match_id": "m",
            "video_path": "x", "rights_status": "licensed", "split": "train"}
    rows = [
        dict(base, clip_id="nan", clip_start_ms=float("nan")),
        dict(base, clip_id="bool", clip_start_ms=True),
        dict(base, clip_id="infinite", clip_end_ms=float("inf")),
    ]
    p.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    errors = validate_manifest(p)
    assert sum("finite non-negative number" in error for error in errors) == 3
