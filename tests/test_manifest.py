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
