import hashlib
import json

import cv2
import numpy as np

from crickvision.validate_sample_manifest import validate_sample_manifest


def make_manifest(tmp_path, *, rights="licensed", image_name="frame.jpg", frame_index=0):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image_path = tmp_path / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(image_path), image)
    record = {
        "schema_version": 1,
        "type": "sampled_frame",
        "video_id": "match-1",
        "source_video_name": "match.mp4",
        "source_sha256": hashlib.sha256(b"video-bytes").hexdigest(),
        "frame_index": frame_index,
        "timestamp_ms": float(frame_index * 100),
        "image_path": image_name,
        "width": 12,
        "height": 8,
        "rights_status": rights,
        "sampling_interval_ms": 1000.0,
    }
    manifest = tmp_path / "frames.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest, record


def test_validates_sample_manifest_and_decoded_dimensions(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    result = validate_sample_manifest(manifest)
    assert result["valid"] is True
    assert result["records"] == 1
    assert result["errors"] == []


def test_warns_when_rights_are_unknown(tmp_path):
    manifest, _ = make_manifest(tmp_path, rights="unknown")
    result = validate_sample_manifest(manifest)
    assert result["valid"] is True
    assert any("rights are unknown" in warning for warning in result["warnings"])


def test_rejects_path_traversal(tmp_path):
    manifest, record = make_manifest(tmp_path)
    record["image_path"] = "../outside.jpg"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = validate_sample_manifest(manifest, verify_images=False)
    assert any("escapes the manifest directory" in error for error in result["errors"])


def test_rejects_prohibited_rights(tmp_path):
    manifest, record = make_manifest(tmp_path)
    record["rights_status"] = "prohibited"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = validate_sample_manifest(manifest)
    assert any("marked prohibited" in error for error in result["errors"])


def test_rejects_mismatched_image_dimensions(tmp_path):
    manifest, record = make_manifest(tmp_path)
    record["width"] = 99
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = validate_sample_manifest(manifest)
    assert any("width does not match" in error for error in result["errors"])


def test_rejects_bad_sha_and_duplicate_frame(tmp_path):
    manifest, record = make_manifest(tmp_path)
    record["source_sha256"] = "not-a-hash"
    second = dict(record, image_path="frame-copy.jpg")
    assert cv2.imwrite(str(tmp_path / "frame-copy.jpg"), np.zeros((8, 12, 3), dtype=np.uint8))
    manifest.write_text(json.dumps(record) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    result = validate_sample_manifest(manifest)
    assert any("source_sha256" in error for error in result["errors"])
    assert any("duplicate frame index" in error for error in result["errors"])
