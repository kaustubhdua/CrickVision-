import json

from PIL import Image

from crickvision.audit_training_manifest import audit_training_manifest


def write_image(path, size=(20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def write_manifest(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def row(image="frames/a.jpg", match="m1", split="train", boxes=None, **extra):
    result = {"image_path": image, "match_id": match, "split": split,
              "boxes_xyxy": [[1, 2, 10, 20]] if boxes is None else boxes}
    result.update(extra)
    return result


def test_valid_manifest_including_reviewed_empty_frame(tmp_path):
    write_image(tmp_path / "frames/a.jpg")
    write_image(tmp_path / "frames/b.jpg")
    manifest = tmp_path / "labels.jsonl"
    write_manifest(manifest, [row(), row("frames/b.jpg", "m2", "validation", [])])
    result = audit_training_manifest(manifest)
    assert result["valid"]
    assert result["records"] == 2
    assert result["images_checked"] == 2


def test_rejects_match_split_leakage_and_duplicate_images(tmp_path):
    write_image(tmp_path / "a.jpg")
    manifest = tmp_path / "labels.jsonl"
    write_manifest(manifest, [row("a.jpg", "same", "train"), row("a.jpg", "same", "test")])
    result = audit_training_manifest(manifest)
    assert not result["valid"]
    assert any("leaks across splits" in error for error in result["errors"])
    assert any("duplicate image_path" in error for error in result["errors"])


def test_rejects_path_traversal_and_out_of_bounds_boxes(tmp_path):
    write_image(tmp_path / "a.jpg")
    manifest = tmp_path / "labels.jsonl"
    write_manifest(manifest, [row("../outside.jpg"), row("a.jpg", "m2", "validation", [[1, 2, 25, 29]])])
    result = audit_training_manifest(manifest)
    assert any("escapes the manifest directory" in error for error in result["errors"])
    assert any("exceeds image bounds" in error for error in result["errors"])


def test_rights_metadata_is_not_treated_as_verified(tmp_path):
    write_image(tmp_path / "a.jpg")
    manifest = tmp_path / "labels.jsonl"
    write_manifest(manifest, [row("a.jpg", rights_status="unknown")])
    result = audit_training_manifest(manifest)
    assert result["valid"]
    assert any("rights are unknown" in warning for warning in result["warnings"])


def test_prohibited_rights_fail_closed(tmp_path):
    write_image(tmp_path / "a.jpg")
    manifest = tmp_path / "labels.jsonl"
    write_manifest(manifest, [row("a.jpg", rights_status="prohibited")])
    result = audit_training_manifest(manifest)
    assert not result["valid"]
    assert any("marked prohibited" in error for error in result["errors"])
