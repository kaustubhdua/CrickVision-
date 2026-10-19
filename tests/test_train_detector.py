import json

from PIL import Image
import pytest

from crickvision.train_detector import PersonBoxDataset


def test_person_box_dataset_loads_boxes_and_empty_frame(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (32, 24), color=(20, 80, 120)).save(image_path)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join([
        json.dumps({"image_path": "frame.png", "split": "train",
                    "boxes_xyxy": [[2, 3, 20, 22]]}),
        json.dumps({"image_path": "frame.png", "split": "validation",
                    "boxes_xyxy": []}),
    ]) + "\n", encoding="utf-8")
    dataset = PersonBoxDataset(manifest, "train")
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 24, 32)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [1]


def test_person_box_dataset_requires_train_and_validation_rows(tmp_path):
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps({
        "image_path": "missing.png", "split": "train", "boxes_xyxy": []
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="no rows for split 'validation'"):
        PersonBoxDataset(manifest, "validation")


def test_person_box_dataset_rejects_invalid_box(tmp_path):
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps({
        "image_path": "frame.png", "split": "train",
        "boxes_xyxy": [[10, 10, 2, 2]]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="positive-area"):
        PersonBoxDataset(manifest, "train")
