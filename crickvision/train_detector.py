"""Fine-tune a person detector from a reviewed JSONL image manifest.

Manifest rows:
{"image_path":"frames/frame001.jpg","split":"train","boxes_xyxy":[[10,20,80,200]]}
{"image_path":"frames/frame900.jpg","split":"validation","boxes_xyxy":[]}
Paths are resolved relative to the manifest file. Do not mix matches across splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor


class PersonBoxDataset(Dataset):
    """One image and zero or more person boxes per JSONL row."""

    def __init__(self, manifest: Path, split: str) -> None:
        self.base = manifest.resolve().parent
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read manifest: {exc}") from exc
        self.rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest line {line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_no}: expected an object")
            if row.get("split") != split:
                continue
            image_path = row.get("image_path")
            boxes = row.get("boxes_xyxy")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"manifest line {line_no}: image_path is required")
            if not isinstance(boxes, list):
                raise ValueError(f"manifest line {line_no}: boxes_xyxy must be a list")
            checked_boxes = []
            for box in boxes:
                if (not isinstance(box, list) or len(box) != 4
                        or any(not isinstance(v, (int, float)) for v in box)):
                    raise ValueError(f"manifest line {line_no}: each box must contain four numbers")
                x1, y1, x2, y2 = (float(v) for v in box)
                if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                    raise ValueError(f"manifest line {line_no}: invalid positive-area xyxy box")
                checked_boxes.append([x1, y1, x2, y2])
            self.rows.append({
                "image_path": (self.base / image_path).resolve(),
                "boxes_xyxy": checked_boxes,
                "line_no": line_no,
            })
        if not self.rows:
            raise ValueError(f"manifest contains no rows for split {split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        try:
            with Image.open(row["image_path"]) as image:
                image_tensor = to_tensor(image.convert("RGB"))
        except OSError as exc:
            raise ValueError(f"cannot load image at manifest line {row['line_no']}: {exc}") from exc
        boxes = torch.tensor(row["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        width, height = image_tensor.shape[2], image_tensor.shape[1]
        if len(boxes) and ((boxes[:, 2] > width).any() or (boxes[:, 3] > height).any()):
            raise ValueError(f"box extends outside image at manifest line {row['line_no']}")
        return image_tensor, target


def _collate(batch):
    return tuple(zip(*batch))


def _make_model(pretrained: bool = True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None if pretrained else None)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def _epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train()
    total = 0.0
    count = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            with torch.no_grad():
                losses = model(images, targets)
                loss = sum(losses.values())
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not initialise from Torchvision COCO-pretrained weights")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs and batch-size must be positive and learning-rate must be > 0")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    device = torch.device(device_name)
    try:
        train_data = PersonBoxDataset(args.manifest, "train")
        val_data = PersonBoxDataset(args.manifest, "validation")
    except ValueError as exc:
        parser.error(str(exc))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            collate_fn=_collate, num_workers=0)
    model = _make_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, momentum=0.9, weight_decay=0.0005,
    )
    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = _epoch(model, train_loader, device, optimizer)
        val_loss = _epoch(model, val_loader, device)
        print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "schema_version": 1,
                "architecture": "fasterrcnn_resnet50_fpn",
                "class_names": ["background", "person"],
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_loss": val_loss,
                "train_loss": train_loss,
                "device": device_name,
                "manifest": str(args.manifest.resolve()),
                "torch_version": torch.__version__,
            }, args.output)
            print(f"saved best checkpoint: {args.output}")
    print("Training complete. Evaluate on a separate held-out test match before making claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
