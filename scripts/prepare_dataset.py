#!/usr/bin/env python3
"""Prepare a YOLO detection dataset from YAML, validate it, or create a synthetic smoke dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector_benchmark.datasets import prepare_yolo_dataset, validate_yolo_dataset


def make_smoke(root: Path, train_count: int, val_count: int) -> Path:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow required to create smoke images. Install: pip install Pillow") from exc
    for split, count in (("train2017", train_count), ("val2017", val_count)):
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = Image.new("RGB", (96, 96), (20, 30, 40))
            draw = ImageDraw.Draw(image)
            offset = (index * 7) % 20
            draw.rectangle((24 + offset, 24, 64 + offset, 64), fill=(220, 80, 40))
            image.save(images / f"smoke_{index:04d}.jpg", quality=90)
            (labels / f"smoke_{index:04d}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(f"path: {root.resolve()}\ntrain: images/train2017\nval: images/val2017\nnames:\n  0: object\n", encoding="utf-8")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML with source_images, source_labels, output, and classes")
    parser.add_argument("--root", type=Path, help="output root; used only with --smoke")
    parser.add_argument("--smoke", action="store_true", help="create a deterministic 4/2-image toy dataset")
    parser.add_argument("--train-images", type=int, default=4)
    parser.add_argument("--val-images", type=int, default=2)
    args = parser.parse_args()
    if args.config and args.smoke:
        parser.error("--config and --smoke are separate preparation modes")
    if not args.config and not args.smoke:
        parser.error("provide --config for a dataset or --smoke to create the toy dataset")
    try:
        if args.config:
            manifest = prepare_yolo_dataset(args.config)
            root = Path(str(manifest["output"]))
            message = "Dataset split already prepared" if manifest["already_prepared"] else "Prepared dataset split"
            print(f"{message}: {manifest['counts']}")
            print(f"Dataset config: {root.resolve() / 'dataset.yaml'}")
        if args.smoke:
            root = args.root or Path("datasets/smoke")
            yaml_path = make_smoke(root, args.train_images, args.val_images)
            print(f"Created smoke dataset and config: {yaml_path}")
        counts = validate_yolo_dataset(root)
        print(f"Dataset validated: {counts}")
    except Exception as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
