"""COCO and tiny smoke dataset location helpers."""
from __future__ import annotations

from pathlib import Path


def validate_yolo_dataset(root: str | Path, splits: tuple[str, ...] = ("train2017", "val2017")) -> dict[str, int]:
    """Validate image and label directories without importing a training framework."""
    root = Path(root).expanduser().resolve()
    counts: dict[str, int] = {}
    for split in splits:
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} data under {root}; expected {image_dir} and {label_dir}")
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            raise ValueError(f"No images found in {image_dir}")
        missing = [image.stem for image in images if not (label_dir / f"{image.stem}.txt").is_file()]
        if missing:
            raise ValueError(f"{len(missing)} labels missing for {split}; first missing image: {missing[0]}")
        counts[split] = len(images)
    return counts


def list_images(root: str | Path, split: str = "val2017") -> list[Path]:
    directory = Path(root) / "images" / split
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
