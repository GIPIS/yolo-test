"""Dataset validation and generic YOLO detection dataset preparation."""
from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def validate_yolo_dataset(root: str | Path, splits: tuple[str, ...] = ("train2017", "val2017")) -> dict[str, int]:
    """Validate image and label directories without importing a training framework."""
    root = Path(root).expanduser().resolve()
    counts: dict[str, int] = {}
    for split in splits:
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} data under {root}; expected {image_dir} and {label_dir}")
        images = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            raise ValueError(f"No images found in {image_dir}")
        missing = [image.stem for image in images if not (label_dir / f"{image.stem}.txt").is_file()]
        if missing:
            raise ValueError(f"{len(missing)} labels missing for {split}; first missing image: {missing[0]}")
        counts[split] = len(images)
    return counts


def _resolve_config_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_dir / path).resolve()


def _class_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        names = {index: str(name) for index, name in enumerate(value)}
    elif isinstance(value, dict):
        try:
            names = {int(index): str(name) for index, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("classes mapping keys must be consecutive integer IDs starting at 0") from exc
    else:
        raise ValueError("classes must be a YAML list or mapping of class IDs to names")
    if not names or sorted(names) != list(range(len(names))) or any(not name.strip() for name in names.values()):
        raise ValueError("classes must contain non-empty names with consecutive integer IDs starting at 0")
    return names


def prepare_yolo_dataset(config_path: str | Path) -> dict[str, Any]:
    """Create a reproducible YOLO detection split from configured image and YOLO txt directories.

    Required config keys: source_images, source_labels, output, classes.
    Optional keys: dataset_name, validation_fraction, seed, link_files.
    """
    config_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Dataset preparation config must be a YAML mapping")
    required = {"source_images", "source_labels", "output", "classes"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing dataset preparation config fields: {', '.join(missing)}")

    config_dir = config_path.parent
    image_dir = _resolve_config_path(raw["source_images"], config_dir)
    label_dir = _resolve_config_path(raw["source_labels"], config_dir)
    output = _resolve_config_path(raw["output"], config_dir)
    classes = _class_names(raw["classes"])
    dataset_name = str(raw.get("dataset_name", config_path.stem))
    val_fraction = float(raw.get("validation_fraction", 0.2))
    seed = int(raw.get("seed", 17))
    link_files = bool(raw.get("link_files", True))
    if not 0 < val_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Configured source_images/source_labels directories do not exist: {image_dir}, {label_dir}")
    if output in (image_dir, label_dir) or image_dir in output.parents or label_dir in output.parents:
        raise ValueError("output must not be inside either source directory")

    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(images) < 2:
        raise ValueError(f"At least two images are required in {image_dir}")
    label_files = {path.stem: path for path in label_dir.glob("*.txt")}
    image_stems = {path.stem for path in images}
    missing_labels = sorted(image_stems - label_files.keys())
    extra_labels = sorted(label_files.keys() - image_stems)
    if missing_labels:
        raise FileNotFoundError(f"{len(missing_labels)} images have no matching YOLO label file; first: {missing_labels[0]}")
    if extra_labels:
        raise ValueError(f"{len(extra_labels)} label files have no matching image; first: {extra_labels[0]}")

    pairs: list[tuple[Path, Path]] = []
    for image in images:
        label = label_files[image.stem]
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO row in {label}:{line_number}; expected class_id x_center y_center width height")
            try:
                class_id = int(fields[0])
                x_center, y_center, width, height = (float(item) for item in fields[1:])
            except ValueError as exc:
                raise ValueError(f"Invalid numeric YOLO row in {label}:{line_number}") from exc
            if class_id not in classes:
                raise ValueError(f"Class ID {class_id} in {label}:{line_number} is not declared in config classes")
            if not all(math.isfinite(value) for value in (x_center, y_center, width, height)) or not (
                0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1
            ):
                raise ValueError(f"Invalid normalized bounding box in {label}:{line_number}")
        pairs.append((image, label))

    manifest_path = output / "split_manifest.json"
    if output.exists() and any(output.iterdir()):
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing_data = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8")) or {}
                existing_classes = _class_names(existing_data.get("names"))
            except (OSError, json.JSONDecodeError, yaml.YAMLError, ValueError):
                existing, existing_classes = {}, {}
            # Also recognize splits made by the previous implementation, without dataset-specific branches.
            legacy_match = (
                existing.get("source") == str(image_dir.parent)
                and existing.get("seed") == seed
                and existing.get("validation_fraction") == val_fraction
            )
            generic_match = (
                existing.get("source_images") == str(image_dir)
                and existing.get("source_labels") == str(label_dir)
                and existing.get("output") == str(output)
                and existing.get("seed") == seed
                and existing.get("validation_fraction") == val_fraction
            )
            if existing_classes == classes and (legacy_match or generic_match):
                validate_yolo_dataset(output)
                return {**existing, "already_prepared": True}
        raise FileExistsError(f"Output directory has existing files and does not match this config: {output}; choose another output path")

    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    splits = {"val2017": shuffled[:val_count], "train2017": shuffled[val_count:]}
    output.mkdir(parents=True, exist_ok=True)
    for split, split_pairs in splits.items():
        images_out, labels_out = output / "images" / split, output / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image, label in split_pairs:
            for source_file, target in ((image, images_out / image.name), (label, labels_out / label.name)):
                if link_files:
                    target.symlink_to(source_file)
                else:
                    shutil.copy2(source_file, target)

    dataset = {
        "dataset_name": dataset_name,
        "path": str(output),
        "train": "images/train2017",
        "val": "images/val2017",
        "names": classes,
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    manifest = {
        "dataset": dataset_name,
        "source_images": str(image_dir),
        "source_labels": str(label_dir),
        "output": str(output),
        "split_method": "seeded image-level shuffle",
        "seed": seed,
        "validation_fraction": val_fraction,
        "link_files": link_files,
        "counts": {split: len(items) for split, items in splits.items()},
        "files": {split: [image.name for image, _ in items] for split, items in splits.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "already_prepared": False}
