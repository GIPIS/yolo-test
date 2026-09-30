#!/usr/bin/env python3
"""Explicitly download selected COCO 2017 image splits and Ultralytics labels."""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already downloaded: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination} (COCO train archive is ~19 GB)")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def safe_extract(archive: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target_resolved):
                raise ValueError(f"Unsafe zip path: {member.filename}")
        zf.extractall(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/coco"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--confirm-large-download", action="store_true", help="required before fetching the ~19 GB training archive")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if "train" in args.splits and not args.confirm_large_download:
        parser.error("train2017 is a ~19 GB download; pass --confirm-large-download to proceed")
    try:
        for split in args.splits:
            archive = root / "downloads" / f"{split}2017.zip"
            download(f"https://images.cocodataset.org/zips/{split}2017.zip", archive)
            safe_extract(archive, root / "images")
        labels = root / "downloads" / "coco2017labels.zip"
        download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip", labels)
        safe_extract(labels, root)
        yaml_path = root / "coco.yaml"
        class_lines = "\n".join(f"  {index}: {name}" for index, name in enumerate(COCO_NAMES))
        yaml_path.write_text(f"path: {root}\ntrain: images/train2017\nval: images/val2017\nnames:\n{class_lines}\n", encoding="utf-8")
        print(f"Dataset files ready. Config: {yaml_path}\nDownloaded archives remain under {root / 'downloads'}; remove them manually after validation if desired.")
    except Exception as exc:
        print(f"Download/preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
