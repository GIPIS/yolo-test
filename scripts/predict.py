#!/usr/bin/env python3
"""Run object detection on an image or image directory and save annotated results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector_benchmark.inference import UltralyticsDetector


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _unit_float(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _print_detections(result: Any) -> None:
    boxes = getattr(result, "boxes", None)
    image_path = getattr(result, "path", "image")
    if boxes is None or len(boxes) == 0:
        print(f"{image_path}: no detections")
        return

    coordinates = _values(boxes.xyxy)
    class_ids = _values(boxes.cls)
    confidences = _values(boxes.conf)
    names = getattr(result, "names", {})
    print(f"{image_path}: {len(coordinates)} detection(s)")
    for class_id, confidence, (x1, y1, x2, y2) in zip(class_ids, confidences, coordinates):
        label = _class_name(names, int(class_id))
        print(
            f"  {label} confidence={float(confidence):.3f} "
            f"xyxy=({float(x1):.1f}, {float(y1):.1f}, {float(x2):.1f}, {float(y2):.1f})"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Ultralytics detection checkpoint (.pt)")
    parser.add_argument("--source", type=Path, required=True, help="Input image or directory of images")
    parser.add_argument("--output", type=Path, required=True, help="Annotated image path or output directory")
    parser.add_argument("--imgsz", type=_positive_int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--conf", type=_unit_float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=_unit_float, default=0.7, help="NMS IoU threshold (default: 0.7)")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu (default: 0)")
    parser.add_argument("--batch", type=_positive_int, default=1, help="Prediction batch size (default: 1)")
    parser.add_argument("--classes", type=int, nargs="+", help="Only predict these class IDs")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        model_path = args.model.expanduser().resolve()
        source_path = args.source.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
        if not source_path.exists():
            raise FileNotFoundError(f"Prediction source does not exist: {source_path}")
        if source_path.is_file():
            if source_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image type: {source_path.suffix}")
            if output_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError("For a single image, --output must be an image filename such as detected.jpg")
            source_images = [source_path]
        elif source_path.is_dir():
            if output_path.suffix:
                raise ValueError("For a directory source, --output must be a directory")
            source_images = sorted(
                path for path in source_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                and not path.is_relative_to(output_path)
            )
            if not source_images:
                raise ValueError(f"No supported images found in {source_path}")
        else:
            raise ValueError(f"Source must be an image file or directory: {source_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True) if source_path.is_file() else output_path.mkdir(parents=True, exist_ok=True)
        if source_path.is_file() and output_path == source_path:
            raise ValueError("Output image must not overwrite the source image")

        detector = UltralyticsDetector(model_path, device=args.device)
        prediction_options: dict[str, Any] = {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
        }
        if args.classes is not None:
            prediction_options["classes"] = args.classes
        results = detector.predict(
            [str(path) for path in source_images], batch=args.batch, **prediction_options
        )

        for result in results:
            _print_detections(result)
            result_path = Path(result.path).resolve()
            if source_path.is_dir():
                try:
                    relative_path = result_path.relative_to(source_path)
                except ValueError:
                    relative_path = Path(result_path.name)
                destination = output_path / relative_path
            else:
                destination = output_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.save(filename=str(destination))
            print(f"  saved: {destination}")
        return 0
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())