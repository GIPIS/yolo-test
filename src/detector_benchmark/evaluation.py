"""COCO-compatible Ultralytics validation wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate(checkpoint: str | Path, data: str | Path, imgsz: int, device: str = "0", batch: int = 1) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is missing. Install with pip install -e '.[benchmark]'.") from exc
    metrics = YOLO(str(checkpoint)).val(data=str(data), imgsz=imgsz, device=device, batch=batch, plots=True, verbose=True)
    box = metrics.box
    return {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50),
            "map50_95": float(box.map), "per_class_map50_95": [float(value) for value in box.maps]}
