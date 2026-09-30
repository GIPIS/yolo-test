"""Ultralytics training and result extraction kept behind a small adapter."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from .benchmarking import memory_stats, reset_peak_memory
from .config import BenchmarkConfig


def train(config: BenchmarkConfig, output_dir: Path, device: str) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is missing. Install with pip install -e '.[benchmark]'.") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.model)
    epoch_started: list[float] = []
    epoch_durations: list[float] = []

    def on_epoch_start(trainer: Any) -> None:
        epoch_started.append(time.perf_counter())

    def on_epoch_end(trainer: Any) -> None:
        if epoch_started:
            epoch_durations.append(time.perf_counter() - epoch_started[-1])

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_epoch_end", on_epoch_end)
    reset_peak_memory(device)
    started = time.perf_counter()
    try:
        result = model.train(
            data=str(config.dataset_yaml), epochs=config.epochs, imgsz=config.imgsz, batch=config.batch,
            device=device, amp=config.amp, workers=config.workers, optimizer=config.optimizer, lr0=config.lr0,
            seed=config.seed, deterministic=True, fraction=config.fraction, pretrained=config.pretrained,
            project=str(output_dir.parent), name=output_dir.name, exist_ok=True, plots=True,
            verbose=True, cache=False, val=True, save=True, save_period=1,
        )
    except Exception as exc:
        if "out of memory" in str(exc).lower() or "hiperroroutofmemory" in str(exc).lower():
            raise RuntimeError(f"FAILED: CUDA/HIP out of memory; model={config.model}, batch={config.batch}, imgsz={config.imgsz}. Parameters were not changed.") from exc
        raise
    elapsed = time.perf_counter() - started
    actual_dir = Path(getattr(result, "save_dir", output_dir))
    csv_source = actual_dir / "results.csv"
    if csv_source.exists() and csv_source.resolve() != (output_dir / "training_metrics.csv").resolve():
        shutil.copy2(csv_source, output_dir / "training_metrics.csv")
    metrics = getattr(result, "results_dict", {}) or {}
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    dataset_size = _count_training_images(config)
    try:
        images_per_second = dataset_size * config.fraction * config.epochs / elapsed if dataset_size else None
    except ZeroDivisionError:
        images_per_second = None
    return {"status": "completed", "model": config.model, "checkpoint": str(actual_dir / "weights" / "best.pt"),
            "total_training_seconds": elapsed, "time_per_epoch_seconds": elapsed / config.epochs,
            "epoch_times_seconds": epoch_durations,
            "last_epoch_delta_seconds": epoch_durations[-1] if epoch_durations else None,
            "approx_images_per_second": images_per_second, "parameters": parameters,
            "batch": config.batch, "imgsz": config.imgsz, "epochs": config.epochs, "amp": config.amp,
            "optimizer": config.optimizer, "lr0": config.lr0, "dataset_yaml": str(config.dataset_yaml),
            "dataset_size_images": dataset_size, "dataset_fraction": config.fraction,
            "dataset_version": "synthetic-smoke" if config.mode == "smoke" else "COCO 2017",
            "accuracy": metrics, "gpu_memory": memory_stats(device),
            "ultralytics_run_dir": str(actual_dir)}


def _count_training_images(config: BenchmarkConfig) -> int | None:
    try:
        dataset = yaml.safe_load(config.dataset_yaml.read_text(encoding="utf-8"))
        root = Path(dataset.get("path", config.data_root))
        if not root.is_absolute():
            root = config.dataset_yaml.parent / root
        train_paths = dataset.get("train")
        paths = train_paths if isinstance(train_paths, list) else [train_paths]
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        count = 0
        for value in paths:
            if not value:
                continue
            train_path = Path(value)
            if not train_path.is_absolute():
                train_path = root / train_path
            if train_path.is_dir():
                count += sum(1 for item in train_path.rglob("*") if item.suffix.lower() in image_extensions)
            elif train_path.is_file():
                count += sum(1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip())
        return count or None
    except (OSError, TypeError, AttributeError, yaml.YAMLError):
        return None
