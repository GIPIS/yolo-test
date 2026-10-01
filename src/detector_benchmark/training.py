"""Ultralytics training and result extraction kept behind a small adapter."""
from __future__ import annotations

import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from .benchmarking import memory_stats, reset_peak_memory
from .config import BenchmarkConfig


def train(config: BenchmarkConfig, output_dir: Path, device: str) -> dict[str, Any]:
    device_count = len(device.split(",")) if device != "cpu" else 0
    if device_count > 1 and config.batch % device_count:
        raise ValueError(f"batch must be divisible by the number of selected GPUs ({device_count}); got {config.batch}.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is missing. Install with pip install -e '.[benchmark]'.") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.model)
    epoch_started: list[float] = []
    epoch_durations: list[float] = []
    validation_started: list[float] = []
    validation_durations: list[float] = []

    def on_epoch_start(trainer: Any) -> None:
        epoch_started.append(time.perf_counter())

    def on_epoch_end(trainer: Any) -> None:
        if epoch_started:
            epoch_durations.append(time.perf_counter() - epoch_started[-1])

    def on_validation_start(trainer: Any) -> None:
        validation_started.append(time.perf_counter())

    def on_validation_end(trainer: Any) -> None:
        if validation_started:
            validation_durations.append(time.perf_counter() - validation_started.pop())

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.add_callback("on_val_start", on_validation_start)
    model.add_callback("on_val_end", on_validation_end)
    reset_peak_memory(device)
    started = time.perf_counter()
    try:
        train_args = {
            "data": str(config.dataset_yaml), "epochs": config.epochs, "imgsz": config.imgsz, "batch": config.batch,
            "device": device, "amp": config.amp, "workers": config.workers, "optimizer": config.optimizer, "lr0": config.lr0,
            "seed": config.seed, "deterministic": config.deterministic, "fraction": config.fraction, "pretrained": config.pretrained,
            "project": str(output_dir.parent), "name": output_dir.name, "exist_ok": True, "plots": config.plots,
            "verbose": True, "cache": False, "val": True, "save": True, "save_period": config.save_period,
        }
        train_args.update(config.train_options)
        result = model.train(**train_args)
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
        dataset_metadata = yaml.safe_load(config.dataset_yaml.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        dataset_metadata = {}
    pure_train_seconds = sum(epoch_durations)
    validation_seconds = sum(validation_durations)
    images_per_second = dataset_size * config.fraction * config.epochs / pure_train_seconds if dataset_size and pure_train_seconds > 0 else None
    images_per_second_wall = dataset_size * config.fraction * config.epochs / elapsed if dataset_size and elapsed > 0 else None
    time_per_epoch = statistics.fmean(epoch_durations) if epoch_durations else None
    return {"status": "completed", "model": config.model, "checkpoint": str(actual_dir / "weights" / "best.pt"),
            "total_training_seconds": elapsed, "pure_train_seconds": pure_train_seconds,
            "validation_seconds": validation_seconds, "time_per_epoch_seconds": time_per_epoch,
            "epoch_times_seconds": epoch_durations,
            "last_epoch_delta_seconds": epoch_durations[-1] if epoch_durations else None,
            "approx_images_per_second": images_per_second, "approx_images_per_second_wall": images_per_second_wall,
            "parameters": parameters,
            "batch": config.batch, "imgsz": config.imgsz, "epochs": config.epochs, "amp": config.amp,
            "deterministic": config.deterministic, "plots": config.plots, "save_period": config.save_period,
            "optimizer": config.optimizer, "lr0": config.lr0, "dataset_yaml": str(config.dataset_yaml),
            "dataset_size_images": dataset_size, "dataset_fraction": config.fraction,
            "dataset_version": dataset_metadata.get(
                "dataset_name", "synthetic-smoke" if config.mode == "smoke" else config.dataset_yaml.stem
            ),
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
