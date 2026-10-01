"""Validated YAML configuration and path expansion."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    model: str
    pretrained: bool
    dataset_yaml: Path
    data_root: Path
    runs_dir: Path
    imgsz: int
    batch: int
    epochs: int
    workers: int
    device: str | int | list[int]
    allow_cpu: bool
    amp: bool
    optimizer: str
    lr0: float
    seed: int
    fraction: float
    warmup_iterations: int
    iterations: int
    train_options: dict[str, Any] = field(default_factory=dict)
    source_images: Path | None = None
    source_labels: Path | None = None
    classes: Any = None
    dataset_name: str | None = None
    validation_fraction: float = 0.2
    link_files: bool = True
    save_period: int = -1
    plots: bool = True
    deterministic: bool = True
    run_tag: str = "amd_rx6800"
    conf: float = 0.25
    iou: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in self.__dict__.items()}


def _expand(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = os.path.expanduser(os.path.expandvars(value))
    pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")
    return pattern.sub(lambda match: os.environ.get(match.group(1), match.group(2) or ""), value)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> BenchmarkConfig:
    """Load config YAML and validate the fields needed by every entry point."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    raw = {key: _expand(value) for key, value in raw.items()}
    raw.update(overrides or {})
    required = {"mode", "model", "runs_dir", "imgsz", "batch", "epochs"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")
    dataset_fields = {"source_images", "source_labels", "output", "classes"}
    configured_dataset_fields = dataset_fields.intersection(raw)
    if configured_dataset_fields and configured_dataset_fields != dataset_fields:
        missing_dataset_fields = sorted(dataset_fields - configured_dataset_fields)
        raise ValueError(f"Missing required dataset preparation fields: {', '.join(missing_dataset_fields)}")
    prepare_dataset = configured_dataset_fields == dataset_fields
    if not prepare_dataset:
        missing_dataset_fields = sorted({"dataset_yaml", "data_root"} - raw.keys())
        if missing_dataset_fields:
            raise ValueError(f"Missing required dataset fields: {', '.join(missing_dataset_fields)}")
    if raw["mode"] not in {"smoke", "benchmark"}:
        raise ValueError("mode must be 'smoke' or 'benchmark'")
    for field in ("imgsz", "batch", "epochs"):
        if int(raw[field]) < 1:
            raise ValueError(f"{field} must be a positive integer")
    if not 0 < float(raw.get("fraction", 1.0)) <= 1:
        raise ValueError("fraction must be in (0, 1]")
    save_period = raw.get("save_period", -1)
    if isinstance(save_period, bool) or not isinstance(save_period, int) or save_period < -1:
        raise ValueError("save_period must be an integer greater than or equal to -1")
    for field_name in ("plots", "deterministic"):
        if field_name in raw and not isinstance(raw[field_name], bool):
            raise ValueError(f"{field_name} must be a boolean")
    run_tag = raw.get("run_tag", "amd_rx6800")
    if not isinstance(run_tag, str) or not run_tag.strip():
        raise ValueError("run_tag must be a non-empty string")
    train_options = raw.get("train_options", {})
    if not isinstance(train_options, dict) or any(not isinstance(key, str) for key in train_options):
        raise ValueError("train_options must be a YAML mapping of Ultralytics train argument names to values")
    reserved_train_options = {
        "data", "epochs", "imgsz", "batch", "device", "amp", "workers", "optimizer", "lr0", "seed",
        "deterministic", "fraction", "pretrained", "project", "name", "exist_ok", "plots", "verbose",
        "cache", "val", "save", "save_period",
    }
    conflicts = sorted(reserved_train_options.intersection(train_options))
    if conflicts:
        raise ValueError(f"train_options cannot override top-level training config fields: {', '.join(conflicts)}")
    if prepare_dataset:
        classes = raw["classes"]
        if not isinstance(classes, (list, dict)) or not classes:
            raise ValueError("classes must be a non-empty YAML list or mapping")
        validation_fraction = float(raw.get("validation_fraction", 0.2))
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        link_files = raw.get("link_files", True)
        if not isinstance(link_files, bool):
            raise ValueError("link_files must be a boolean")

        def resolve_config_path(value: Any) -> Path:
            path_value = Path(str(value)).expanduser()
            return (path_value if path_value.is_absolute() else config_path.parent / path_value).resolve()

        data_root = resolve_config_path(raw["output"])
        path_fields = {
            "dataset_yaml": data_root / "dataset.yaml",
            "data_root": data_root,
            "runs_dir": resolve_config_path(raw["runs_dir"]),
        }
        source_images = resolve_config_path(raw["source_images"])
        source_labels = resolve_config_path(raw["source_labels"])
    else:
        path_fields = {key: Path(str(raw[key])).expanduser() for key in ("dataset_yaml", "data_root", "runs_dir")}
        # Legacy configs use repository-relative paths; self-contained configs use config-relative paths.
        for key, value in path_fields.items():
            if not value.is_absolute():
                path_fields[key] = (config_path.parent.parent / value).resolve()
        classes = None
        validation_fraction = 0.2
        link_files = True
        source_images = None
        source_labels = None
    return BenchmarkConfig(
        mode=str(raw["mode"]), model=str(raw["model"]), pretrained=bool(raw.get("pretrained", True)),
        **path_fields, imgsz=int(raw["imgsz"]), batch=int(raw["batch"]), epochs=int(raw["epochs"]),
        workers=int(raw.get("workers", 4)), device=raw.get("device", 0), allow_cpu=bool(raw.get("allow_cpu", False)),
        amp=bool(raw.get("amp", True)), optimizer=str(raw.get("optimizer", "SGD")), lr0=float(raw.get("lr0", 0.01)),
        seed=int(raw.get("seed", 17)), fraction=float(raw.get("fraction", 1.0)),
        warmup_iterations=int(raw.get("warmup_iterations", 20)), iterations=int(raw.get("iterations", 100)),
        train_options=train_options,
        source_images=source_images, source_labels=source_labels,
        classes=classes, dataset_name=raw.get("dataset_name"), validation_fraction=validation_fraction,
        link_files=link_files,
        save_period=save_period, plots=raw.get("plots", True), deterministic=raw.get("deterministic", True),
        run_tag=run_tag,
        conf=float(raw.get("conf", 0.25)), iou=float(raw.get("iou", 0.7)),
    )
