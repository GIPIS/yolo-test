"""Validated YAML configuration and path expansion."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
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
    device: str | int
    allow_cpu: bool
    amp: bool
    optimizer: str
    lr0: float
    seed: int
    fraction: float
    warmup_iterations: int
    iterations: int
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
    required = {"mode", "model", "dataset_yaml", "data_root", "runs_dir", "imgsz", "batch", "epochs"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")
    if raw["mode"] not in {"smoke", "benchmark"}:
        raise ValueError("mode must be 'smoke' or 'benchmark'")
    for field in ("imgsz", "batch", "epochs"):
        if int(raw[field]) < 1:
            raise ValueError(f"{field} must be a positive integer")
    if not 0 < float(raw.get("fraction", 1.0)) <= 1:
        raise ValueError("fraction must be in (0, 1]")
    path_fields = {key: Path(str(raw[key])).expanduser() for key in ("dataset_yaml", "data_root", "runs_dir")}
    # Relative paths are interpreted from the repository, not the caller's cwd.
    for key, value in path_fields.items():
        if not value.is_absolute():
            path_fields[key] = (config_path.parent.parent / value).resolve()
    return BenchmarkConfig(
        mode=str(raw["mode"]), model=str(raw["model"]), pretrained=bool(raw.get("pretrained", True)),
        **path_fields, imgsz=int(raw["imgsz"]), batch=int(raw["batch"]), epochs=int(raw["epochs"]),
        workers=int(raw.get("workers", 4)), device=raw.get("device", 0), allow_cpu=bool(raw.get("allow_cpu", False)),
        amp=bool(raw.get("amp", True)), optimizer=str(raw.get("optimizer", "SGD")), lr0=float(raw.get("lr0", 0.01)),
        seed=int(raw.get("seed", 17)), fraction=float(raw.get("fraction", 1.0)),
        warmup_iterations=int(raw.get("warmup_iterations", 20)), iterations=int(raw.get("iterations", 100)),
        conf=float(raw.get("conf", 0.25)), iou=float(raw.get("iou", 0.7)),
    )
