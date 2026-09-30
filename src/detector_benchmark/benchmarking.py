"""Benchmark metadata, GPU timing synchronization and OOM classification."""
from __future__ import annotations

import gc
import platform
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import git_commit
from .hardware import detect_environment


def synchronize(device: str = "0") -> None:
    if device != "cpu":
        import torch
        torch.cuda.synchronize(int(device) if device.isdigit() else None)


def memory_stats(device: str = "0") -> dict[str, float | None]:
    if device == "cpu":
        return {"allocated_gib": None, "reserved_gib": None, "peak_allocated_gib": None}
    import torch
    index = int(device) if device.isdigit() else torch.cuda.current_device()
    gib = 1024 ** 3
    return {"allocated_gib": torch.cuda.memory_allocated(index) / gib,
            "reserved_gib": torch.cuda.memory_reserved(index) / gib,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / gib}


def reset_peak_memory(device: str = "0") -> None:
    if device != "cpu":
        import torch
        torch.cuda.reset_peak_memory_stats(int(device) if device.isdigit() else None)


def release_gpu_memory(device: str = "0") -> None:
    """Collect Python objects and release cached accelerator allocations after a case."""
    gc.collect()
    if device == "cpu":
        return
    import torch
    try:
        torch.cuda.synchronize()
    except Exception:
        # A prior device error (for example OOM) can surface during synchronization.
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        # Cleanup is best-effort and must not hide the case's original failure.
        pass


def mean_result_phase_times(results: Any) -> dict[str, float]:
    """Average Ultralytics per-image phase timings over every result in one batch."""
    phases = ("preprocess", "inference", "postprocess")
    if not results:
        return {phase: 0.0 for phase in phases}
    return {
        phase: sum(float((result.speed or {}).get(phase, 0.0)) for result in results) / len(results)
        for phase in phases
    }


def preload_images(paths: Iterable[str | Path]) -> dict[Path, Any]:
    """Decode image paths into OpenCV BGR arrays for optional in-memory prediction timing."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--preload requires OpenCV (installed with Ultralytics)") from exc
    loaded: dict[Path, Any] = {}
    for value in paths:
        path = Path(value).expanduser()
        if path not in loaded:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not decode image for --preload: {path}")
            loaded[path] = image
    return loaded


def is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, MemoryError) or "out of memory" in text or "hiperroroutofmemory" in text or "cuda error: out of memory" in text


def environment_record() -> dict[str, Any]:
    info = detect_environment()
    info["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    info["git_commit"] = git_commit()
    info["platform"] = platform.platform()
    info["python"] = sys.version
    return info
