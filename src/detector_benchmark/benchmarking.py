"""Benchmark metadata, GPU timing synchronization and OOM classification."""
from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
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
