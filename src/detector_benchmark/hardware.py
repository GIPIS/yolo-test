"""Optional PyTorch/ROCm hardware discovery with explicit GPU requirements."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from .artifacts import git_commit


ROCM_ENV_KEYS = ("ROCM_HOME", "ROCM_PATH", "HIP_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION", "PYTORCH_ROCM_ARCH")


def detect_environment(torch_module: Any | None = None) -> dict[str, Any]:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:
            torch_module = None
    cpu_model = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith(("model name", "hardware")):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info: dict[str, Any] = {
        "python": sys.version, "os": platform.platform(), "kernel": platform.release(),
        "cpu": cpu_model, "cpu_count": os.cpu_count(),
        "memory_total_bytes": None, "memory_available_bytes": None, "memory_used_bytes": None,
        "memory_percent": None, "torch_version": None, "hip_version": None,
        "cuda_available": False, "device_count": 0, "devices": [],
        "rocm_environment": {key: os.environ.get(key) for key in ROCM_ENV_KEYS if os.environ.get(key)},
        "ultralytics_version": None, "gpu_detected": False, "gpu_status": "GPU NOT detected",
    }
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["memory_total_bytes"] = memory.total
        info["memory_available_bytes"] = memory.available
        info["memory_used_bytes"] = memory.used
        info["memory_percent"] = memory.percent
    except ImportError:
        pass
    try:
        from importlib.metadata import version
        info["ultralytics_version"] = version("ultralytics")
    except Exception:
        pass
    if torch_module is not None:
        info["torch_version"] = getattr(torch_module, "__version__", None)
        info["hip_version"] = getattr(getattr(torch_module, "version", None), "hip", None)
        try:
            info["cuda_available"] = bool(torch_module.cuda.is_available())
            info["device_count"] = int(torch_module.cuda.device_count())
            for index in range(info["device_count"]):
                device = {"index": index, "name": torch_module.cuda.get_device_name(index)}
                try:
                    props = torch_module.cuda.get_device_properties(index)
                    device["properties"] = {key: getattr(props, key) for key in ("total_memory", "multi_processor_count", "major", "minor") if hasattr(props, key)}
                except Exception as exc:
                    device["properties_error"] = str(exc)
                info["devices"].append(device)
        except Exception as exc:
            info["torch_device_error"] = str(exc)
    info["gpu_detected"] = bool(info["cuda_available"] and info["device_count"] > 0)
    info["gpu_status"] = "GPU detected" if info["gpu_detected"] else "GPU NOT detected"
    for command in ("rocminfo", "amd-smi", "rocm-smi"):
        executable = shutil.which(command)
        if executable:
            try:
                output = subprocess.run([executable], capture_output=True, text=True, timeout=8, check=False)
                info["system_gpu_tool"] = {"command": command, "returncode": output.returncode, "summary": output.stdout[:4000]}
            except Exception as exc:
                info["system_gpu_tool_error"] = str(exc)
            break
    return info


def require_gpu(
    info: dict[str, Any], allow_cpu: bool = False, requested_device: str | int | list[int] | None = None
) -> str:
    """Return device spec; never permit implicit CPU fallback for a benchmark."""
    if isinstance(requested_device, list):
        if not requested_device or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in requested_device):
            raise RuntimeError("GPU device lists must contain one or more non-negative integer indices.")
        if len(set(requested_device)) != len(requested_device):
            raise RuntimeError("GPU device lists cannot contain duplicate indices.")
        device = ",".join(str(index) for index in requested_device)
    else:
        device = str(requested_device if requested_device is not None else "0")
    if device == "cpu":
        if allow_cpu:
            return "cpu"
        raise RuntimeError("device is configured as CPU; set allow_cpu: true or explicitly override with --allow-cpu. This is not a GPU benchmark.")
    if info.get("gpu_detected"):
        if not info.get("hip_version"):
            if allow_cpu:
                return "cpu"
            raise RuntimeError("A GPU is visible, but this PyTorch build does not report HIP. Refusing to use a CUDA-only/non-ROCm build for an AMD benchmark.")
        index_text = device.split(":", 1)[1] if device.startswith("cuda:") else device
        if device == "cuda":
            return device
        indices = index_text.split(",")
        if any(not index.isdigit() for index in indices):
            raise RuntimeError(f"Unsupported GPU device selection {device!r}; use an index or a list such as [0, 1].")
        normalized_indices = [str(int(index)) for index in indices]
        if len(set(normalized_indices)) != len(normalized_indices):
            raise RuntimeError("GPU device selections cannot contain duplicate indices.")
        unavailable = [index for index in normalized_indices if int(index) >= int(info.get("device_count", 0))]
        if unavailable:
            raise RuntimeError(
                f"Configured GPU device(s) {', '.join(unavailable)} are unavailable; "
                f"PyTorch reports {info.get('device_count', 0)} device(s)."
            )
        normalized_selection = ",".join(normalized_indices)
        return normalized_selection if "," in normalized_selection else device
    if allow_cpu:
        return "cpu"
    details = info.get("torch_device_error") or "PyTorch reports no HIP/CUDA-visible GPU."
    raise RuntimeError(
        f"GPU NOT detected. Refusing GPU benchmark. {details} Install a PyTorch ROCm build "
        "matching the host ROCm stack, verify /dev/kfd and /dev/dri access, then rerun "
        "scripts/check_environment.py. CPU is allowed only with --allow-cpu or allow_cpu: true."
    )


def _selected_device_indices(device: str, torch_module: Any) -> list[int]:
    if device == "cpu":
        return []
    if device == "cuda":
        return [int(torch_module.cuda.current_device())]
    index_text = device.split(":", 1)[1] if device.startswith("cuda:") else device
    return [int(index) for index in index_text.split(",")]


def memory_stats(device: str = "0") -> dict[str, float | None]:
    if device == "cpu":
        return {"allocated_gib": None, "reserved_gib": None, "peak_allocated_gib": None}
    import torch

    indices = _selected_device_indices(device, torch)
    gib = 1024 ** 3
    return {
        "allocated_gib": sum(torch.cuda.memory_allocated(index) for index in indices) / gib,
        "reserved_gib": sum(torch.cuda.memory_reserved(index) for index in indices) / gib,
        "peak_allocated_gib": sum(torch.cuda.max_memory_allocated(index) for index in indices) / gib,
    }


def reset_peak_memory(device: str = "0") -> None:
    if device == "cpu":
        return
    import torch

    for index in _selected_device_indices(device, torch):
        torch.cuda.reset_peak_memory_stats(index)


def environment_record() -> dict[str, Any]:
    info = detect_environment()
    info["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    info["git_commit"] = git_commit()
    info["platform"] = platform.platform()
    info["python"] = sys.version
    return info
