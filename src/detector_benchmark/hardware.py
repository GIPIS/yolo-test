"""Optional PyTorch/ROCm hardware discovery with explicit GPU requirements."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any


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
        "memory_total_bytes": None, "torch_version": None, "hip_version": None,
        "cuda_available": False, "device_count": 0, "devices": [],
        "rocm_environment": {key: os.environ.get(key) for key in ROCM_ENV_KEYS if os.environ.get(key)},
        "ultralytics_version": None, "gpu_detected": False, "gpu_status": "GPU NOT detected",
    }
    try:
        import psutil
        info["memory_total_bytes"] = psutil.virtual_memory().total
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


def require_gpu(info: dict[str, Any], allow_cpu: bool = False, requested_device: str | int | None = None) -> str:
    """Return device spec; never permit implicit CPU fallback for a benchmark."""
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
        if index_text.isdigit() and int(index_text) >= int(info.get("device_count", 0)):
            raise RuntimeError(f"Configured GPU device {device} is unavailable; PyTorch reports {info.get('device_count', 0)} device(s).")
        if not (index_text.isdigit() or device == "cuda"):
            raise RuntimeError(f"Unsupported GPU device selection {device!r}; use a visible index such as 0 or explicit CPU mode.")
        return device
    if allow_cpu:
        return "cpu"
    details = info.get("torch_device_error") or "PyTorch reports no HIP/CUDA-visible GPU."
    raise RuntimeError(
        f"GPU NOT detected. Refusing GPU benchmark. {details} Install a PyTorch ROCm build "
        "matching the host ROCm stack, verify /dev/kfd and /dev/dri access, then rerun "
        "scripts/check_environment.py. CPU is allowed only with --allow-cpu or allow_cpu: true."
    )
