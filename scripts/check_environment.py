#!/usr/bin/env python3
"""Print hardware and software diagnostics; exit nonzero when GPU is unavailable."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector_benchmark.hardware import detect_environment


def main() -> int:
    info = detect_environment()
    print(json.dumps(info, indent=2, default=str))
    print(f"\n{info['gpu_status']}")
    if not info["gpu_detected"]:
        print("Action: install a matching PyTorch ROCm build, check AMD GPU/OS support and /dev/kfd,/dev/dri permissions; see README troubleshooting.", file=sys.stderr)
        return 1
    if not info.get("hip_version"):
        print("WARNING: a GPU is visible but torch.version.hip is empty; this may be a CUDA build, not ROCm.", file=sys.stderr)
        return 1
    print(f"GPU ready via PyTorch HIP {info['hip_version']} (PyTorch uses torch.cuda APIs for HIP).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
