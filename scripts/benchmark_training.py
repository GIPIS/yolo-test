#!/usr/bin/env python3
"""Run selected training cases; no full matrix is run unless explicitly requested."""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector_benchmark.artifacts import run_name, write_json
from detector_benchmark.benchmarking import environment_record, release_gpu_memory
from detector_benchmark.config import load_config
from detector_benchmark.hardware import require_gpu
from detector_benchmark.reporting import write_summary
from detector_benchmark.training import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", help="explicit model list, e.g. yolo26n.pt yolo26s.pt")
    parser.add_argument("--imgsz", nargs="+", type=int, help="explicit resolution list, e.g. 640 960")
    parser.add_argument("--batches", nargs="+", type=int, help="explicit batch list, e.g. 1 4 8")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    env = environment_record()
    root = config.runs_dir / run_name("training-matrix", tag=config.run_tag)
    root_created = False
    try:
        root.mkdir(parents=True)
        root_created = True
        shutil.copy2(args.config, root / "source_config.yaml")
        write_json(root / "environment.json", env)
        device = require_gpu(env, allow_cpu=args.allow_cpu or config.allow_cpu, requested_device=config.device)
        records = []
        for model in args.models or [config.model]:
            for imgsz in args.imgsz or [config.imgsz]:
                for batch in args.batches or [config.batch]:
                    case = replace(config, model=model, imgsz=imgsz, batch=batch, epochs=args.epochs or config.epochs)
                    case_dir = root / run_name(f"{Path(model).stem}-{imgsz}-{batch}", tag=config.run_tag)
                    case_dir.mkdir()
                    write_json(case_dir / "config.json", {**case.to_dict(), "device_used": device})
                    try:
                        result = train(case, case_dir, device)
                    except Exception as exc:
                        oom = "out of memory" in str(exc).lower() or "hiperroroutofmemory" in str(exc).lower()
                        result = {"status": "FAILED", "reason": "CUDA/HIP out of memory" if oom else str(exc), "model": model, "batch": batch, "imgsz": imgsz}
                    finally:
                        release_gpu_memory(device)
                    write_json(case_dir / "results.json", result)
                    records.append(result)
                    write_json(root / "benchmark.json", {"status": "completed" if all(item["status"] == "completed" for item in records) else "FAILED", "results": records})
        payload = {"status": "completed" if all(item["status"] == "completed" for item in records) else "FAILED", "hardware": env["devices"], "torch_version": env["torch_version"], "hip_version": env["hip_version"], "dataset": str(config.dataset_yaml), "results": records}
        write_json(root / "results.json", payload)
        write_summary(root / "benchmark_summary.md", payload)
        print(f"Training benchmark artifacts: {root}")
        return 0 if payload["status"] == "completed" else 1
    except Exception as exc:
        if root_created:
            failed = {"status": "FAILED", "reason": str(exc), "model": config.model, "batch": config.batch, "imgsz": config.imgsz, "epochs": config.epochs}
            write_json(root / "results.json", failed)
            write_json(root / "benchmark.json", failed)
            write_summary(root / "benchmark_summary.md", failed)
        print(f"Training benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
