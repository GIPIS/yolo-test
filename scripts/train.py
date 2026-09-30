#!/usr/bin/env python3
"""Train a configured Ultralytics detector and persist reproducibility artifacts."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector_benchmark.artifacts import git_commit, run_name, write_json
from detector_benchmark.benchmarking import environment_record
from detector_benchmark.config import load_config
from detector_benchmark.hardware import require_gpu
from detector_benchmark.reporting import write_summary
from detector_benchmark.training import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="explicitly permit CPU execution; not a GPU benchmark")
    args = parser.parse_args()
    config = load_config(args.config)
    env = environment_record()
    name = run_name(config.model, tag=config.run_tag)
    run_dir = config.runs_dir / name
    run_created = False
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        run_created = True
        shutil.copy2(args.config, run_dir / "source_config.yaml")
        write_json(run_dir / "config.json", {**config.to_dict(), "git_commit": git_commit(), "device_requested": config.device})
        write_json(run_dir / "environment.json", env)
        device = require_gpu(env, allow_cpu=args.allow_cpu or config.allow_cpu, requested_device=config.device)
        # Keep the earlier device_requested record and add the resolved device_used value.
        write_json(run_dir / "config.json", {**config.to_dict(), "git_commit": git_commit(), "device_used": device})
        metrics = train(config, run_dir, device)
        write_json(run_dir / "results.json", metrics)
        write_summary(run_dir / "benchmark_summary.md", {**metrics, "hardware": env["devices"], "torch_version": env["torch_version"], "hip_version": env["hip_version"], "dataset": str(config.dataset_yaml)})
        print(f"Training complete: {run_dir}")
        return 0
    except Exception as exc:
        if run_created:
            failed = {"status": "FAILED", "reason": str(exc), "model": config.model,
                      "batch": config.batch, "imgsz": config.imgsz, "epochs": config.epochs,
                      "dataset_yaml": str(config.dataset_yaml)}
            write_json(run_dir / "results.json", failed)
            write_summary(run_dir / "benchmark_summary.md", failed)
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
