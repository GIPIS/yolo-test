#!/usr/bin/env python3
"""Measure warm inference latency and throughput for explicitly selected settings."""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector_benchmark.artifacts import run_name, write_json
from detector_benchmark.benchmarking import (
    environment_record,
    mean_result_phase_times,
    memory_stats,
    preload_images,
    release_gpu_memory,
    reset_peak_memory,
    synchronize,
)
from detector_benchmark.config import load_config
from detector_benchmark.datasets import list_images
from detector_benchmark.hardware import require_gpu
from detector_benchmark.inference import UltralyticsDetector
from detector_benchmark.metrics import summarize
from detector_benchmark.reporting import write_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", action="append", help="repeat to test selected models; defaults to config model")
    parser.add_argument("--imgsz", type=int, action="append", help="repeat for selected resolutions")
    parser.add_argument("--batches", type=int, nargs="+", help="selected batch sizes (no automatic matrix)")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--preload", action="store_true", help="decode benchmark input images into memory before timing")
    args = parser.parse_args()
    config = load_config(args.config)
    env = environment_record()
    output = config.runs_dir / run_name("inference-" + config.model, tag=config.run_tag)
    output_created = False
    try:
        output.mkdir(parents=True)
        output_created = True
        shutil.copy2(args.config, output / "source_config.yaml")
        write_json(output / "environment.json", env)
        device = require_gpu(env, allow_cpu=args.allow_cpu or config.allow_cpu, requested_device=config.device)
        images = list_images(config.data_root, "val2017")
        if not images:
            raise FileNotFoundError(f"No validation images found under {config.data_root / 'images/val2017'}; prepare dataset first.")
        all_results = []
        for model_name in args.model or [config.model]:
            for imgsz in args.imgsz or [config.imgsz]:
                for batch in args.batches or [config.batch]:
                    if batch < 1:
                        raise ValueError("batch sizes must be positive")
                    detector = None
                    case_batches = []
                    image_cache = {}
                    predictions = None
                    batch_images = None
                    try:
                        detector = UltralyticsDetector(model_name, device)
                        detector.load()
                        model_parameters = sum(parameter.numel() for parameter in detector.model.model.parameters())
                        samples: dict[str, list[float]] = {key: [] for key in ("preprocess", "inference", "postprocess", "total")}
                        case_batches = [
                            [images[(iteration * batch + offset) % len(images)] for offset in range(batch)]
                            for iteration in range(config.warmup_iterations + config.iterations)
                        ]
                        if args.preload:
                            image_cache = preload_images(list({image for case_batch in case_batches for image in case_batch}))
                            case_batches = [[image_cache[image] for image in case_batch] for case_batch in case_batches]
                        reset_peak_memory(device)
                        for batch_images in case_batches[:config.warmup_iterations]:
                            detector.batch_predict(batch_images, batch=batch, imgsz=imgsz, conf=config.conf, iou=config.iou)
                        synchronize(device)
                        for batch_images in case_batches[config.warmup_iterations:]:
                            synchronize(device)
                            started = time.perf_counter()
                            predictions = detector.batch_predict(batch_images, batch=batch, imgsz=imgsz, conf=config.conf, iou=config.iou)
                            synchronize(device)
                            wall_ms_per_image = (time.perf_counter() - started) * 1000 / batch
                            samples["total"].append(wall_ms_per_image)
                            for phase, phase_ms in mean_result_phase_times(predictions).items():
                                samples[phase].append(phase_ms)
                        record = {"status": "completed", "model": model_name, "checkpoint": model_name, "parameters": model_parameters,
                                  "imgsz": imgsz, "batch": batch, "warmup_iterations": config.warmup_iterations,
                                  "iterations": config.iterations, "latency_ms": {key: summarize(values) for key, values in samples.items()},
                                  "images_per_second": 1000 * batch / (statistics.fmean(samples["total"]) * batch) if samples["total"] else None,
                                  "gpu_memory": memory_stats(device), "input_preloaded": args.preload,
                                  "timing_semantics": {"total": "synchronized end-to-end predict wall time per image; includes disk read/decode unless --preload is set",
                                                       "phases": "Ultralytics model-side preprocessing/inference/postprocessing per image; aggregated across all Results in each batch"}}
                    except Exception as exc:
                        if "out of memory" in str(exc).lower() or "hiperroroutofmemory" in str(exc).lower():
                            record = {"status": "FAILED", "reason": "CUDA/HIP out of memory", "detail": str(exc), "model": model_name, "batch": batch, "imgsz": imgsz}
                        else:
                            raise
                    finally:
                        if detector is not None:
                            del detector
                        case_batches.clear()
                        image_cache.clear()
                        predictions = None
                        batch_images = None
                        release_gpu_memory(device)
                    all_results.append(record)
                    write_json(output / "benchmark.json", {"status": "completed" if all(item["status"] == "completed" for item in all_results) else "FAILED", "results": all_results})
        payload = {"status": "completed" if all(item["status"] == "completed" for item in all_results) else "FAILED", "model": args.model or config.model, "hardware": env["devices"], "torch_version": env["torch_version"], "hip_version": env["hip_version"], "inference": all_results[0] if len(all_results) == 1 else {}, "results": all_results, "dataset": str(config.dataset_yaml)}
        write_json(output / "results.json", payload)
        write_summary(output / "benchmark_summary.md", payload)
        print(f"Inference results saved in {output}")
        return 0 if payload["status"] == "completed" else 1
    except Exception as exc:
        if output_created:
            failed = {"status": "FAILED", "reason": str(exc), "model": args.model or config.model, "imgsz": args.imgsz or [config.imgsz], "batches": args.batches or [config.batch]}
            write_json(output / "results.json", failed)
            write_json(output / "benchmark.json", failed)
            write_summary(output / "benchmark_summary.md", failed)
        print(f"Inference benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
