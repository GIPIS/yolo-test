#!/usr/bin/env python3
"""Evaluate a checkpoint and persist standard detection accuracy metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector_benchmark.artifacts import write_json
from detector_benchmark.config import load_config
from detector_benchmark.evaluation import evaluate
from detector_benchmark.hardware import detect_environment, require_gpu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config or (args.run / "source_config.yaml"))
        env = detect_environment()
        device = require_gpu(env, allow_cpu=args.allow_cpu or config.allow_cpu, requested_device=config.device)
        checkpoint = args.checkpoint or args.run / "weights" / "best.pt"
        metrics = evaluate(checkpoint, config.dataset_yaml, config.imgsz, device, config.batch)
        write_json(args.run / "evaluation.json", metrics)
        print(f"Evaluation metrics saved to {args.run / 'evaluation.json'}: {metrics}")
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
