#!/usr/bin/env python3
"""Copy native PyTorch checkpoint and optionally export an ONNX model."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector_benchmark.artifacts import write_json
from detector_benchmark.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--format", choices=("onnx", "pt", "all"), default="all")
    parser.add_argument("--imgsz", type=int)
    args = parser.parse_args()
    try:
        config = load_config(args.run / "source_config.yaml")
        checkpoint = args.checkpoint or args.run / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        export_dir = args.run / "exports"
        export_dir.mkdir(exist_ok=True)
        pytorch_copy = export_dir / checkpoint.name
        if checkpoint.resolve() != pytorch_copy.resolve():
            shutil.copy2(checkpoint, pytorch_copy)
        outputs = {"pytorch_checkpoint": str(pytorch_copy), "onnx": None, "runtime": "not specified; ONNX is a model format, not an AMD acceleration runtime"}
        if args.format in {"onnx", "all"}:
            os.environ["YOLO_AUTOINSTALL"] = "False"
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install Ultralytics: pip install -e '.[benchmark]'") from exc
            onnx_path = YOLO(str(checkpoint)).export(format="onnx", imgsz=args.imgsz or config.imgsz, device="cpu")
            outputs["onnx"] = str(onnx_path)
        write_json(export_dir / "export.json", outputs)
        print(f"Exports saved in {export_dir}: {outputs}")
        return 0
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
