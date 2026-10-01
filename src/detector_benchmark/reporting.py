"""Write concise human-readable run summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_summary(path: str | Path, payload: dict[str, Any]) -> None:
    training = payload.get("training", payload)
    accuracy = training.get("accuracy", payload.get("accuracy", {}))
    memory = training.get("gpu_memory", {})
    rows = [
        "# Training summary", "",
        f"- Status: {payload.get('status', training.get('status', 'completed'))}",
        f"- Hardware: {payload.get('hardware', 'See environment.json')}",
        f"- Software: PyTorch {payload.get('torch_version', 'see environment.json')} / HIP {payload.get('hip_version', 'see environment.json')}",
        f"- Dataset: {payload.get('dataset', training.get('dataset_yaml', 'see config.yaml'))}",
        f"- Model: {training.get('model', payload.get('model', 'unknown'))}",
        f"- Configuration: imgsz={training.get('imgsz', payload.get('imgsz'))}, batch={training.get('batch', payload.get('batch'))}, epochs={training.get('epochs', payload.get('epochs'))}, AMP={training.get('amp', payload.get('amp'))}",
        "", "## Performance",
        f"- Training time: {training.get('total_training_seconds', 'n/a')} s",
        f"- Pure training time: {training.get('pure_train_seconds', 'n/a')} s",
        f"- Validation time: {training.get('validation_seconds', 'n/a')} s",
        f"- Training images/s (pure / estimate): {training.get('approx_images_per_second', 'n/a')}",
        f"- Training images/s (wall / estimate): {training.get('approx_images_per_second_wall', 'n/a')}",
        f"- Peak allocated GPU memory: {memory.get('peak_allocated_gib', 'n/a')} GiB", "",
        "## Accuracy",
        f"- Precision: {accuracy.get('metrics/precision(B)', accuracy.get('precision', 'n/a'))}",
        f"- Recall: {accuracy.get('metrics/recall(B)', accuracy.get('recall', 'n/a'))}",
        f"- mAP50: {accuracy.get('metrics/mAP50(B)', accuracy.get('map50', 'n/a'))}",
        f"- mAP50-95: {accuracy.get('metrics/mAP50-95(B)', accuracy.get('map50_95', 'n/a'))}", "",
        "## Observations / warnings",
        f"- {payload.get('reason', payload.get('warning', 'No automated interpretation; compare only runs with matching data and settings.'))}", "",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows), encoding="utf-8")
