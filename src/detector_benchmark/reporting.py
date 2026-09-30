"""Write concise human-readable run summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_summary(path: str | Path, payload: dict[str, Any]) -> None:
        cases = payload.get("results", [])
        training = payload.get("training", payload)
        if cases and not training.get("total_training_seconds"):
                training = cases[0]
        accuracy = training.get("accuracy", payload.get("accuracy", {}))
        inference = payload.get("inference", {})
        if not inference and cases and "latency_ms" in cases[0]:
                inference = cases[0]
        memory = training.get("gpu_memory", inference.get("gpu_memory", {}))
        total_latency = inference.get("latency_ms", {}).get("total", {})
        rows = [
                "# Benchmark summary", "",
                f"- Status: {payload.get('status', training.get('status', 'completed'))}",
                f"- Hardware: {payload.get('hardware', 'See environment.json')}",
                f"- Software: PyTorch {payload.get('torch_version', 'see environment.json')} / HIP {payload.get('hip_version', 'see environment.json')}",
                f"- Dataset: {payload.get('dataset', training.get('dataset_yaml', 'see config.yaml'))}",
                f"- Model: {training.get('model', payload.get('model', 'unknown'))}",
                f"- Configuration: imgsz={training.get('imgsz', payload.get('imgsz'))}, batch={training.get('batch', payload.get('batch'))}, epochs={training.get('epochs', payload.get('epochs'))}, AMP={training.get('amp', payload.get('amp'))}",
                "", "## Performance",
                f"- Training time: {training.get('total_training_seconds', 'n/a')} s",
                f"- Training images/s (estimate): {training.get('approx_images_per_second', 'n/a')}",
                f"- Inference latency (p50): {total_latency.get('p50', 'n/a')} ms/image",
                f"- Inference throughput: {inference.get('images_per_second', 'n/a')} images/s",
                f"- Peak allocated GPU memory: {memory.get('peak_allocated_gib', 'n/a')} GiB", "",
                "## Accuracy",
                f"- Precision: {accuracy.get('metrics/precision(B)', accuracy.get('precision', 'n/a'))}",
                f"- Recall: {accuracy.get('metrics/recall(B)', accuracy.get('recall', 'n/a'))}",
                f"- mAP50: {accuracy.get('metrics/mAP50(B)', accuracy.get('map50', 'n/a'))}",
                f"- mAP50-95: {accuracy.get('metrics/mAP50-95(B)', accuracy.get('map50_95', 'n/a'))}", "",
                "## Observations / warnings",
                f"- {payload.get('reason', payload.get('warning', 'No automated interpretation; compare only runs with matching data and settings.'))}", "",
        ]
        if len(cases) > 1:
                rows.extend(["## Matrix cases", ""])
                for case in cases:
                        if "latency_ms" in case:
                                p50 = case.get("latency_ms", {}).get("total", {}).get("p50", "n/a")
                                rows.append(f"- {case.get('status')}: {case.get('model')} / imgsz={case.get('imgsz')} / batch={case.get('batch')} / p50={p50} ms / throughput={case.get('images_per_second')} images/s")
                        else:
                                rows.append(f"- {case.get('status')}: {case.get('model')} / imgsz={case.get('imgsz')} / batch={case.get('batch')} / training={case.get('total_training_seconds', 'n/a')} s / accuracy={case.get('accuracy', {})}")
                rows.append("")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(rows), encoding="utf-8")
