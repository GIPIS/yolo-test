from detector_benchmark.reporting import write_summary


def test_summary_includes_training_and_inference_timing_definitions(tmp_path):
    path = tmp_path / "summary.md"
    write_summary(path, {
        "status": "completed",
        "training": {
            "status": "completed",
            "model": "yolo26n.pt",
            "pure_train_seconds": 10.0,
            "validation_seconds": 3.0,
            "epoch_times_seconds": [4.0, 6.0],
            "time_per_epoch_seconds": 5.0,
            "approx_images_per_second": 20.0,
            "approx_images_per_second_wall": 15.0,
        },
        "inference": {
            "latency_ms": {"total": {"p50": 12.0}},
            "timing_semantics": {"total": "end-to-end includes decode", "phases": "per-image model phases"},
        },
    })
    summary = path.read_text(encoding="utf-8")
    assert "Pure training time: 10.0 s" in summary
    assert "Validation time: 3.0 s" in summary
    assert "Inference total timing: end-to-end includes decode" in summary
    assert "Inference phase timing: per-image model phases" in summary
