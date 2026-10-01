from detector_benchmark.reporting import write_summary


def test_summary_includes_training_performance_and_accuracy(tmp_path):
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
        "accuracy": {"metrics/mAP50(B)": 0.62, "metrics/mAP50-95(B)": 0.41},
    })
    summary = path.read_text(encoding="utf-8")
    assert "Pure training time: 10.0 s" in summary
    assert "Validation time: 3.0 s" in summary
    assert "mAP50: 0.62" in summary
    assert "mAP50-95: 0.41" in summary
    assert "# Training summary" in summary
    assert "Inference latency" not in summary
