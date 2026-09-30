from pathlib import Path

import pytest

from detector_benchmark.datasets import validate_yolo_dataset
from detector_benchmark.metrics import summarize


def test_dataset_validation_counts(tmp_path):
    for split, count in (("train2017", 2), ("val2017", 1)):
        images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(count):
            (images / f"{index}.jpg").touch()
            (labels / f"{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    assert validate_yolo_dataset(tmp_path) == {"train2017": 2, "val2017": 1}


def test_dataset_missing_labels_fails(tmp_path):
    image_dir = tmp_path / "images" / "train2017"
    label_dir = tmp_path / "labels" / "train2017"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "unlabeled.jpg").touch()
    with pytest.raises(ValueError, match="labels missing"):
        validate_yolo_dataset(tmp_path, ("train2017",))


def test_timing_statistics():
    result = summarize([1, 2, 3, 4, 5])
    assert result["mean"] == 3
    assert result["p50"] == 3
    assert result["p95"] == pytest.approx(4.8)
