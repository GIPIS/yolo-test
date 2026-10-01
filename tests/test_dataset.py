from pathlib import Path

import pytest

from detector_benchmark.datasets import prepare_yolo_dataset, validate_yolo_dataset


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


def test_prepare_configured_dataset_creates_deterministic_yolo_splits(tmp_path):
    source = tmp_path / "source"
    (source / "images_input").mkdir(parents=True)
    (source / "annotations_input").mkdir()
    for index in range(10):
        (source / "images_input" / f"image_{index}.jpg").write_bytes(b"fake image")
        (source / "annotations_input" / f"image_{index}.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    output = tmp_path / "prepared"
    config = tmp_path / "dataset-config.yaml"
    config.write_text(
        "dataset_name: arbitrary-dataset\n"
        "source_images: source/images_input\n"
        "source_labels: source/annotations_input\n"
        "output: prepared\n"
        "classes:\n  - background-object\n  - target-object\n"
        "validation_fraction: 0.2\nseed: 11\nlink_files: true\n",
        encoding="utf-8",
    )

    manifest = prepare_yolo_dataset(config)
    assert manifest["counts"] == {"val2017": 2, "train2017": 8}
    assert validate_yolo_dataset(output) == {"train2017": 8, "val2017": 2}
    yaml_text = (output / "dataset.yaml").read_text()
    assert "dataset_name: arbitrary-dataset" in yaml_text
    assert "1: target-object" in yaml_text
    assert all(path.is_symlink() for path in (output / "images" / "train2017").iterdir())

    repeated = prepare_yolo_dataset(config)
    assert repeated["already_prepared"] is True
    config.write_text(config.read_text(encoding="utf-8").replace("validation_fraction: 0.2", "validation_fraction: 0.3"), encoding="utf-8")
    with pytest.raises(FileExistsError, match="does not match this config"):
        prepare_yolo_dataset(config)


def test_prepare_configured_dataset_rejects_undeclared_class(tmp_path):
    source = tmp_path / "source"
    images, labels = source / "images", source / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for index in range(2):
        (images / f"{index}.jpg").touch()
        (labels / f"{index}.txt").write_text("2 0.5 0.5 0.2 0.2\n")
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "source_images: source/images\nsource_labels: source/labels\noutput: out\nclasses: [only-class]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not declared in config classes"):
        prepare_yolo_dataset(config)
