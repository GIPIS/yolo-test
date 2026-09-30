from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from detector_benchmark.config import load_config
from detector_benchmark.training import train


def test_training_reports_train_validation_and_wall_timings(monkeypatch, tmp_path):
    base = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    train_images = tmp_path / "images" / "train2017"
    train_images.mkdir(parents=True)
    for index in range(2):
        (train_images / f"{index}.jpg").touch()
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "images/train2017", "val": "images/train2017", "names": ["item"]}))
    config = replace(base, dataset_yaml=dataset_yaml, data_root=tmp_path, epochs=2,
                     deterministic=False, plots=False, save_period=3)
    ticks = iter([0, 1, 4, 5, 7, 8, 10, 11, 14, 15])
    monkeypatch.setattr("detector_benchmark.training.time.perf_counter", lambda: next(ticks))
    captured = {}

    class FakeYOLO:
        def __init__(self, model):
            self.model = SimpleNamespace(parameters=lambda: [])
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks.setdefault(event, []).append(callback)

        def fire(self, event):
            for callback in self.callbacks.get(event, []):
                callback(SimpleNamespace())

        def train(self, **kwargs):
            captured.update(kwargs)
            for _ in range(2):
                self.fire("on_train_epoch_start")
                self.fire("on_train_epoch_end")
                self.fire("on_val_start")
                self.fire("on_val_end")
            return SimpleNamespace(save_dir=tmp_path / "run", results_dict={"precision": 0.5})

    monkeypatch.setitem(__import__("sys").modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    result = train(config, tmp_path / "run", "cpu")

    assert result["total_training_seconds"] == 15
    assert result["epoch_times_seconds"] == [3, 2]
    assert result["pure_train_seconds"] == 5
    assert result["validation_seconds"] == 5
    assert result["time_per_epoch_seconds"] == 2.5
    assert result["approx_images_per_second"] == 0.8
    assert result["approx_images_per_second_wall"] == 4 / 15
    assert captured["deterministic"] is False
    assert captured["plots"] is False
    assert captured["save_period"] == 3
