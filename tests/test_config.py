from pathlib import Path

import pytest

from detector_benchmark.config import load_config


def test_load_smoke_config():
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    assert config.mode == "smoke"
    assert config.batch == 2
    assert config.allow_cpu is True
    assert config.dataset_yaml.is_absolute()
    assert config.train_options == {}
    assert config.save_period == -1
    assert config.plots is True
    assert config.deterministic is True
    assert config.run_tag == "amd_rx6800"


def test_load_self_contained_training_config_resolves_paths_from_config(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "training.yaml"
    path.write_text(
        "mode: benchmark\nmodel: yolo26n.pt\nsource_images: ../raw/images\nsource_labels: ../raw/labels\n"
        "output: ../prepared\nclasses: [crack, corrosion]\nruns_dir: ../runs\n"
        "imgsz: 640\nbatch: 4\nepochs: 5\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.source_images == (tmp_path / "raw" / "images").resolve()
    assert config.source_labels == (tmp_path / "raw" / "labels").resolve()
    assert config.data_root == (tmp_path / "prepared").resolve()
    assert config.dataset_yaml == (tmp_path / "prepared" / "dataset.yaml").resolve()
    assert config.runs_dir == (tmp_path / "runs").resolve()
    assert config.classes == ["crack", "corrosion"]


def test_load_config_example():
    config = load_config(Path(__file__).parents[1] / "config_example.yaml")

    assert config.source_images == Path("/path/to/MBDD2025/JPEGImages")
    assert config.source_labels == Path("/path/to/MBDD2025/Labels")
    assert config.dataset_yaml == (Path(__file__).parents[1] / "datasets/prepared/dataset.yaml").resolve()


def test_training_options_are_loaded_and_cannot_override_core_settings(tmp_path):
    config_path = Path(__file__).parents[1] / "configs" / "mbdd2025.yaml"
    config = load_config(config_path)
    assert config.device == [0, 1, 2]
    assert config.train_options["mosaic"] == 1.0
    assert config.train_options["hsv_h"] == 0.015

    text = config_path.read_text(encoding="utf-8").replace("  mosaic: 1.0", "  epochs: 99")
    bad_config = tmp_path / "bad-training-options.yaml"
    bad_config.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="cannot override top-level"):
        load_config(bad_config)


def test_reject_invalid_mode(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("mode: invalid\nmodel: yolo26n.yaml\ndataset_yaml: data.yaml\ndata_root: data\nruns_dir: runs\nimgsz: 32\nbatch: 1\nepochs: 1\n")
    with pytest.raises(ValueError, match="mode"):
        load_config(path)


def test_training_controls_are_configurable_and_validated(tmp_path):
    path = tmp_path / "controls.yaml"
    path.write_text(
        "mode: benchmark\nmodel: yolo26n.pt\ndataset_yaml: data.yaml\ndata_root: data\nruns_dir: runs\n"
        "imgsz: 640\nbatch: 4\nepochs: 3\nsave_period: 2\nplots: false\n"
        "deterministic: false\nrun_tag: work-station/a\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.save_period == 2
    assert config.plots is False
    assert config.deterministic is False
    assert config.run_tag == "work-station/a"

    for invalid, message in (("save_period: -2", "save_period"), ("plots: \"yes\"", "plots"), ("deterministic: \"yes\"", "deterministic")):
        bad_path = tmp_path / f"bad-{message}.yaml"
        bad_path.write_text(
            "mode: benchmark\nmodel: yolo26n.pt\ndataset_yaml: data.yaml\ndata_root: data\nruns_dir: runs\n"
            "imgsz: 640\nbatch: 4\nepochs: 3\n" + invalid + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=message):
            load_config(bad_path)


def test_train_options_cannot_duplicate_new_top_level_settings(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "mode: benchmark\nmodel: yolo26n.pt\ndataset_yaml: data.yaml\ndata_root: data\nruns_dir: runs\n"
        "imgsz: 640\nbatch: 4\nepochs: 3\ntrain_options:\n  save_period: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="save_period"):
        load_config(path)
