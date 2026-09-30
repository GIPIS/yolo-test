from pathlib import Path

import pytest

from detector_benchmark.config import load_config


def test_load_smoke_config():
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    assert config.mode == "smoke"
    assert config.batch == 2
    assert config.allow_cpu is True
    assert config.dataset_yaml.is_absolute()


def test_reject_invalid_mode(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("mode: invalid\nmodel: yolo26n.yaml\ndataset_yaml: data.yaml\ndata_root: data\nruns_dir: runs\nimgsz: 32\nbatch: 1\nepochs: 1\n")
    with pytest.raises(ValueError, match="mode"):
        load_config(path)
