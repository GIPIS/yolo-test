import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_predict_script():
    script_path = Path(__file__).parents[1] / "scripts" / "predict.py"
    spec = importlib.util.spec_from_file_location("predict_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBoxes(SimpleNamespace):
    def __len__(self):
        return len(self.xyxy.values)


class FakeResult:
    names = {0: "crack"}
    boxes = FakeBoxes(
        xyxy=FakeTensor([[10, 20, 100, 120]]),
        cls=FakeTensor([0]),
        conf=FakeTensor([0.91]),
    )

    def __init__(self, path):
        self.path = path

    def save(self, filename):
        Path(filename).write_bytes(b"annotated image")


def test_predict_saves_annotated_image_and_prints_pixel_boxes(monkeypatch, tmp_path, capsys):
    predict_script = _load_predict_script()
    source = tmp_path / "input.jpg"
    source.write_bytes(b"image")
    model = tmp_path / "model.pt"
    model.write_bytes(b"checkpoint")
    output = tmp_path / "detected.jpg"
    captured = {}

    class FakeDetector:
        def __init__(self, checkpoint, device):
            captured["checkpoint"] = checkpoint
            captured["device"] = device

        def predict(self, images, **kwargs):
            captured["images"] = images
            captured["options"] = kwargs
            return [FakeResult(images[0])]

    monkeypatch.setattr(predict_script, "UltralyticsDetector", FakeDetector)
    monkeypatch.setattr(
        "sys.argv",
        ["predict.py", "--model", str(model), "--source", str(source), "--output", str(output), "--classes", "0"],
    )

    assert predict_script.main() == 0
    assert output.read_bytes() == b"annotated image"
    assert captured["options"] == {
        "batch": 1,
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "classes": [0],
    }
    report = capsys.readouterr().out
    assert "crack confidence=0.910" in report
    assert "xyxy=(10.0, 20.0, 100.0, 120.0)" in report


def test_predict_folder_preserves_relative_paths(monkeypatch, tmp_path):
    predict_script = _load_predict_script()
    source = tmp_path / "images"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "input.jpg").write_bytes(b"image")
    model = tmp_path / "model.pt"
    model.write_bytes(b"checkpoint")
    output = tmp_path / "output"

    class FakeDetector:
        def __init__(self, checkpoint, device):
            pass

        def predict(self, images, **kwargs):
            return [FakeResult(images[0])]

    monkeypatch.setattr(predict_script, "UltralyticsDetector", FakeDetector)
    monkeypatch.setattr(
        "sys.argv",
        ["predict.py", "--model", str(model), "--source", str(source), "--output", str(output)],
    )

    assert predict_script.main() == 0
    assert (output / "nested" / "input.jpg").read_bytes() == b"annotated image"