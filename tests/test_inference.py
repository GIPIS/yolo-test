from pathlib import Path

from detector_benchmark.inference import UltralyticsDetector


def test_ultralytics_detector_forwards_batch_size():
    class FakeModel:
        def __init__(self):
            self.call = None

        def predict(self, **kwargs):
            self.call = kwargs
            return []

    model = FakeModel()
    detector = UltralyticsDetector("weights.pt", device="0")
    detector.model = model
    images = [Path("a.jpg"), Path("b.jpg")]

    detector.batch_predict(images, batch=8, imgsz=640)

    assert model.call["source"] == images
    assert model.call["batch"] == 8
    assert model.call["imgsz"] == 640
    assert model.call["device"] == "0"
