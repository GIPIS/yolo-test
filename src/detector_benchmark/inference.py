"""Framework-neutral detection interface and Ultralytics adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence


class Detector(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def predict(self, image: str | Path, **kwargs: Any) -> Any: ...

    def batch_predict(self, images: Sequence[str | Path], **kwargs: Any) -> Any:
        return self.predict(list(images), **kwargs)


class UltralyticsDetector(Detector):
    def __init__(self, checkpoint: str | Path, device: str = "0") -> None:
        self.checkpoint = str(checkpoint)
        self.device = device
        self.model: Any = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the 'benchmark' extra before using YOLO inference: pip install -e '.[benchmark]'") from exc
        self.model = YOLO(self.checkpoint)

    def predict(self, image: str | Path | Sequence[str | Path], **kwargs: Any) -> Any:
        if self.model is None:
            self.load()
        return self.model.predict(source=image, device=self.device, verbose=False, **kwargs)
