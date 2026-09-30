import sys
from pathlib import Path
from types import SimpleNamespace

from detector_benchmark import benchmarking


def test_result_phase_times_average_every_result():
    results = [
        SimpleNamespace(speed={"preprocess": 1.0, "inference": 4.0, "postprocess": 0.5}),
        SimpleNamespace(speed={"preprocess": 3.0, "inference": 6.0, "postprocess": 1.5}),
    ]
    assert benchmarking.mean_result_phase_times(results) == {
        "preprocess": 2.0,
        "inference": 5.0,
        "postprocess": 1.0,
    }


def test_preload_decodes_unique_paths(monkeypatch, tmp_path):
    calls = []
    fake_cv2 = SimpleNamespace(imread=lambda path: calls.append(path) or f"decoded:{path}")
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    first, second = tmp_path / "one.jpg", tmp_path / "two.jpg"

    loaded = benchmarking.preload_images([first, second, first])

    assert loaded == {first: f"decoded:{first}", second: f"decoded:{second}"}
    assert calls == [str(first), str(second)]


def test_release_gpu_memory_calls_cuda_cleanup(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
    ))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(benchmarking.gc, "collect", lambda: calls.append("gc"))

    benchmarking.release_gpu_memory("0")

    assert calls == ["gc", "synchronize", "empty_cache"]


def test_release_gpu_memory_is_cpu_noop_for_torch(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
    ))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(benchmarking.gc, "collect", lambda: calls.append("gc"))

    benchmarking.release_gpu_memory("cpu")

    assert calls == ["gc"]
