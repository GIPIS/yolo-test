from types import SimpleNamespace
import sys

import pytest

from detector_benchmark.hardware import detect_environment, require_gpu


class FakeTorch:
    __version__ = "test"
    version = SimpleNamespace(hip="6.2")
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_name=lambda index: "AMD Radeon RX 6800",
                           get_device_properties=lambda index: SimpleNamespace(total_memory=16, multi_processor_count=60))


def test_gpu_detection_mocked():
    info = detect_environment(FakeTorch)
    assert info["gpu_detected"] is True
    assert info["devices"][0]["name"] == "AMD Radeon RX 6800"
    assert require_gpu(info) == "0"


def test_gpu_required_unless_explicit_cpu():
    info = detect_environment(SimpleNamespace(__version__="cpu", version=SimpleNamespace(hip=None), cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)))
    with pytest.raises(RuntimeError, match="GPU NOT detected"):
        require_gpu(info)
    assert require_gpu(info, allow_cpu=True) == "cpu"


def test_cuda_only_build_is_not_accepted_for_amd():
    cuda_only = SimpleNamespace(
        __version__="cuda-build",
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                             get_device_name=lambda index: "Other GPU",
                             get_device_properties=lambda index: SimpleNamespace(total_memory=8)),
    )
    info = detect_environment(cuda_only)
    with pytest.raises(RuntimeError, match="does not report HIP"):
        require_gpu(info)
    assert require_gpu(info, allow_cpu=True) == "cpu"


def test_multi_gpu_device_selection_is_validated_and_normalized():
    info = detect_environment(FakeTorch)
    info["device_count"] = 3

    assert require_gpu(info, requested_device=[0, 1, 2]) == "0,1,2"
    assert require_gpu(info, requested_device="0,2") == "0,2"
    with pytest.raises(RuntimeError, match="unavailable"):
        require_gpu(info, requested_device=[0, 3])
    with pytest.raises(RuntimeError, match="duplicate"):
        require_gpu(info, requested_device=[0, 0])


def test_environment_reports_available_system_memory(monkeypatch):
    memory = SimpleNamespace(total=100, available=40, used=60, percent=60.0)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: memory))

    info = detect_environment(SimpleNamespace())

    assert info["memory_total_bytes"] == 100
    assert info["memory_available_bytes"] == 40
    assert info["memory_used_bytes"] == 60
    assert info["memory_percent"] == 60.0


def test_gpu_memory_stats_cover_selected_devices(monkeypatch):
    from detector_benchmark.hardware import memory_stats, reset_peak_memory

    calls = []
    class FakeCuda:
        memory_allocated = staticmethod(lambda index: (index + 1) * 1024 ** 3)
        memory_reserved = staticmethod(lambda index: (index + 2) * 1024 ** 3)
        max_memory_allocated = staticmethod(lambda index: (index + 3) * 1024 ** 3)
        reset_peak_memory_stats = staticmethod(lambda index: calls.append(index))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda))

    assert memory_stats("0,1,2") == {
        "allocated_gib": 6.0,
        "reserved_gib": 9.0,
        "peak_allocated_gib": 12.0,
    }
    reset_peak_memory("0,1,2")
    assert calls == [0, 1, 2]
