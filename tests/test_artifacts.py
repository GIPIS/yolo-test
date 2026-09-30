import json
from datetime import datetime, timezone

from detector_benchmark.artifacts import json_safe, run_name, write_json


def test_run_name_is_safe_and_deterministic():
    now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    assert run_name("yolo26n.pt", now) == "2026-09-30_120000_amd_rx6800_yolo26n"
    assert run_name("yolo26n.pt", now, tag="Work station/A") == "2026-09-30_120000_work-station-a_yolo26n"


def test_json_serialization(tmp_path):
    path = write_json(tmp_path / "result.json", {"path": tmp_path, "nan": float("nan")})
    parsed = json.loads(path.read_text())
    assert parsed["path"] == str(tmp_path)
    assert parsed["nan"] is None
    assert json_safe((1, 2)) == [1, 2]
