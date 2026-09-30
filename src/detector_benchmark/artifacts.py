"""Local experiment naming, metadata and JSON serialization."""
from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_name(model: str, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", Path(model).stem).strip("-").lower() or "model"
    return f"{timestamp}_amd_rx6800_{slug}"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
