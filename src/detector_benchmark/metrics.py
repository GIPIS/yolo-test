"""Small statistics helpers for repeatable timing summaries."""
from __future__ import annotations

import statistics
from typing import Iterable


def summarize(values: Iterable[float]) -> dict[str, float | None]:
    items = sorted(float(value) for value in values)
    if not items:
        return {key: None for key in ("mean", "median", "p50", "p95", "min", "max")}
    def percentile(fraction: float) -> float:
        index = (len(items) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(items) - 1)
        return items[lower] + (items[upper] - items[lower]) * (index - lower)
    return {"mean": statistics.fmean(items), "median": statistics.median(items), "p50": percentile(0.50),
            "p95": percentile(0.95), "min": items[0], "max": items[-1]}
