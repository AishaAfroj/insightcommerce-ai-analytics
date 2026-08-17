"""Measure the required filtered aggregation latency and write reproducible evidence."""

from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insightcommerce.config import BENCHMARK_DIR  # noqa: E402
from insightcommerce.database import AnalyticsDatabase  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile for a non-empty timing series."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def run_benchmark(iterations: int = 30) -> dict:
    """Run a warm-cache country-filtered monthly aggregation benchmark."""

    with AnalyticsDatabase() as database:
        database.filtered_monthly_aggregation(
            "2025-01-01", "2025-12-31", ["USA", "India", "Germany"]
        )
        timings = [
            database.filtered_monthly_aggregation(
                "2025-01-01", "2025-12-31", ["USA", "India", "Germany"]
            ).elapsed_ms
            for _ in range(iterations)
        ]
    median_ms = statistics.median(timings)
    result = {
        "benchmark": "filtered_monthly_aggregation",
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "iterations": iterations,
        "filters": {
            "date": ["2025-01-01", "2025-12-31"],
            "countries": ["USA", "India", "Germany"],
        },
        "median_ms": round(median_ms, 3),
        "p95_ms": round(percentile(timings, 0.95), 3),
        "max_ms": round(max(timings), 3),
        "threshold_ms": 500,
        "status": "PASS" if median_ms < 500 else "FAIL",
    }
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_DIR / "performance.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    output = run_benchmark()
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["status"] == "PASS" else 1)

