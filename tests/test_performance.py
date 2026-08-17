"""Performance acceptance test for Task A filtered aggregation."""

import statistics

from insightcommerce.database import AnalyticsDatabase


def test_filtered_aggregation_median_under_500ms() -> None:
    with AnalyticsDatabase() as database:
        database.filtered_monthly_aggregation(
            "2025-01-01", "2025-12-31", ["USA", "India", "Germany"]
        )
        timings = [
            database.filtered_monthly_aggregation(
                "2025-01-01", "2025-12-31", ["USA", "India", "Germany"]
            ).elapsed_ms
            for _ in range(10)
        ]
    assert statistics.median(timings) < 500
