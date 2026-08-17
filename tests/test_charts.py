"""Tests for chart recommendation and manual compatibility checks."""

import pandas as pd

from insightcommerce.charts import ChartSpec, chart_is_compatible, recommend_chart


def test_time_series_recommends_line() -> None:
    frame = pd.DataFrame(
        {"month": pd.to_datetime(["2025-01-01", "2025-02-01"]), "revenue": [1.0, 2.0]}
    )
    assert recommend_chart(frame).chart_type == "line"


def test_country_result_recommends_choropleth() -> None:
    frame = pd.DataFrame({"country": ["USA", "India"], "revenue": [10.0, 20.0]})
    spec = recommend_chart(frame)
    assert spec.chart_type == "choropleth"
    assert chart_is_compatible(frame, ChartSpec("bar", "country", "revenue"))


def test_invalid_ai_hint_falls_back() -> None:
    frame = pd.DataFrame({"country": ["USA"], "revenue": [10.0]})
    assert recommend_chart(frame, "line", "missing", "revenue").chart_type != "line"

