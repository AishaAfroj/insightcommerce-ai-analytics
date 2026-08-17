"""Preset insight questions for reliable one-click demonstrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PresetInsight:
    """Display metadata for one suggested natural-language question."""

    label: str
    question: str
    purpose: str


PRESET_INSIGHTS = (
    PresetInsight(
        "Monthly revenue",
        "Show the monthly revenue trend for 2025.",
        "Time-series aggregation and seasonality.",
    ),
    PresetInsight(
        "Country leaders",
        "Which countries generated the most revenue?",
        "Geographic market comparison.",
    ),
    PresetInsight(
        "Product leaders",
        "Show the top 10 products by revenue.",
        "Product performance ranking.",
    ),
    PresetInsight(
        "Category mix",
        "Compare revenue across product categories.",
        "Derived taxonomy comparison.",
    ),
    PresetInsight(
        "Quarterly seasonality",
        "Compare every quarter and highlight Q4 performance.",
        "Modeled Q4 seasonality.",
    ),
    PresetInsight(
        "Country AOV",
        "Which country has the highest average order value?",
        "Market-normalized customer value.",
    ),
)
