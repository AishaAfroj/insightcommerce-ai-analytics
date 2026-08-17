"""Data-grounded result narration that never invents fields or values."""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd


def format_value(value: object) -> str:
    """Format common numeric and timestamp values for a concise explanation."""

    if isinstance(value, (float, int)):
        number = float(value)
        if math.isfinite(number) and abs(number) >= 1_000:
            return f"{number:,.2f}"
        return f"{number:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def narrate_result(frame: pd.DataFrame, title: str) -> str:
    """Summarize only values present in the executed result table."""

    if frame.empty:
        return f"{title}: no records matched the question and current filters."
    if frame.shape == (1, 1):
        column = str(frame.columns[0]).replace("_", " ")
        return f"{title}: {column} is {format_value(frame.iloc[0, 0])}."
    numeric = frame.select_dtypes(include="number").columns.tolist()
    labels = [column for column in frame.columns if column not in numeric]
    if numeric and labels:
        metric = numeric[0]
        ordered = frame.sort_values(metric, ascending=False)
        leader = ordered.iloc[0]
        return (
            f"{title}: {format_value(leader[labels[0]])} has the highest "
            f"{metric.replace('_', ' ')} "
            f"in this result at {format_value(leader[metric])}. The table contains "
            f"{len(frame):,} grouped rows."
        )
    return f"{title}: the executed query returned {len(frame):,} rows and {len(frame.columns)} columns."
