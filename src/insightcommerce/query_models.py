"""Strict data models exchanged by LLM providers and the query engine."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ChartType = Literal[
    "auto",
    "table",
    "metric",
    "bar",
    "line",
    "area",
    "scatter",
    "histogram",
    "box",
    "heatmap",
    "choropleth",
    "treemap",
]


class QueryPlan(BaseModel):
    """Validated structured output requested from an LLM."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=8, max_length=8_000)
    title: str = Field(min_length=2, max_length=120)
    chart_type: ChartType = "auto"
    x: str = Field(default="", max_length=80)
    y: str = Field(default="", max_length=80)
    color: str = Field(default="", max_length=80)
    rationale: str = Field(min_length=2, max_length=400)


class QueryAnswer(BaseModel):
    """Grounded result returned to the dashboard after safe execution."""

    question: str
    plan: QueryPlan
    frame: pd.DataFrame
    narrative: str
    elapsed_ms: float
    attempts: int
    provider: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


def query_plan_json_schema() -> dict:
    """Return a provider-neutral JSON Schema with strict additionalProperties."""

    schema = QueryPlan.model_json_schema()
    schema["additionalProperties"] = False
    return schema

