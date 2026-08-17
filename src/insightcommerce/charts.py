"""AI-guided chart recommendation, compatibility checks, and Plotly builders."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .query_models import ChartType

CHART_OPTIONS: tuple[ChartType, ...] = (
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
)


@dataclass(frozen=True)
class ChartSpec:
    """Resolved visualization fields after AI recommendation or manual override."""

    chart_type: ChartType
    x: str = ""
    y: str = ""
    color: str = ""


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include="number").columns.tolist()


def _datetime_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    numeric = set(_numeric_columns(frame))
    return [column for column in frame.columns if column not in numeric]


def chart_is_compatible(frame: pd.DataFrame, spec: ChartSpec) -> bool:
    """Reject chart hints that reference missing or incompatible result columns."""

    columns = set(frame.columns)
    if any(value and value not in columns for value in (spec.x, spec.y, spec.color)):
        return False
    numeric = set(_numeric_columns(frame))
    if spec.chart_type in {"metric"}:
        return frame.shape == (1, 1) or bool(numeric)
    if spec.chart_type in {"line", "area", "bar", "scatter", "treemap", "choropleth"}:
        return bool(spec.x and spec.y and spec.y in numeric)
    if spec.chart_type in {"histogram", "box"}:
        return bool((spec.x and spec.x in numeric) or (spec.y and spec.y in numeric))
    if spec.chart_type == "heatmap":
        return bool(spec.x and spec.y and spec.color and spec.color in numeric)
    return True


def recommend_chart(
    frame: pd.DataFrame,
    ai_hint: ChartType = "auto",
    x: str = "",
    y: str = "",
    color: str = "",
) -> ChartSpec:
    """Use a compatible AI hint, otherwise infer a chart from result shape and types."""

    hinted = ChartSpec(ai_hint, x, y, color)
    if ai_hint != "auto" and chart_is_compatible(frame, hinted):
        return hinted

    numeric = _numeric_columns(frame)
    datetimes = _datetime_columns(frame)
    categories = _categorical_columns(frame)
    if frame.empty:
        return ChartSpec("table")
    if frame.shape == (1, 1) and numeric:
        return ChartSpec("metric", y=numeric[0])
    if datetimes and numeric:
        return ChartSpec("line", x=datetimes[0], y=numeric[0])
    country = next((column for column in frame.columns if column.lower() == "country"), "")
    if country and numeric and 2 <= len(frame) <= 250:
        return ChartSpec("choropleth", x=country, y=numeric[0])
    if categories and numeric and len(frame) <= 50:
        return ChartSpec("bar", x=categories[0], y=numeric[0])
    if len(numeric) >= 2:
        return ChartSpec("scatter", x=numeric[0], y=numeric[1])
    if numeric:
        return ChartSpec("histogram", x=numeric[0])
    return ChartSpec("table")


def build_chart(frame: pd.DataFrame, spec: ChartSpec, title: str) -> go.Figure:
    """Build one interactive Plotly figure from a resolved compatible specification."""

    if spec.chart_type == "auto":
        spec = recommend_chart(frame)
    if not chart_is_compatible(frame, spec):
        spec = recommend_chart(frame)
    template = "plotly_white"
    labels = {column: column.replace("_", " ").title() for column in frame.columns}

    if spec.chart_type == "bar":
        figure = px.bar(frame, x=spec.x, y=spec.y, color=spec.color or None, labels=labels)
    elif spec.chart_type == "line":
        figure = px.line(
            frame, x=spec.x, y=spec.y, color=spec.color or None, markers=True, labels=labels
        )
    elif spec.chart_type == "area":
        figure = px.area(frame, x=spec.x, y=spec.y, color=spec.color or None, labels=labels)
    elif spec.chart_type == "scatter":
        figure = px.scatter(
            frame,
            x=spec.x,
            y=spec.y,
            color=spec.color or None,
            opacity=0.68,
            labels=labels,
        )
    elif spec.chart_type == "histogram":
        value = spec.x or spec.y
        figure = px.histogram(frame, x=value, color=spec.color or None, labels=labels, nbins=40)
    elif spec.chart_type == "box":
        value = spec.y or spec.x
        figure = px.box(
            frame,
            x=spec.x if spec.y else None,
            y=value if spec.y else None,
            color=spec.color or None,
            points="outliers",
            labels=labels,
        )
    elif spec.chart_type == "choropleth":
        figure = px.choropleth(
            frame,
            locations=spec.x,
            locationmode="country names",
            color=spec.y,
            color_continuous_scale="Blues",
            labels=labels,
        )
    elif spec.chart_type == "treemap":
        path = [column for column in (spec.color, spec.x) if column]
        figure = px.treemap(frame, path=path or [spec.x], values=spec.y, labels=labels)
    elif spec.chart_type == "heatmap":
        pivot = frame.pivot_table(
            index=spec.y, columns=spec.x, values=spec.color, aggfunc="sum", fill_value=0
        )
        figure = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": labels.get(spec.x, spec.x), "y": labels.get(spec.y, spec.y)},
        )
    else:
        display = frame.head(100).copy()
        figure = go.Figure(
            data=[
                go.Table(
                    header={"values": list(display.columns), "fill_color": "#DBEAFE"},
                    cells={"values": [display[column] for column in display.columns]},
                )
            ]
        )

    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template=template,
        margin={"l": 30, "r": 25, "t": 70, "b": 35},
        colorway=["#2563EB", "#0EA5E9", "#14B8A6", "#F59E0B", "#8B5CF6"],
        hoverlabel={"bgcolor": "white"},
    )
    return figure


def monthly_revenue_chart(frame: pd.DataFrame) -> go.Figure:
    """Line chart for the full filtered monthly revenue series."""

    monthly = (
        frame.assign(month=frame["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(revenue=("order_value", "sum"), orders=("order_id", "count"))
    )
    return build_chart(monthly, ChartSpec("line", "month", "revenue"), "Monthly revenue")


def country_revenue_chart(frame: pd.DataFrame) -> go.Figure:
    """World choropleth for revenue in the selected countries."""

    grouped = frame.groupby("country", as_index=False).agg(revenue=("order_value", "sum"))
    return build_chart(
        grouped, ChartSpec("choropleth", "country", "revenue"), "Revenue by country"
    )


def category_revenue_chart(frame: pd.DataFrame) -> go.Figure:
    """Ranked bar chart for the documented derived taxonomy."""

    grouped = (
        frame.groupby("product_category", observed=True, as_index=False)
        .agg(revenue=("order_value", "sum"))
        .sort_values("revenue")
    )
    figure = px.bar(grouped, x="revenue", y="product_category", orientation="h")
    figure.update_layout(template="plotly_white", title="Revenue by product category")
    return figure


def quarter_country_heatmap(frame: pd.DataFrame) -> go.Figure:
    """Heatmap showing country-by-quarter revenue concentration."""

    grouped = frame.groupby(["quarter", "country"], as_index=False).agg(
        revenue=("order_value", "sum")
    )
    return build_chart(
        grouped,
        ChartSpec("heatmap", x="quarter", y="country", color="revenue"),
        "Country × quarter revenue heatmap",
    )


def product_treemap(frame: pd.DataFrame) -> go.Figure:
    """Hierarchical category-to-product revenue treemap."""

    grouped = frame.groupby(["product_category", "product"], observed=True, as_index=False).agg(
        revenue=("order_value", "sum")
    )
    figure = px.treemap(
        grouped,
        path=[px.Constant("All electronics"), "product_category", "product"],
        values="revenue",
        color="revenue",
        color_continuous_scale="Blues",
    )
    figure.update_layout(template="plotly_white", title="Category and product revenue hierarchy")
    return figure


def price_order_scatter(frame: pd.DataFrame, sample_size: int = 4_000) -> go.Figure:
    """Sampled scatter plot that remains responsive for 100k+ transactions."""

    sample = frame.sample(min(sample_size, len(frame)), random_state=42)
    figure = px.scatter(
        sample,
        x="price",
        y="order_value",
        color="product_category",
        size="quantity",
        hover_data=["country", "product"],
        opacity=0.6,
    )
    figure.update_layout(template="plotly_white", title="Price, quantity, and order value")
    return figure


def category_price_box(frame: pd.DataFrame) -> go.Figure:
    """Box plot for unit-price dispersion by product category."""

    sample = frame.sample(min(12_000, len(frame)), random_state=42)
    figure = px.box(sample, x="product_category", y="price", points="outliers")
    figure.update_xaxes(tickangle=-30)
    figure.update_layout(template="plotly_white", title="Unit-price distribution by category")
    return figure


def order_value_histogram(frame: pd.DataFrame) -> go.Figure:
    """Histogram for the transaction-value distribution."""

    figure = px.histogram(frame, x="order_value", nbins=60, marginal="box")
    figure.update_layout(template="plotly_white", title="Order-value distribution")
    return figure

