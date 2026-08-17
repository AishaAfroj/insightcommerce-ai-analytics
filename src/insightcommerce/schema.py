"""Schema contracts and metadata used by ingestion, prompts, and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

RAW_COLUMNS = (
    "order_id",
    "date",
    "customer_id",
    "customer_name",
    "customer_email",
    "country",
    "product",
    "price",
    "quantity",
    "order_value",
)

DERIVED_COLUMNS = (
    "product_category",
    "year",
    "quarter",
    "month",
    "month_name",
    "week",
    "day_of_week",
    "is_q4",
)


@dataclass(frozen=True)
class ColumnSpec:
    """Human-readable description of one analytics column."""

    name: str
    dtype: str
    description: str
    pii: bool = False
    derived: bool = False


COLUMN_SPECS = (
    ColumnSpec("order_id", "string", "Unique transaction identifier."),
    ColumnSpec("date", "date", "Transaction date in 2025."),
    ColumnSpec("customer_id", "string", "Synthetic customer identifier."),
    ColumnSpec("customer_name", "string", "Synthetic customer name; excluded from AI samples.", True),
    ColumnSpec("customer_email", "string", "Synthetic email; excluded from AI prompts and exports.", True),
    ColumnSpec("country", "string", "Transaction country across 15 markets."),
    ColumnSpec("product", "string", "Electronic product name from a catalog of 105 products."),
    ColumnSpec("price", "float", "Unit price in USD."),
    ColumnSpec("quantity", "integer", "Units purchased in the transaction."),
    ColumnSpec("order_value", "float", "Revenue in USD; exactly price multiplied by quantity."),
    ColumnSpec("product_category", "string", "Derived high-level electronics taxonomy.", derived=True),
    ColumnSpec("year", "integer", "Calendar year derived from date.", derived=True),
    ColumnSpec("quarter", "integer", "Calendar quarter derived from date.", derived=True),
    ColumnSpec("month", "integer", "Calendar month number derived from date.", derived=True),
    ColumnSpec("month_name", "string", "Abbreviated calendar month.", derived=True),
    ColumnSpec("week", "integer", "ISO week number derived from date.", derived=True),
    ColumnSpec("day_of_week", "string", "Calendar weekday derived from date.", derived=True),
    ColumnSpec("is_q4", "boolean", "True for October through December.", derived=True),
)


def schema_catalog() -> list[dict[str, Any]]:
    """Return prompt-ready, JSON-serializable metadata for all queryable fields."""

    return [asdict(spec) for spec in COLUMN_SPECS]


def public_query_columns() -> list[str]:
    """Return fields safe for general analytics output."""

    return [spec.name for spec in COLUMN_SPECS if not spec.pii]


def schema_prompt_text() -> str:
    """Render a compact schema description for the LLM system prompt."""

    lines = ["Table: orders"]
    for spec in COLUMN_SPECS:
        if spec.pii:
            continue
        origin = "derived" if spec.derived else "source"
        lines.append(f"- {spec.name} ({spec.dtype}, {origin}): {spec.description}")
    return "\n".join(lines)

