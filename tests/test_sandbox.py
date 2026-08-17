"""Security tests for generated SQL validation."""

import pytest

from insightcommerce.sandbox import UnsafeQueryError, validate_sql


def test_safe_select_is_wrapped_with_limit() -> None:
    sql = validate_sql(
        "SELECT country, SUM(order_value) AS revenue FROM orders GROUP BY country"
    )
    assert "generated_result" in sql
    assert "LIMIT 5000" in sql


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "SELECT * FROM orders",
        "SELECT customer_email FROM orders",
        "SELECT * FROM read_csv_auto('/tmp/secret.csv')",
        "COPY orders TO '/tmp/orders.csv'",
        "SELECT country FROM information_schema.tables",
        "SELECT country FROM orders; DELETE FROM orders",
    ],
)
def test_unsafe_sql_is_rejected(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql)


def test_count_star_is_allowed() -> None:
    assert "COUNT(*)" in validate_sql("SELECT COUNT(*) AS orders FROM orders")

