"""Read-only SQL validation and time-bounded execution for generated code."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlglot import exp, parse

from .database import AnalyticsDatabase


class UnsafeQueryError(ValueError):
    """Raised when generated SQL violates the read-only safety policy."""


class QueryTimeoutError(TimeoutError):
    """Raised when generated SQL exceeds the execution deadline."""


FORBIDDEN_NODE_TYPES = (
    exp.Alter,
    exp.Command,
    exp.Create,
    exp.Delete,
    exp.Drop,
    exp.Insert,
    exp.Merge,
    exp.Transaction,
    exp.Update,
)
FORBIDDEN_FUNCTIONS = {
    "READ_CSV",
    "READ_CSV_AUTO",
    "READ_JSON",
    "READ_JSON_AUTO",
    "READ_PARQUET",
    "HTTPFS",
    "GLOB",
    "ENV",
    "GETENV",
}
SENSITIVE_COLUMNS = {"customer_name", "customer_email"}


@dataclass(frozen=True)
class SafeQueryResult:
    """Validated SQL result with execution metadata."""

    frame: pd.DataFrame
    sql: str
    elapsed_ms: float


def validate_sql(sql: str, row_limit: int = 5_000) -> str:
    """Allow one SELECT/CTE query over `orders` and enforce an output row limit."""

    if not sql or len(sql) > 8_000:
        raise UnsafeQueryError("SQL is empty or exceeds the 8,000-character limit.")
    statements = [statement for statement in parse(sql, read="duckdb") if statement is not None]
    if len(statements) != 1:
        raise UnsafeQueryError("Exactly one SQL statement is allowed.")
    statement = statements[0]
    if not isinstance(statement, (exp.Select, exp.Union)):
        raise UnsafeQueryError("Only SELECT queries and SELECT-based CTEs are allowed.")
    for node_type in FORBIDDEN_NODE_TYPES:
        if statement.find(node_type):
            raise UnsafeQueryError(f"Forbidden SQL operation: {node_type.__name__}")

    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    table_names = {table.name.lower() for table in statement.find_all(exp.Table)}
    disallowed_tables = table_names - {"orders"} - cte_names
    if disallowed_tables:
        raise UnsafeQueryError(f"Only the `orders` table is allowed: {sorted(disallowed_tables)}")
    if "orders" not in table_names:
        raise UnsafeQueryError("The query must read from the `orders` table.")

    for column in statement.find_all(exp.Column):
        if column.name.lower() in SENSITIVE_COLUMNS:
            raise UnsafeQueryError(f"Sensitive column is not queryable: {column.name}")
    for star in statement.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            raise UnsafeQueryError("SELECT * is forbidden; select only required fields.")
    for function in statement.find_all(exp.Func):
        function_name = function.sql_name().upper()
        if function_name in FORBIDDEN_FUNCTIONS:
            raise UnsafeQueryError(f"External or environment function is forbidden: {function_name}")

    normalized = statement.sql(dialect="duckdb", pretty=False).rstrip(";")
    capped_limit = min(max(int(row_limit), 1), 10_000)
    return f"SELECT * FROM ({normalized}) AS generated_result LIMIT {capped_limit}"


class SafeSQLExecutor:
    """Execute validated SQL in a restricted DuckDB connection with a timeout."""

    def __init__(self, timeout_seconds: float = 5.0, row_limit: int = 5_000) -> None:
        self.timeout_seconds = timeout_seconds
        self.row_limit = row_limit

    def execute(self, sql: str) -> SafeQueryResult:
        safe_sql = validate_sql(sql, self.row_limit)
        database = AnalyticsDatabase()
        output: dict[str, Any] = {}

        def run() -> None:
            try:
                output["timing"] = database.query(safe_sql)
            except BaseException as exc:  # captured and re-raised on the caller thread
                output["error"] = exc

        worker = threading.Thread(target=run, name="safe-sql-query", daemon=True)
        worker.start()
        worker.join(self.timeout_seconds)
        if worker.is_alive():
            database.interrupt()
            worker.join(1.0)
            database.close()
            raise QueryTimeoutError(f"Query exceeded {self.timeout_seconds:.1f} seconds.")
        database.close()
        if "error" in output:
            raise output["error"]
        timing = output["timing"]
        return SafeQueryResult(frame=timing.frame, sql=safe_sql, elapsed_ms=timing.elapsed_ms)

