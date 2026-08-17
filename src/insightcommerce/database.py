"""DuckDB analytics backend with fast filtered aggregations."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import PARQUET_PATH
from .data import prepare_dataset


@dataclass(frozen=True)
class QueryTiming:
    """Result and wall-clock timing for an analytics query."""

    frame: pd.DataFrame
    elapsed_ms: float


def create_connection(parquet_path: Path = PARQUET_PATH) -> duckdb.DuckDBPyConnection:
    """Create an isolated, externally restricted in-memory DuckDB connection."""

    if not parquet_path.exists():
        prepare_dataset(parquet_path=parquet_path)
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '1GB'")
    safe_path = str(parquet_path).replace("'", "''")
    connection.execute(f"CREATE TABLE orders AS SELECT * FROM read_parquet('{safe_path}')")
    # Register the trusted local file first, then prevent generated SQL from
    # opening any additional files, URLs, extensions, or external resources.
    connection.execute("SET enable_external_access = false")
    return connection


class AnalyticsDatabase:
    """Thread-safe query facade for dashboard filters and generated SQL."""

    def __init__(self, parquet_path: Path = PARQUET_PATH) -> None:
        self.parquet_path = parquet_path
        self._lock = threading.RLock()
        self._connection = create_connection(parquet_path)

    def query(self, sql: str, parameters: list[Any] | None = None) -> QueryTiming:
        """Execute one parameterized read-only query and return its timing."""

        started = time.perf_counter()
        with self._lock:
            frame = self._connection.execute(sql, parameters or []).fetchdf()
        return QueryTiming(frame=frame, elapsed_ms=(time.perf_counter() - started) * 1000)

    def filtered_orders(
        self,
        start_date: str,
        end_date: str,
        countries: list[str] | None = None,
        categories: list[str] | None = None,
        limit: int = 5_000,
    ) -> QueryTiming:
        """Return filtered records without interpolating user values into SQL."""

        where = ["date BETWEEN ? AND ?"]
        params: list[Any] = [start_date, end_date]
        if countries:
            where.append(f"country IN ({','.join('?' for _ in countries)})")
            params.extend(countries)
        if categories:
            where.append(f"product_category IN ({','.join('?' for _ in categories)})")
            params.extend(categories)
        params.append(int(min(max(limit, 1), 10_000)))
        sql = f"SELECT * FROM orders WHERE {' AND '.join(where)} ORDER BY date LIMIT ?"
        return self.query(sql, params)

    def filtered_monthly_aggregation(
        self,
        start_date: str,
        end_date: str,
        countries: list[str] | None = None,
    ) -> QueryTiming:
        """Benchmark target: monthly revenue and orders under optional country filters."""

        where = ["date BETWEEN ? AND ?"]
        params: list[Any] = [start_date, end_date]
        if countries:
            where.append(f"country IN ({','.join('?' for _ in countries)})")
            params.extend(countries)
        sql = f"""
            SELECT date_trunc('month', date) AS month,
                   SUM(order_value) AS revenue,
                   COUNT(*) AS orders,
                   AVG(order_value) AS average_order_value
            FROM orders
            WHERE {' AND '.join(where)}
            GROUP BY 1
            ORDER BY 1
        """
        return self.query(sql, params)

    def close(self) -> None:
        """Release the in-memory DuckDB connection."""

        with self._lock:
            self._connection.close()

    def __enter__(self) -> AnalyticsDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
