"""Run and record a ten-question accuracy benchmark for the safe AI query pipeline."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insightcommerce.assistant import AnalyticsAssistant  # noqa: E402
from insightcommerce.charts import ChartSpec, chart_is_compatible, recommend_chart  # noqa: E402
from insightcommerce.config import BENCHMARK_DIR  # noqa: E402
from insightcommerce.database import AnalyticsDatabase  # noqa: E402
from insightcommerce.providers import RuleBasedProvider  # noqa: E402


@dataclass(frozen=True)
class BenchmarkCase:
    """One natural-language question and its independently executed reference SQL."""

    question: str
    reference_sql: str


CASES = (
    BenchmarkCase("What is the total revenue?", "SELECT SUM(order_value) AS total_revenue FROM orders"),
    BenchmarkCase("How many orders are there?", "SELECT COUNT(*) AS orders FROM orders"),
    BenchmarkCase("How many units were sold?", "SELECT SUM(quantity) AS units FROM orders"),
    BenchmarkCase(
        "Show the monthly revenue trend for 2025.",
        "SELECT date_trunc('month', date) AS month, SUM(order_value) AS revenue FROM orders GROUP BY 1 ORDER BY 1",
    ),
    BenchmarkCase(
        "Which countries generated the most revenue?",
        "SELECT country, SUM(order_value) AS revenue, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY revenue DESC",
    ),
    BenchmarkCase(
        "Which country has the highest average order value?",
        "SELECT country, AVG(order_value) AS average_order_value FROM orders GROUP BY 1 ORDER BY average_order_value DESC",
    ),
    BenchmarkCase(
        "Show the top 10 products by revenue.",
        "SELECT product, SUM(order_value) AS revenue, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY revenue DESC LIMIT 10",
    ),
    BenchmarkCase(
        "Compare revenue across product categories.",
        "SELECT product_category, SUM(order_value) AS revenue, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY revenue DESC",
    ),
    BenchmarkCase(
        "Compare every quarter and highlight Q4 performance.",
        "SELECT quarter, SUM(order_value) AS revenue, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY 1",
    ),
    BenchmarkCase(
        "What are the leading products by revenue?",
        "SELECT product, SUM(order_value) AS revenue, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY revenue DESC LIMIT 10",
    ),
)


def normalized(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column order and numeric precision for exact reference comparison."""

    result = frame.reset_index(drop=True).copy()
    for column in result.select_dtypes(include="number"):
        result[column] = result[column].astype(float).round(6)
    return result


def pass_mark(value: bool) -> str:
    """Render a compact benchmark status label."""

    return "PASS" if value else "FAIL"


def run_benchmark() -> dict:
    """Evaluate exact result, structure, chart compatibility, retry count, and safety."""

    assistant = AnalyticsAssistant(RuleBasedProvider())
    rows = []
    with AnalyticsDatabase() as database:
        for index, case in enumerate(CASES, start=1):
            answer = assistant.ask(case.question)
            reference = database.query(case.reference_sql).frame
            exact_result = normalized(answer.frame).equals(normalized(reference))
            required_columns = set(reference.columns).issubset(answer.frame.columns)
            recommended = recommend_chart(
                answer.frame,
                answer.plan.chart_type,
                answer.plan.x,
                answer.plan.y,
                answer.plan.color,
            )
            chart_valid = chart_is_compatible(
                answer.frame,
                ChartSpec(recommended.chart_type, recommended.x, recommended.y, recommended.color),
            )
            single_attempt = answer.attempts == 1
            grounded_narrative = bool(answer.narrative) and "no records" not in answer.narrative.lower()
            checks = [exact_result, required_columns, chart_valid, single_attempt, grounded_narrative]
            rows.append(
                {
                    "id": index,
                    "question": case.question,
                    "exact_result": exact_result,
                    "required_columns": required_columns,
                    "chart_valid": chart_valid,
                    "attempts": answer.attempts,
                    "grounded_narrative": grounded_narrative,
                    "execution_ms": round(answer.elapsed_ms, 3),
                    "status": "PASS" if all(checks) else "FAIL",
                }
            )
    passed = sum(row["status"] == "PASS" for row in rows)
    result = {
        "benchmark": "ten_question_accuracy",
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "provider": "offline-fallback",
        "questions": len(rows),
        "passed": passed,
        "accuracy": passed / len(rows),
        "criteria_per_question": [
            "exact numeric/tabular result",
            "required result columns",
            "compatible chart recommendation",
            "success without retry",
            "non-empty grounded narrative",
        ],
        "results": rows,
    }
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_DIR / "question_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    markdown = [
        "# Ten-question benchmark",
        "",
        f"Provider: `{result['provider']}`  ",
        f"Accuracy: **{passed}/{len(rows)} ({result['accuracy']:.0%})**",
        "",
        "| # | Question | Exact | Columns | Chart | Attempts | Narrative | Status |",
        "|---:|---|:---:|:---:|:---:|---:|:---:|:---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['id']} | {row['question']} | {pass_mark(row['exact_result'])} | "
            f"{pass_mark(row['required_columns'])} | {pass_mark(row['chart_valid'])} | "
            f"{row['attempts']} | {pass_mark(row['grounded_narrative'])} | "
            f"**{row['status']}** |"
        )
    (BENCHMARK_DIR / "question_benchmark.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    output = run_benchmark()
    print(json.dumps({key: output[key] for key in ("questions", "passed", "accuracy")}, indent=2))
    raise SystemExit(0 if output["accuracy"] == 1 else 1)
