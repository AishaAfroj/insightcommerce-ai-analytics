"""LLM provider adapters plus a deterministic offline demonstration provider."""

from __future__ import annotations

import json
import re
from typing import Protocol

import requests

from .config import LLMSettings
from .query_models import query_plan_json_schema


class LLMProviderError(RuntimeError):
    """Raised when an LLM endpoint is unavailable or returns unusable content."""


class LLMProvider(Protocol):
    """Minimal interface consumed by the analytics assistant."""

    name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return a JSON string representing a query plan."""


class OllamaProvider:
    """Local, free LLM adapter using Ollama's chat endpoint."""

    name = "ollama"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": query_plan_json_schema(),
            "options": {"temperature": 0, "num_predict": 900},
        }
        try:
            response = requests.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            return str(response.json()["message"]["content"])
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError(f"Ollama request failed: {exc}") from exc


class OpenAIResponsesProvider:
    """Optional hosted adapter using the OpenAI Responses API Structured Outputs."""

    name = "openai"

    def __init__(self, settings: LLMSettings) -> None:
        if not settings.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is not configured.")
        self.settings = settings

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.settings.openai_model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "insightcommerce_query_plan",
                    "strict": True,
                    "schema": query_plan_json_schema(),
                }
            },
            "store": False,
            "max_output_tokens": 1_000,
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            for item in data.get("output", []):
                if item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return str(content["text"])
            raise LLMProviderError("OpenAI response contained no output_text item.")
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc


class RuleBasedProvider:
    """Offline fallback for common demo questions; never pretends to be a live LLM."""

    name = "offline-fallback"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        question_match = re.search(r"CURRENT QUESTION\n(.+?)(?:\n\n|$)", user_prompt, re.S)
        question = (question_match.group(1) if question_match else user_prompt).strip().lower()
        plan = self._plan(question)
        return json.dumps(plan)

    @staticmethod
    def _plan(question: str) -> dict[str, str]:
        if "total revenue" in question:
            return {
                "sql": "SELECT SUM(order_value) AS total_revenue FROM orders",
                "title": "Total revenue",
                "chart_type": "metric",
                "x": "",
                "y": "total_revenue",
                "color": "",
                "rationale": "Summing order value produces the dataset's total simulated revenue.",
            }
        if "how many orders" in question or "order count" in question:
            return {
                "sql": "SELECT COUNT(*) AS orders FROM orders",
                "title": "Total order count",
                "chart_type": "metric",
                "x": "",
                "y": "orders",
                "color": "",
                "rationale": "Counting transaction rows produces the total number of orders.",
            }
        if "units" in question or "quantity sold" in question:
            return {
                "sql": "SELECT SUM(quantity) AS units FROM orders",
                "title": "Total units sold",
                "chart_type": "metric",
                "x": "",
                "y": "units",
                "color": "",
                "rationale": "Summing quantity produces the total number of simulated units sold.",
            }
        if "monthly" in question or "trend" in question or "over time" in question:
            return {
                "sql": "SELECT date_trunc('month', date) AS month, SUM(order_value) AS revenue FROM orders GROUP BY 1 ORDER BY 1",
                "title": "Monthly revenue trend",
                "chart_type": "line",
                "x": "month",
                "y": "revenue",
                "color": "",
                "rationale": "Monthly aggregation shows how simulated revenue changes through 2025.",
            }
        if "q4" in question or "quarter" in question:
            return {
                "sql": "SELECT quarter, SUM(order_value) AS revenue, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY 1",
                "title": "Quarterly performance",
                "chart_type": "bar",
                "x": "quarter",
                "y": "revenue",
                "color": "",
                "rationale": "Quarterly revenue and order counts expose the modeled Q4 seasonality.",
            }
        if "categor" in question:
            return {
                "sql": "SELECT product_category, SUM(order_value) AS revenue, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY revenue DESC",
                "title": "Revenue by product category",
                "chart_type": "bar",
                "x": "product_category",
                "y": "revenue",
                "color": "",
                "rationale": "The derived product taxonomy supports transparent category comparison.",
            }
        if "product" in question:
            return {
                "sql": "SELECT product, SUM(order_value) AS revenue, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY revenue DESC LIMIT 10",
                "title": "Top products by revenue",
                "chart_type": "bar",
                "x": "product",
                "y": "revenue",
                "color": "",
                "rationale": "Product-level aggregation identifies the ten strongest revenue contributors.",
            }
        if "average" in question or "aov" in question:
            return {
                "sql": "SELECT country, AVG(order_value) AS average_order_value FROM orders GROUP BY 1 ORDER BY average_order_value DESC",
                "title": "Average order value by country",
                "chart_type": "bar",
                "x": "country",
                "y": "average_order_value",
                "color": "",
                "rationale": "Mean order value enables like-for-like comparison across country markets.",
            }
        return {
            "sql": "SELECT country, SUM(order_value) AS revenue, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY revenue DESC",
            "title": "Revenue by country",
            "chart_type": "choropleth",
            "x": "country",
            "y": "revenue",
            "color": "",
            "rationale": "Country aggregation answers the broadest market-performance question.",
        }


def build_provider(name: str, settings: LLMSettings | None = None) -> LLMProvider:
    """Construct one configured provider by display-friendly name."""

    resolved = (settings or LLMSettings.from_environment())
    normalized = name.strip().lower()
    if normalized in {"ollama", "local"}:
        return OllamaProvider(resolved)
    if normalized in {"openai", "hosted"}:
        return OpenAIResponsesProvider(resolved)
    if normalized in {"offline", "fallback", "offline-fallback"}:
        return RuleBasedProvider()
    raise ValueError(f"Unknown LLM provider: {name}")
