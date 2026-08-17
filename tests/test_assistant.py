"""Tests for structured plan parsing and exactly one corrective retry."""

import json

import pytest

from insightcommerce.assistant import AnalyticsAssistant, AnalyticsAssistantError
from insightcommerce.memory import ConversationMemory
from insightcommerce.sandbox import SafeSQLExecutor


def plan(sql: str) -> str:
    return json.dumps(
        {
            "sql": sql,
            "title": "Test result",
            "chart_type": "metric",
            "x": "",
            "y": "orders",
            "color": "",
            "rationale": "Test aggregation.",
        }
    )


class FakeProvider:
    name = "fake"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        output = self.outputs[self.calls]
        self.calls += 1
        return output


def test_one_retry_recovers_from_unsafe_plan() -> None:
    provider = FakeProvider([plan("DROP TABLE orders"), plan("SELECT COUNT(*) AS orders FROM orders")])
    memory = ConversationMemory(max_turns=5)
    assistant = AnalyticsAssistant(provider, SafeSQLExecutor(), memory)
    answer = assistant.ask("How many orders are there?")
    assert provider.calls == 2
    assert answer.attempts == 2
    assert int(answer.frame.iloc[0]["orders"]) == 108_300
    assert len(memory) == 1


def test_no_third_attempt_after_retry_failure() -> None:
    provider = FakeProvider([plan("DROP TABLE orders"), plan("DELETE FROM orders")])
    assistant = AnalyticsAssistant(provider)
    with pytest.raises(AnalyticsAssistantError):
        assistant.ask("Try an unsafe query")
    assert provider.calls == 2
