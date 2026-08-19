"""Tests for local Ollama compatibility and useful provider errors."""

from __future__ import annotations

import pytest
import requests

from insightcommerce.config import LLMSettings
from insightcommerce.providers import LLMProviderError, OllamaProvider


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} response")

    def json(self) -> dict:
        return self._payload


def test_ollama_uses_portable_json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_post(url: str, *, json: dict, timeout: int) -> FakeResponse:
        captured.update(url=url, payload=json, timeout=timeout)
        return FakeResponse(200, {"message": {"content": '{"sql":"SELECT 1"}'}})

    monkeypatch.setattr("insightcommerce.providers.requests.post", fake_post)
    settings = LLMSettings(ollama_model="qwen3:8b", timeout_seconds=120)

    output = OllamaProvider(settings).generate("system", "user")

    assert output == '{"sql":"SELECT 1"}'
    assert captured["payload"]["model"] == "qwen3:8b"
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["think"] is False
    assert '"chart_type"' in captured["payload"]["messages"][0]["content"]
    assert "Do not nest chart fields" in captured["payload"]["messages"][0]["content"]
    assert captured["timeout"] == 120


def test_ollama_http_error_includes_server_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, *, json: dict, timeout: int) -> FakeResponse:
        del url, json, timeout
        return FakeResponse(400, {}, '{"error":"failed to parse grammar"}')

    monkeypatch.setattr("insightcommerce.providers.requests.post", fake_post)

    with pytest.raises(LLMProviderError, match="failed to parse grammar"):
        OllamaProvider(LLMSettings()).generate("system", "user")
