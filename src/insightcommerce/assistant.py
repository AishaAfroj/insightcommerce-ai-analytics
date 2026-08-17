"""Schema-aware AI query orchestration with one safe corrective retry."""

from __future__ import annotations

from .memory import ConversationMemory
from .narrative import narrate_result
from .prompts import build_generation_prompt, build_retry_prompt
from .providers import LLMProvider
from .query_models import QueryAnswer, QueryPlan
from .sandbox import SafeSQLExecutor


class AnalyticsAssistantError(RuntimeError):
    """Raised after both the initial LLM plan and the single retry fail."""


class AnalyticsAssistant:
    """Convert a question to SQL, validate it, execute it, and remember the result."""

    def __init__(
        self,
        provider: LLMProvider,
        executor: SafeSQLExecutor | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.provider = provider
        self.executor = executor or SafeSQLExecutor()
        self.memory = memory if memory is not None else ConversationMemory(max_turns=5)

    def ask(self, question: str) -> QueryAnswer:
        """Run the initial plan plus at most one error-informed corrective retry."""

        clean_question = question.strip()
        if len(clean_question) < 3:
            raise ValueError("Please enter an analytics question with at least three characters.")

        system_prompt, user_prompt = build_generation_prompt(clean_question, self.memory)
        raw_output = ""
        first_error: Exception | None = None
        for attempt in (1, 2):
            try:
                raw_output = self.provider.generate(system_prompt, user_prompt)
                plan = QueryPlan.model_validate_json(raw_output)
                result = self.executor.execute(plan.sql)
                narrative = narrate_result(result.frame, plan.title)
                self.memory.add(clean_question, plan.sql, narrative)
                return QueryAnswer(
                    question=clean_question,
                    plan=plan,
                    frame=result.frame,
                    narrative=narrative,
                    elapsed_ms=result.elapsed_ms,
                    attempts=attempt,
                    provider=self.provider.name,
                )
            except Exception as exc:
                if attempt == 2:
                    raise AnalyticsAssistantError(
                        f"The query could not be generated safely after one retry: {exc}"
                    ) from exc
                first_error = exc
                system_prompt, user_prompt = build_retry_prompt(
                    clean_question, self.memory, raw_output, exc
                )
        raise AnalyticsAssistantError(f"Unexpected query failure: {first_error}")
