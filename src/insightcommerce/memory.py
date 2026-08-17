"""Bounded conversational memory for follow-up analytics questions."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConversationTurn:
    """One successful question-to-query interaction."""

    question: str
    sql: str
    answer: str


class ConversationMemory:
    """Keep only the five most recent successful turns by default."""

    def __init__(self, max_turns: int = 5) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def add(self, question: str, sql: str, answer: str) -> None:
        """Append one grounded conversation turn."""

        self._turns.append(ConversationTurn(question=question, sql=sql, answer=answer))

    def clear(self) -> None:
        """Forget all stored turns."""

        self._turns.clear()

    def as_list(self) -> list[dict[str, str]]:
        """Return serializable memory for session state and tests."""

        return [asdict(turn) for turn in self._turns]

    def prompt_context(self) -> str:
        """Render recent turns as compact, explicitly untrusted context."""

        if not self._turns:
            return "No previous turns."
        blocks = []
        for index, turn in enumerate(self._turns, start=1):
            blocks.append(
                f"Turn {index}\nQuestion: {turn.question}\nSQL: {turn.sql}\nAnswer: {turn.answer}"
            )
        return "\n\n".join(blocks)

    def __len__(self) -> int:
        return len(self._turns)

