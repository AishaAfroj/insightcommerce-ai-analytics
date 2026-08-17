"""Tests for the required five-turn conversational memory."""

from insightcommerce.memory import ConversationMemory


def test_memory_keeps_only_five_turns() -> None:
    memory = ConversationMemory(max_turns=5)
    for number in range(7):
        memory.add(f"q{number}", f"SELECT {number}", f"a{number}")
    turns = memory.as_list()
    assert len(turns) == 5
    assert turns[0]["question"] == "q2"
    assert turns[-1]["question"] == "q6"

