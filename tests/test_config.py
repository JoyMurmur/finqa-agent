"""Tests for prompt template contracts.

These tests exist to guard the prompt variable contracts — if a variable name is
renamed in the template, the nodes that inject it will silently break at runtime.
"""

from src.agent.settings import (
    REFLECTOR_PROMPT_TEMPLATE,
    SOLVER_PROMPT_TEMPLATE,
)


def test_solver_prompt_has_context_variable():
    assert "context" in SOLVER_PROMPT_TEMPLATE.input_variables


def test_reflector_prompt_has_required_variables():
    required = {
        "document_context",
        "previous_turns",
        "current_question",
        "candidate_answer",
    }
    assert required <= set(REFLECTOR_PROMPT_TEMPLATE.input_variables)


def test_solver_prompt_renders():
    messages = SOLVER_PROMPT_TEMPLATE.format_messages(context="some document text")
    assert len(messages) == 1
    assert "some document text" in messages[0].content


def test_reflector_prompt_renders():
    messages = REFLECTOR_PROMPT_TEMPLATE.format_messages(
        document_context="doc",
        previous_turns="None",
        current_question="What is the revenue?",
        candidate_answer="100.0",
    )
    assert len(messages) == 1
    content = messages[0].content
    assert "What is the revenue?" in content
    assert "100.0" in content
