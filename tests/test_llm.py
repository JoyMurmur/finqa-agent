"""Tests for GenAI client connection and LLM invocation.

These tests require a valid GOOGLE_API_KEY or GEMINI_API_KEY (or VertexAI credentials)
and are skipped automatically when credentials are absent.
Users should set up test credentials in a .env file or environment variables to enable these tests.
Config values are
"""

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agent.settings import AgentConfig, LLMConfig

load_dotenv()

HAS_API_KEY = bool(
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
)

skip_no_creds = pytest.mark.skipif(
    not HAS_API_KEY, reason="No Google API credentials set"
)


def test_llm_config_defaults():
    # Update these values if defaults in LLMConfig change intentionally.
    cfg = LLMConfig()
    assert cfg.temperature == 1
    assert cfg.max_output_tokens == 1024
    assert cfg.max_retries == 3


def test_agent_config_defaults():
    # Update these values if defaults in AgentConfig change intentionally.
    cfg = AgentConfig()
    assert cfg.reflection_max_retries == 2
    assert cfg.max_tool_calls == 3


@skip_no_creds
def test_llm_invoke_simple():
    # Update model name if the default model changes.
    llm = LLMConfig(thinking_budget=None).build_llm()
    response = llm.invoke([HumanMessage(content="Reply with the single word: hello")])
    assert response.content
    assert isinstance(response.content, str | list)


@skip_no_creds
def test_llm_structured_output():
    from src.agent.state import SolverResponse

    llm = LLMConfig(thinking_budget=None).build_llm()
    structured = llm.with_structured_output(
        schema=SolverResponse.model_json_schema(), method="json_schema"
    )
    result = structured.invoke(
        [HumanMessage(content='Return reasoning="2+2" and answer=4.0')]
    )
    assert result["answer"] == pytest.approx(4.0)
