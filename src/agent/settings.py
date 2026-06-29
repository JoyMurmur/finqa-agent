"""
Loads and exposes all agent settings as typed Python objects.
The YAML files in config/ hold the raw values; this module translates them into
dataclasses and prompt templates from raw dicts or strings.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

_config_dir = Path(__file__).parents[2] / "config"
_prompts = yaml.safe_load((_config_dir / "prompts.yaml").read_text())

SOLVER_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("system", _prompts["solver"])]
)
REFLECTOR_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("system", _prompts["reflector"])]
)


@dataclass
class LLMConfig:
    # Google GenAI API settings (from env)
    vertexai: bool | None = None
    project: str | None = None
    location: str = "global"

    # LLM parameters
    model: str = "gemini-3-flash-preview"
    temperature: float = 1
    max_output_tokens: int = 1024
    seed: int = 42
    max_retries: int = 3
    request_timeout: int = 120

    # for gemini 3+, default is high and thinking_budget is deprecated
    thinking_level: str | None = "high"
    # for gemini 2+, thinking_budget is used
    thinking_budget: int | None = 1024

    def __post_init__(self) -> None:
        self.model = os.getenv("DEFAULT_LLM_MODEL", self.model)
        self.vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in (
            "true",
            "1",
        )
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if self.model.startswith("gemini-3"):
            self.thinking_budget = None
        elif self.model.startswith("gemini-2"):
            self.thinking_level = None

    def build_llm(self) -> ChatGoogleGenerativeAI:
        """Instantiate the LLM from this config."""
        return ChatGoogleGenerativeAI(
            model=self.model,
            location=self.location,
            vertexai=self.vertexai,
            project=self.project,
            temperature=self.temperature,
            thinking_budget=self.thinking_budget,
            seed=self.seed,
            max_retries=self.max_retries,
            request_timeout=self.request_timeout,
        )


@dataclass
class AgentConfig:
    reflection_max_retries: int = 2  # max retries for reflection loop per question
    max_tool_calls: int = 3  # max calculator tool calls per question
