"""
Defines the shared graph state and structured output schemas.
SolverResponse and ReflectionResponse are Pydantic models used for LLM structured output.
AgentState is the TypedDict used by every LangGraph node.
"""

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class SolverResponse(BaseModel):
    reasoning: str = Field(description="Formula and steps used to compute the answer")
    answer: float = Field(description="The numeric answer to the question")


class ReflectionResponse(BaseModel):
    is_correct: bool = Field(description="Whether the answer is correct")
    critique: str | None = Field(
        description="One sentence: what is wrong and what value/operation to use instead. No calculations."
        "If correct, leave empty.",
    )


# GRAPH STATE
class AgentState(TypedDict):
    # Contextual information
    document_context: str

    # Conversation history
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Output
    solver: SolverResponse | None
    reflection: ReflectionResponse | None

    # Counters
    retry_count: int
    tool_call_count: int
