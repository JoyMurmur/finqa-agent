from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.agent.settings import AgentConfig, LLMConfig
from src.data import ConvFinQARecord, doc_to_markdown

# -------------- Configurations --------------
_config_dir = Path(__file__).parents[3] / "config"
prompts = yaml.safe_load((_config_dir / "prompts.yaml").read_text())

agent_config = AgentConfig(reflection_max_retries=1)

solver_config = LLMConfig(model="gemini-3-flash-preview")
reflector_config = LLMConfig(model="gemini-3-flash-preview")


# -------------- Utilities --------------
def _format_previous_turns(previous_turns: list[dict[str, str]] | None) -> str:
    if not previous_turns:
        return "None"
    lines: list[str] = []
    for i, turn in enumerate(previous_turns, start=1):
        lines.append(f"Turn {i} Question: {turn['question']}")
        lines.append(f"Turn {i} Answer: {turn['answer']}")
    return "\n".join(lines)


# ---------- Structured Output Schemas ----------
class SolverResponse(BaseModel):
    reasoning: str = Field(description="Formula and steps used to compute the answer")
    answer: float = Field(description="The numeric answer to the question")


class Reflection(BaseModel):
    is_correct: bool = Field(description="Whether the answer is correct")
    critique: str | None = Field(
        default=None,
        description="One sentence: what is wrong and what value/operation to use instead. No calculations."
        "If correct, leave empty.",
    )


# ---------- Agent State ----------
class AgentState(TypedDict):
    # Contextual information
    document_context: str

    # Conversation history
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Output
    solver: SolverResponse | None
    reflection: Reflection | None
    retry_count: int


# -------------- Agent Nodes --------------
def _extract_turns(messages: Sequence[BaseMessage]) -> tuple[str, list[dict[str, str]]]:
    """Return (current_question, previous_turns) derived from message history."""
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
    current_question = human_msgs[-1].content if human_msgs else ""
    previous_turns = [
        {"question": str(h.content), "answer": str(a.content)}
        for h, a in zip(human_msgs, ai_msgs, strict=False)
    ]
    return current_question, previous_turns


def get_llm(config: LLMConfig) -> ChatGoogleGenerativeAI:
    """Instantiate a ChatGoogleGenerativeAI model from an LLMConfig."""
    return ChatGoogleGenerativeAI(
        model=config.model,
        location=config.location,
        vertexai=config.vertexai,
        project=config.project,
        temperature=config.temperature,
        thinking_budget=config.thinking_budget,
        seed=config.seed,
        max_retries=config.max_retries,
        request_timeout=config.request_timeout,
    )


def solver_node(state: AgentState) -> dict:
    """Invoke the solver LLM, optionally injecting reflection critique on retry."""
    structured_model = get_llm(solver_config).with_structured_output(
        schema=SolverResponse.model_json_schema(), method="json_schema"
    )

    messages = list(state["messages"])

    # If this is a retry, add reflection feedback to the prompt
    if state["reflection"] is not None and not state["reflection"]["is_correct"]:
        messages.append(
            HumanMessage(
                content=f"Critique: {state['reflection']['critique']}. Try again."
            )
        )

    response = structured_model.invoke(messages)

    return {
        "solver": response,
    }


def reflect_node(state: AgentState) -> dict:
    """Invoke the reflector LLM to assess the solver's answer."""
    structured_model = get_llm(reflector_config).with_structured_output(
        schema=Reflection.model_json_schema(), method="json_schema"
    )

    current_question, previous_turns = _extract_turns(state["messages"])

    reflection_prompt = prompts["reflector"].format(
        document_context=state["document_context"],
        current_question=current_question,
        candidate_answer=state["solver"]["answer"],
        previous_turns=_format_previous_turns(previous_turns),
    )

    response = structured_model.invoke([HumanMessage(content=reflection_prompt)])

    return {
        "reflection": response,
        "retry_count": state["retry_count"] + 1,
    }


def should_continue(state: AgentState) -> str:
    """Route to END or back to generate."""
    if state["reflection"]["is_correct"]:
        return "output"
    if state["retry_count"] > agent_config.reflection_max_retries:
        return "output"
    return "solver"


def output_node(state: AgentState) -> dict:
    """Emit the final answer as an AI message."""
    return {
        "messages": [AIMessage(content=str(state["solver"]["answer"]))],
    }


# -------------- Compile Workflow --------------
def build_workflow() -> CompiledStateGraph:
    """Build and compile the solver-reflector workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("solver", solver_node)
    workflow.add_node("reflector", reflect_node)
    workflow.add_node("output", output_node)
    workflow.add_edge(START, "solver")
    workflow.add_edge("solver", "reflector")
    workflow.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "solver": "solver",
            "output": "output",
        },
    )
    workflow.add_edge("output", END)
    return workflow.compile()


agent = build_workflow()


# -------------- Utility Functions to Interact with the Agent --------------
def _per_turn_reset() -> dict:
    return {"solver": None, "reflection": None, "retry_count": 0}


def initialize_chat(record: ConvFinQARecord) -> dict:
    """Create the initial agent state for a given record."""
    formatted_context = doc_to_markdown(record)
    return {
        "document_context": formatted_context,
        "messages": [
            SystemMessage(content=prompts["solver"].format(context=formatted_context))
        ],
        **_per_turn_reset(),
    }


def chat_turn(state: AgentState, user_message: str) -> tuple[dict, str]:
    """Run one synchronous conversation turn and return the updated state and reply."""
    state = {
        **state,
        **_per_turn_reset(),
        "messages": [*state["messages"], HumanMessage(content=user_message)],
    }
    response = agent.invoke(state)
    return response, response["messages"][-1].content


async def async_chat_turn(state: AgentState, user_message: str) -> tuple[dict, str]:
    """Run one asynchronous conversation turn and return the updated state and reply."""
    state = {
        **state,
        **_per_turn_reset(),
        "messages": [*state["messages"], HumanMessage(content=user_message)],
    }
    response = await agent.ainvoke(state)
    return response, response["messages"][-1].content
