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

from src.agent.settings import LLMConfig
from src.data import ConvFinQARecord, doc_to_markdown

# -------------- Configurations --------------
_config_dir = Path(__file__).parents[3] / "config"
prompts = yaml.safe_load((_config_dir / "prompts.yaml").read_text())

solver_config = LLMConfig(model="gemini-3-flash-preview")


# ---------- Structured Output Schemas ----------
class SolverResponse(BaseModel):
    reasoning: str = Field(
        description="Briefly explain the formula and steps used to get the answer",
    )
    answer: float = Field(description="The numeric answer to the question")


class AgentState(TypedDict):
    # document context
    document_context: str

    # generated output
    messages: Annotated[Sequence[BaseMessage], add_messages]
    solver: SolverResponse | None


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
    """Invoke the solver LLM and return the structured answer."""
    structured_model = get_llm(solver_config).with_structured_output(
        schema=SolverResponse.model_json_schema(), method="json_schema"
    )
    response = structured_model.invoke(state["messages"])

    return {
        "messages": [AIMessage(content=str(response["answer"]))],
        "solver": response,
    }


# -------------- Compile Workflow --------------
def build_workflow() -> CompiledStateGraph:
    """Build and compile the single-node solver workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("solver", solver_node)
    workflow.add_edge(START, "solver")
    workflow.add_edge("solver", END)

    return workflow.compile()


baseline_agent = build_workflow()


# -------------- Utility Functions to Interact with the Agent --------------
def initialize_chat(record: ConvFinQARecord) -> dict:
    """Create the initial agent state for a given record."""
    formatted_context = doc_to_markdown(record)
    system_prompt = prompts["solver"].format(context=formatted_context)
    return {
        "messages": [SystemMessage(content=system_prompt)],
        "document_context": formatted_context,
        "solver": None,
    }


def chat_turn(state: dict, user_message: str) -> tuple[dict, str]:
    """Run one synchronous conversation turn and return the updated state and reply."""
    state["messages"].append(HumanMessage(content=user_message))
    response = baseline_agent.invoke(state)
    reply = response["messages"][-1].content
    return response, reply


async def async_chat_turn(state: dict, user_message: str) -> tuple[dict, str]:
    """Run one asynchronous conversation turn and return the updated state and reply."""
    state["messages"].append(HumanMessage(content=user_message))
    response = await baseline_agent.ainvoke(state)
    reply = response["messages"][-1].content
    return response, reply
