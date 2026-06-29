import ast
import operator as op
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from src.agent.settings import AgentConfig, LLMConfig
from src.data import ConvFinQARecord, doc_to_markdown

# -------------- Configurations --------------
_config_dir = Path(__file__).parents[3] / "config"
prompts = yaml.safe_load((_config_dir / "prompts.yaml").read_text())

agent_config = AgentConfig(
    reflection_max_retries=1,
    max_tool_calls=3,
)

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


# ---------- Calculator Tool ----------
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool()
def calculate(expression: str) -> str:
    """
    Evaluates arithmetic expressions to solve financial reasoning tasks.
    Use this tool to derive metrics from values extracted from text and tables.

    Conceptual templates for financial conventions:
    - Portion/Contribution: Segment_Value / Total_Value (to find share of total)
    - Year-over-Year Change: (Current_Year - Prior_Year) / Prior_Year
    - Margin Analysis: (Total_Revenue - Cost_of_Goods_Sold) / Total_Revenue
    - Scaling/Normalization: Value_in_Millions / 1000 (to convert to Billions)
    - Net Position: Positive_Inflow + (Negative_Outflow)
    - Average/Weighted Portion: (Value_A + Value_B) / Number_of_Periods

    Note: Convert values in parentheses (e.g., '(500)') to negative numbers (e.g., '-500')
    before passing to the expression.

    Args:
        expression: A string math expression (e.g., 'segment_value / total_value').
    """
    return str(_safe_eval(ast.parse(expression, mode="eval").body))


tools = [calculate]
tool_node = ToolNode(tools, handle_tool_errors=True)


# ---------- Structured Output Schemas ----------
class SolverResponse(BaseModel):
    reasoning: str = Field(description="Formula and steps used to compute the answer")
    answer: float = Field(description="The numeric answer to the question")


class ReflectionResponse(BaseModel):
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
    reflection: ReflectionResponse | None
    retry_count: int
    tool_call_count: int


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


# Initiate LLM instances to avoid repeated instantiation
_solver_llm = get_llm(solver_config)
_solver_llm_with_tools = _solver_llm.bind_tools(tools, tool_choice="auto")
_solver_llm_structured = _solver_llm.with_structured_output(
    schema=SolverResponse.model_json_schema(), method="json_schema"
)
_reflector_llm_structured = get_llm(reflector_config).with_structured_output(
    schema=ReflectionResponse.model_json_schema(), method="json_schema"
)


def solver_node(state: AgentState) -> dict:
    """Invoke the solver LLM with tools, optionally injecting reflection critique on retry."""
    messages = list(state["messages"])

    # If this is a retry, add reflection feedback to the prompt
    if state["reflection"] is not None and not state["reflection"]["is_correct"]:
        messages.append(
            HumanMessage(
                content=f"Critique: {state['reflection']['critique']}. Try again."
            )
        )

    response = _solver_llm_with_tools.invoke(messages)
    tool_calls_made = len(getattr(response, "tool_calls", []))
    return {
        "messages": [response],
        "tool_call_count": state["tool_call_count"] + tool_calls_made,
    }


def extract_node(state: AgentState) -> dict:
    """Extract the final structured answer from the message history."""
    result = _solver_llm_structured.invoke(state["messages"])
    return {"solver": result}


def reflect_node(state: AgentState) -> dict:
    """Invoke the reflector LLM to assess the solver's answer."""
    current_question, previous_turns = _extract_turns(state["messages"])

    reflection_prompt = prompts["reflector"].format(
        document_context=state["document_context"],
        current_question=current_question,
        candidate_answer=state["solver"],
        previous_turns=_format_previous_turns(previous_turns),
    )

    response = _reflector_llm_structured.invoke(
        [HumanMessage(content=reflection_prompt)]
    )
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
    """Build and compile the solver-reflector-tool workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("solver", solver_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("reflector", reflect_node)
    workflow.add_node("output", output_node)

    workflow.add_edge(START, "solver")
    workflow.add_conditional_edges(
        "solver",
        lambda s: (
            "tools"
            if getattr(s["messages"][-1], "tool_calls", None)
            and s["tool_call_count"] < agent_config.max_tool_calls
            else "extract"
        ),
        {"tools": "tools", "extract": "extract"},
    )
    workflow.add_edge("tools", "solver")
    workflow.add_edge("extract", "reflector")
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
    return {"solver": None, "reflection": None, "retry_count": 0, "tool_call_count": 0}


def initialize_chat(record: ConvFinQARecord) -> dict:
    """Create the initial agent state for a given record."""
    formatted_context = doc_to_markdown(record)
    system_prompt = prompts["solver"].format(context=formatted_context)
    return {
        "document_context": formatted_context,
        "messages": [SystemMessage(content=system_prompt)],
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
