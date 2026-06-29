"""
Implements all LangGraph node functions and routing logic for the agent workflow.
AgentNodes holds the LLM instances and is instantiated once by Agent in agent.py.

Note that logger is kept for runtime observability for some nodes
For production, consider replacing with tracing and structured logging.
"""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.agent.settings import REFLECTOR_PROMPT_TEMPLATE, AgentConfig, LLMConfig
from src.agent.state import AgentState, ReflectionResponse, SolverResponse
from src.agent.tools import tools
from src.logger import get_logger

logger = get_logger(__name__)


def _format_previous_turns(previous_turns: list[dict[str, str]] | None) -> str:
    """Format previous turns as a string for prompt injection. If no previous turns, return "None"."""
    if not previous_turns:
        return "None"
    lines: list[str] = []
    for i, turn in enumerate(previous_turns, start=1):
        lines.append(f"Turn {i} Question: {turn['question']}")
        lines.append(f"Turn {i} Answer: {turn['answer']}")
    return "\n".join(lines)


class AgentNodes:
    def __init__(
        self,
        solver_config: LLMConfig,
        reflector_config: LLMConfig,
        agent_config: AgentConfig,
    ) -> None:
        """Initialise agent nodes with LLM configs and retry/tool call limits."""
        self.reflection_max_retries = agent_config.reflection_max_retries
        self.max_tool_calls = agent_config.max_tool_calls

        # Instantiate LLM once to avoid repeated instance creation across nodes
        _solver_llm = solver_config.build_llm()
        self._solver_llm_with_tools = _solver_llm.bind_tools(tools, tool_choice="auto")
        self._solver_llm_structured = _solver_llm.with_structured_output(
            schema=SolverResponse.model_json_schema(), method="json_schema"
        )
        self._reflector_llm_structured = (
            reflector_config.build_llm().with_structured_output(
                schema=ReflectionResponse.model_json_schema(), method="json_schema"
            )
        )

    def _extract_turns(
        self, messages: Sequence[BaseMessage]
    ) -> tuple[str, list[dict[str, str]]]:
        """Return (current_question, previous_turns) derived from message history."""
        msgs = list(messages)
        human_indices = [i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)]
        if not human_indices:
            return "", []

        current_question = msgs[human_indices[-1]].text

        previous_turns = []
        for start, end in zip(human_indices, human_indices[1:], strict=False):
            segment = msgs[start:end]
            answer = next(
                (
                    m
                    for m in reversed(segment)
                    if isinstance(m, AIMessage)
                    and not m.additional_kwargs.get("function_call")
                ),
                None,
            )
            if answer:
                previous_turns.append(
                    {"question": msgs[start].text, "answer": answer.text}
                )

        return current_question, previous_turns

    def _build_solver_messages(self, state: AgentState) -> list[BaseMessage]:
        """Build solver input messages and inject critique feedback for retries."""
        messages = list(state["messages"])

        # If this is a retry, add reflection feedback to the prompt
        if state["reflection"] is not None and not state["reflection"]["is_correct"]:
            messages.append(
                HumanMessage(
                    content=f"Critique: {state['reflection']['critique']}. Try again."
                )
            )

        return messages

    def solver_node(self, state: AgentState) -> dict:
        """Invoke solver LLM with tools and update message history."""
        messages = self._build_solver_messages(state)
        response = self._solver_llm_with_tools.invoke(messages)
        tool_calls_made = len(getattr(response, "tool_calls", []))
        logger.info(
            "solver completed: input_messages=%s, tool_calls_made=%s has_tool_calls=%s",
            messages[-1],
            tool_calls_made,
            bool(tool_calls_made),
        )
        return {
            "messages": [response],
            "tool_call_count": state["tool_call_count"] + tool_calls_made,
        }

    def route_after_solver(self, state: AgentState) -> str:
        """Route to tool node or extract node after solver response."""
        if (
            getattr(state["messages"][-1], "tool_calls", None)
            and state["tool_call_count"] < self.max_tool_calls
        ):
            return "tools"
        return "extract"

    def extract_node(self, state: AgentState) -> dict:
        """Extract structured answer from solver output."""
        response: SolverResponse = self._solver_llm_structured.invoke(state["messages"])
        logger.info("extract completed: answer=%s", response["answer"])
        return {"solver": response}

    def reflect_node(self, state: AgentState) -> dict:
        """Critique solver answer and update reflection state."""
        current_question, previous_turns = self._extract_turns(state["messages"])

        assert state["solver"] is not None
        formatted_prompt = REFLECTOR_PROMPT_TEMPLATE.format_messages(
            document_context=state["document_context"],
            current_question=current_question,
            candidate_answer=state["solver"]["answer"],
            previous_turns=_format_previous_turns(previous_turns),
        )

        formatted_prompt.append(
            HumanMessage(content="Review the candidate answer now.")
        )
        response: ReflectionResponse = self._reflector_llm_structured.invoke(
            formatted_prompt
        )

        logger.info(
            "reflect completed: question=%s answer=%s previous_turns=%s is_correct=%s critique=%s",
            current_question,
            state["solver"]["answer"],
            _format_previous_turns(previous_turns),
            response["is_correct"],
            response["critique"],
        )
        return {
            "reflection": response,
            "retry_count": state["retry_count"] + 1,
        }

    def should_continue(self, state: AgentState) -> str:
        """Route to END or back to solver node."""
        assert state["reflection"] is not None

        if state["reflection"]["is_correct"]:
            return "output"
        if state["retry_count"] > self.reflection_max_retries:
            return "output"
        return "solver"

    def output_node(self, state: AgentState) -> dict:
        """Emit final answer as an AIMessage to user."""
        assert state["solver"] is not None
        return {
            "messages": [AIMessage(content=str(state["solver"]["answer"]))],
        }
