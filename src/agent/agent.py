"""
Defines the Agent class that owns the LangGraph workflow and exposes chat turn methods.
Use initialize_chat to create state, then chat_turn per user message.

Graph:
START → solver ─┬─(tool calls)──→ tools → solver (loop)
                └─(no tool calls)→ extract → reflector ─┬─(correct / max retries) → output → END
                                                        └─(incorrect) → solver (retry)
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.nodes import AgentNodes
from src.agent.settings import SOLVER_PROMPT_TEMPLATE, AgentConfig, LLMConfig
from src.agent.state import AgentState
from src.agent.tools import tool_node


class Agent:
    def __init__(
        self,
        solver_config: LLMConfig,
        reflector_config: LLMConfig,
        agent_config: AgentConfig,
    ) -> None:
        self._nodes = AgentNodes(
            solver_config=solver_config,
            reflector_config=reflector_config,
            agent_config=agent_config,
        )
        self._graph = self._build_workflow()

    def _build_workflow(self) -> CompiledStateGraph:
        """Build and compile the agent workflow graph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("solver", self._nodes.solver_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("extract", self._nodes.extract_node)
        workflow.add_node("reflector", self._nodes.reflect_node)
        workflow.add_node("output", self._nodes.output_node)

        workflow.add_edge(START, "solver")
        workflow.add_conditional_edges(
            "solver",
            self._nodes.route_after_solver,
            {"tools": "tools", "extract": "extract"},
        )
        workflow.add_edge("tools", "solver")
        workflow.add_edge("extract", "reflector")
        workflow.add_conditional_edges(
            "reflector",
            self._nodes.should_continue,
            {"solver": "solver", "output": "output"},
        )
        workflow.add_edge("output", END)
        return workflow.compile()

    # -------------- Utility Functions to Interact with the Agent --------------
    def _apply_turn(self, state: AgentState, user_message: str) -> AgentState:
        """Add new user message to message history and reset per-turn variables."""
        return {
            # keep document context the same for the entire conversation
            "document_context": state["document_context"],
            # add user message to history
            "messages": [*state["messages"], HumanMessage(content=user_message)],
            # reset per-turn variables
            "solver": None,
            "reflection": None,
            "retry_count": 0,
            "tool_call_count": 0,
        }

    def initialize_chat(self, document_context: str) -> AgentState:
        """Build initial agent state with system prompt for the given document context."""
        return {
            "document_context": document_context,
            "messages": SOLVER_PROMPT_TEMPLATE.format_messages(
                context=document_context
            ),
            "solver": None,
            "reflection": None,
            "retry_count": 0,
            "tool_call_count": 0,
        }

    def chat_turn(self, state: AgentState, user_message: str) -> tuple[AgentState, str]:
        """Run one synchronous turn and return updated state plus final answer."""
        response: AgentState = self._graph.invoke(self._apply_turn(state, user_message))
        reply_message: str = response["messages"][-1].content
        return response, reply_message
