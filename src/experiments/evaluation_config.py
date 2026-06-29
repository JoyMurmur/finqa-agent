import importlib

AGENT_REGISTRY = {
    "baseline": {
        "module": "src.experiments.agent_candidates.baseline_agent",
        "run_name": "solver-only-agent",
        "source_file": "experiments/agent_candidates/baseline_agent.py",
    },
    "reflection": {
        "module": "src.experiments.agent_candidates.reflection_agent",
        "run_name": "solver-reflector-agent",
        "source_file": "experiments/agent_candidates/reflection_agent.py",
    },
    "reflection_with_tool": {
        "module": "src.experiments.agent_candidates.reflection_agent_with_tool",
        "run_name": "solver-reflector-tool-agent",
        "source_file": "experiments/agent_candidates/reflection_agent_with_tool.py",
    },
}


def load_selected_agent_runtime(
    selected_agent: str | None = None,
) -> dict:
    """Load the runtime components of the selected agent candidate."""
    key = selected_agent.strip().lower()

    if key not in AGENT_REGISTRY:
        valid = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(f"Unknown AGENT_VARIANT='{key}'. Valid options: {valid}")

    meta = AGENT_REGISTRY[key]
    module = importlib.import_module(meta["module"])

    return {
        "key": key,
        "module_path": meta["module"],
        "run_name": meta["run_name"],
        "source_file": meta["source_file"],
        "initialize_chat": module.initialize_chat,
        "chat_turn": module.chat_turn,
        "async_chat_turn": module.async_chat_turn,
        "solver_config": module.solver_config,
        "reflector_config": (
            module.reflector_config if hasattr(module, "reflector_config") else None
        ),
    }
