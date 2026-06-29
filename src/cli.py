"""
Main typer app for ConvFinQA
"""

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich import print as rich_print

from src.agent import Agent
from src.agent.settings import AgentConfig, LLMConfig
from src.data import doc_to_markdown, get_record

load_dotenv()

# -------------- Load Configurations --------------
_config_dir = Path(__file__).parent.parent / "config"
_llm = yaml.safe_load((_config_dir / "llm.yaml").read_text())
_agent = yaml.safe_load((_config_dir / "agent.yaml").read_text())

solver_config = LLMConfig(**_llm["solver"])
reflector_config = LLMConfig(**_llm["reflector"])
agent_config = AgentConfig(**_agent)

app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)


# -------------- Main App Logic --------------
@app.callback()
def main_callback() -> None:
    """ConvFinQA CLI entrypoint."""


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Ask questions about a specific record"""
    record = get_record(record_id)

    # Validate the record exists before starting the chat loop
    if record is None:
        rich_print(f"[red]Record not found:[/red] [bold]{record_id}[/bold]")
        raise typer.Exit(code=1)

    # Initialise the agent with the record and start the chat loop
    agent = Agent(
        solver_config=solver_config,
        reflector_config=reflector_config,
        agent_config=agent_config,
    )
    # Build initial state with the record context and system prompt once
    context = doc_to_markdown(record)
    state = agent.initialize_chat(context)
    rich_print(
        f"[green]Loaded record [bold]{record_id}[/bold]. Type 'exit' to quit.[/green]"
    )
    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        # Apply per-turn updates and emit the agent's reply (numeric answer)
        state, reply = agent.chat_turn(state, message)
        rich_print(f"[blue][bold]assistant:[/bold] {reply}[/blue]")


if __name__ == "__main__":
    app()
