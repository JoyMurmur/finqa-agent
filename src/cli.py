"""
Main typer app for ConvFinQA
"""

import typer
from dotenv import load_dotenv
from rich import print as rich_print

from src.agent import Agent
from src.agent.settings import load_configs
from src.data import doc_to_markdown, get_record

load_dotenv()

# -------------- Load Configurations --------------
agent = Agent(*load_configs())

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
