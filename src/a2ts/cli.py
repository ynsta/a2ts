"""CLI application and pipeline orchestration for a2ts."""

import typer
from rich.console import Console

app = typer.Typer(help="a2ts: Contextualized Single-Stream Audio/Video Transcriber")
console = Console()


@app.command()
def info() -> None:
    """Show system information and a2ts configuration."""
    console.print("[bold green]a2ts[/bold green] - Audio/Video Transcriber ready.")


if __name__ == "__main__":
    app()
