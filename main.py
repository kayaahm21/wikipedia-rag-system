"""
main.py - CLI Entry Point for Local Wiki-RAG Assistant

Provides three commands:
  python main.py ingest  -- Scrape Wikipedia, chunk, embed, and store in ChromaDB
  python main.py chat    -- Interactive Q&A loop with streaming responses
  python main.py clear   -- Delete the vector database and reset the system

Within chat mode:
  /sources  -- Show the source chunks used for the last answer
  /stats    -- Show database statistics
  /help     -- Show available commands
  /quit     -- Exit chat mode
"""

import sys
import os

# Force UTF-8 output on Windows to support special characters
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(force_terminal=True)


def print_banner():
    """Display the application banner."""
    banner = Text()
    banner.append("Local Wiki-RAG Assistant", style="bold cyan")
    banner.append("\n  Powered by ", style="dim")
    banner.append("Ollama ", style="bold yellow")
    banner.append("+ ", style="dim")
    banner.append("ChromaDB", style="bold green")
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))


def cmd_ingest():
    """Run the full ingestion pipeline: scrape -> chunk -> embed -> store."""
    from ingest import ingest_all
    from database import store_chunks, get_collection_stats

    print_banner()
    console.print("[bold yellow]Starting ingestion pipeline...[/bold yellow]\n")

    # Step 1: Scrape and chunk
    chunks = ingest_all()

    if not chunks:
        console.print("[bold red]No chunks were generated. Aborting.[/bold red]")
        return

    # Step 2: Embed and store
    store_chunks(chunks)

    # Step 3: Show stats
    stats = get_collection_stats()
    table = Table(title="Database Statistics", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total Chunks", str(stats["total_chunks"]))
    table.add_row("People Chunks", str(stats["people_chunks"]))
    table.add_row("Places Chunks", str(stats["places_chunks"]))
    console.print(table)


def cmd_chat():
    """Enter the interactive Q&A chat loop."""
    from rag import generate_response
    from database import get_collection_stats

    print_banner()

    # Check if database has data
    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        console.print(
            "[bold red]No data found in database![/bold red]\n"
            "  Run [bold cyan]python main.py ingest[/bold cyan] first to populate the database.\n"
        )
        return

    console.print(f"[dim]Database loaded: {stats['total_chunks']} chunks "
                  f"({stats['people_chunks']} people, {stats['places_chunks']} places)[/dim]")
    console.print()
    console.print("[dim]Commands: /sources /stats /help /quit[/dim]")
    console.print("[dim]Type your question and press Enter.[/dim]\n")

    last_sources = []
    last_query_type = ""

    while True:
        try:
            query = console.input("[bold green]You > [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue

        # Handle commands
        if query.lower() == "/quit":
            console.print("[dim]Goodbye![/dim]")
            break

        if query.lower() == "/help":
            help_table = Table(title="Available Commands", border_style="cyan")
            help_table.add_column("Command", style="bold cyan")
            help_table.add_column("Description")
            help_table.add_row("/sources", "Show source chunks used for the last answer")
            help_table.add_row("/stats", "Show database statistics")
            help_table.add_row("/help", "Show this help message")
            help_table.add_row("/quit", "Exit chat mode")
            console.print(help_table)
            console.print()
            continue

        if query.lower() == "/sources":
            if not last_sources:
                console.print("[yellow]No sources available yet. Ask a question first.[/yellow]\n")
            else:
                console.print(f"\n[bold cyan]Sources for last query[/bold cyan] (type: {last_query_type})")
                for i, src in enumerate(last_sources, 1):
                    console.print(
                        f"  [{i}] [bold]{src['entity']}[/bold] ({src['type']}) "
                        f"-- relevance: {src['relevance']:.3f}"
                    )
                    console.print(f"      [dim]{src['chunk_preview']}[/dim]")
                console.print()
            continue

        if query.lower() == "/stats":
            stats = get_collection_stats()
            table = Table(title="Database Statistics", border_style="cyan")
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="green", justify="right")
            table.add_row("Total Chunks", str(stats["total_chunks"]))
            table.add_row("People Chunks", str(stats["people_chunks"]))
            table.add_row("Places Chunks", str(stats["places_chunks"]))
            console.print(table)
            console.print()
            continue

        # Process the query
        console.print()
        console.print("[bold cyan]Assistant > [/bold cyan]", end="")

        try:
            response, sources, query_type = generate_response(query, stream=True)
            last_sources = sources
            last_query_type = query_type
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]")
            console.print("[dim]Make sure Ollama is running and llama3.2 is pulled.[/dim]")

        console.print()


def cmd_clear():
    """Clear the ChromaDB database and reset the system."""
    from database import clear_database

    print_banner()
    console.print("[bold yellow]WARNING: This will delete ALL data from the vector database.[/bold yellow]")

    try:
        confirm = console.input("[bold]Are you sure? (yes/no): [/bold]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return

    if confirm in ("yes", "y"):
        clear_database()
    else:
        console.print("[dim]Cancelled.[/dim]")


def main():
    """Parse CLI arguments and dispatch to the appropriate command."""
    if len(sys.argv) < 2:
        print_banner()
        console.print("\n[bold]Usage:[/bold]")
        console.print("  python main.py [bold cyan]ingest[/bold cyan]  -- Scrape Wikipedia & build database")
        console.print("  python main.py [bold cyan]chat[/bold cyan]    -- Start interactive Q&A")
        console.print("  python main.py [bold cyan]clear[/bold cyan]   -- Delete database & reset")
        console.print()
        return

    command = sys.argv[1].lower()

    if command == "ingest":
        cmd_ingest()
    elif command == "chat":
        cmd_chat()
    elif command == "clear":
        cmd_clear()
    else:
        console.print(f"[bold red]Unknown command: '{command}'[/bold red]")
        console.print("  Available commands: ingest, chat, clear")


if __name__ == "__main__":
    main()
