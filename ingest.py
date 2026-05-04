"""
ingest.py - Wikipedia Data Ingestion Module

Scrapes Wikipedia pages for 40 predefined entities (20 people, 20 places),
cleans the text, and splits it into appropriately sized chunks for embedding.
"""

import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console(force_terminal=True)

# ─── Entity Definitions ───────────────────────────────────────────────────────

PEOPLE = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    "Isaac Newton",
    "Galileo Galilei",
    "Charles Darwin",
    "Winston Churchill",
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Martin Luther King Jr.",
    "Serena Williams",
    "Michael Jordan",
    "Muhammad Ali",
]

PLACES = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    "Stonehenge",
    "Acropolis of Athens",
    "Petra",
    "Chichen Itza",
    "Angkor Wat",
    "Mount Fuji",
    "Niagara Falls",
    "Victoria Falls",
    "Sydney Opera House",
    "Burj Khalifa",
]

# ─── Text Splitter Configuration ──────────────────────────────────────────────

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", ", ", " ", ""],
)


import time

def fetch_wikipedia_page(entity_name: str, retries: int = 3, delay: int = 2) -> str | None:
    """
    Fetch the full text content of a Wikipedia page for a given entity.

    Uses the wikipedia library's search + page methods with auto_suggest
    disabled to get precise matches. Falls back to the first search result
    if the exact title doesn't match. Includes a retry loop for intermittent
    API failures.

    Args:
        entity_name: The name of the person or place to search for.
        retries: Number of times to retry on failure.
        delay: Seconds to wait between retries.

    Returns:
        The full text content of the Wikipedia page, or None if not found.
    """
    for attempt in range(retries):
        try:
            page = wikipedia.page(entity_name, auto_suggest=False)
            return page.content
        except wikipedia.exceptions.DisambiguationError as e:
            # Pick the first option from disambiguation
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return page.content
            except Exception:
                console.print(f"  [red]X[/red] Could not resolve disambiguation for '{entity_name}'")
                return None
        except wikipedia.exceptions.PageError:
            # Try searching instead
            try:
                results = wikipedia.search(entity_name)
                if results:
                    page = wikipedia.page(results[0], auto_suggest=False)
                    return page.content
            except Exception:
                pass
            console.print(f"  [red]X[/red] Page not found for '{entity_name}'")
            return None
        except Exception as e:
            if attempt < retries - 1:
                console.print(f"  [yellow]![/yellow] API error for '{entity_name}', retrying in {delay}s... ({e})")
                time.sleep(delay)
            else:
                console.print(f"  [red]X[/red] Error fetching '{entity_name}' after {retries} attempts: {e}")
                return None


def clean_text(text: str) -> str:
    """
    Clean raw Wikipedia text by removing reference markers, excessive
    whitespace, and section markers that don't carry useful content.

    Args:
        text: Raw Wikipedia page content.

    Returns:
        Cleaned text ready for chunking.
    """
    import re

    # Remove reference markers like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)
    # Remove "== See also ==" and everything after common trailing sections
    for section in ["== See also ==", "== References ==", "== External links ==", "== Further reading =="]:
        idx = text.find(section)
        if idx != -1:
            text = text[:idx]
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def chunk_text(text: str, entity_name: str, entity_type: str) -> list[dict]:
    """
    Split text into chunks with metadata attached.

    Args:
        text: The cleaned text to split.
        entity_name: Name of the entity (used for metadata).
        entity_type: Either 'person' or 'place' (used for metadata filtering).

    Returns:
        A list of dicts with keys: 'text', 'entity', 'type', 'chunk_index'.
    """
    chunks = text_splitter.split_text(text)
    return [
        {
            "text": chunk,
            "entity": entity_name,
            "type": entity_type,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]


def ingest_all() -> list[dict]:
    """
    Main ingestion pipeline: fetches, cleans, and chunks Wikipedia pages
    for all 40 entities defined in PEOPLE and PLACES.

    Returns:
        A list of all chunk dicts ready for embedding and storage.
    """
    all_chunks = []
    total_entities = len(PEOPLE) + len(PLACES)

    console.print("\n[bold cyan]Starting Wikipedia Ingestion[/bold cyan]")
    console.print(f"   Entities to process: [bold]{total_entities}[/bold]")
    console.print(f"   Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Ingesting...", total=total_entities)

        # Process people
        for name in PEOPLE:
            progress.update(task, description=f"[cyan]Fetching: {name}")
            content = fetch_wikipedia_page(name)
            if content:
                cleaned = clean_text(content)
                chunks = chunk_text(cleaned, name, "person")
                all_chunks.extend(chunks)
                console.print(f"  [green]+[/green] {name} -- {len(chunks)} chunks")
            progress.advance(task)

        # Process places
        for name in PLACES:
            progress.update(task, description=f"[cyan]Fetching: {name}")
            content = fetch_wikipedia_page(name)
            if content:
                cleaned = clean_text(content)
                chunks = chunk_text(cleaned, name, "place")
                all_chunks.extend(chunks)
                console.print(f"  [green]+[/green] {name} -- {len(chunks)} chunks")
            progress.advance(task)

    console.print(f"\n[bold green]Ingestion complete![/bold green] Total chunks: [bold]{len(all_chunks)}[/bold]\n")
    return all_chunks


if __name__ == "__main__":
    chunks = ingest_all()
    print(f"Sample chunk: {chunks[0] if chunks else 'No chunks generated'}")
