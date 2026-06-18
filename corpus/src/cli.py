"""
cli.py

Command-line interface for the UAE/MENA AI policy search tool.

Usage:
    python cli.py "your query here"
    python cli.py "your query here" --top-k 3
    python cli.py --rebuild   # force re-embedding of the corpus
    python cli.py             # interactive mode
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Fix UnicodeEncodeError on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from search_engine import PolicySearchEngine

CORPUS_DIR = Path(__file__).parent

console = Console()


def print_results(results: list[dict], query: str, elapsed: float) -> None:
    console.print()
    console.print(f"[bold]{len(results)} results[/bold] for [italic]\"{query}\"[/italic]  ({elapsed*1000:.0f} ms)")
    console.print()

    for rank, r in enumerate(results, start=1):
        score_bar = "█" * int(r["score"] * 20)
        header = Text()
        header.append(f"{rank}. ", style="bold")
        header.append(r["doc_title"], style="bold cyan")
        header.append(f"  ·  {r['section']}", style="dim")

        body = Text(r["text"][:420] + ("..." if len(r["text"]) > 420 else ""))

        footer = Text(f"\nscore {r['score']:.3f}  {score_bar}", style="green")

        console.print(Panel(Text.assemble(header, "\n\n", body, footer), expand=True, border_style="grey50"))


def run_query(engine: PolicySearchEngine, query: str, top_k: int) -> None:
    start = time.perf_counter()
    results = engine.search(query, top_k=top_k)
    elapsed = time.perf_counter() - start
    print_results(results, query, elapsed)


def interactive_loop(engine: PolicySearchEngine, top_k: int) -> None:
    console.print(Panel(
        "[bold]UAE / MENA AI Policy Search[/bold]\n"
        "Semantic search over a small corpus of original summaries covering UAE and Saudi AI policy.\n"
        "Type a question or topic. Type 'quit' or 'exit' to leave.",
        border_style="blue",
    ))

    while True:
        try:
            query = console.input("\n[bold cyan]query> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            console.print("Exiting.")
            break

        run_query(engine, query, top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a small corpus of UAE/MENA AI policy summaries.")
    parser.add_argument("query", nargs="?", default=None, help="Search query. Omit to enter interactive mode.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the embedding index")
    args = parser.parse_args()

    console.print("[dim]Building TF-IDF index...[/dim]")
    engine = PolicySearchEngine(corpus_dir=CORPUS_DIR)
    engine.build_index(force_rebuild=args.rebuild)
    console.print(f"[dim]Indexed {len(engine.chunks)} chunks from {len(set(c.doc_id for c in engine.chunks))} documents.[/dim]")

    if args.query:
        run_query(engine, args.query, args.top_k)
    else:
        interactive_loop(engine, args.top_k)


if __name__ == "__main__":
    main()
