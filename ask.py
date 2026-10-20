"""CLI query interface for BookRAG."""

import sys
import argparse
import logging
from typing import List, Dict

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from ingest.validator import sanitize_for_console
from generation.qa_chain import BookRAGPipeline

console = Console(highlight=False, soft_wrap=True)


def print_response(response, show_chunks: bool = False):
    """Pretty-print the generated answer, citations, and optional chunks."""
    try:
        console.print()
        console.print(Panel(Markdown(sanitize_for_console(response.answer)), title="[bold green]BookRAG Tutor Answer[/bold green]", border_style="green"))

        if response.sources:
            src_table = Table(title="Sources Cited", show_header=True, header_style="bold cyan")
            src_table.add_column("Chapter", style="white")
            src_table.add_column("Page", justify="center", style="yellow")
            src_table.add_column("Type", justify="center", style="magenta")
            for src in response.sources:
                src_table.add_row(sanitize_for_console(src.chapter), str(src.page), src.type)
            console.print(src_table)

        if show_chunks and response.retrieved_chunks:
            console.print("\n[bold yellow]Retrieved Context Chunks (Top 5):[/bold yellow]")
            for i, chunk in enumerate(response.retrieved_chunks, 1):
                score = chunk.get("rerank_score", chunk.get("score", 0.0))
                heading = sanitize_for_console(chunk.get("metadata", {}).get("heading_path", ""))
                content_preview = sanitize_for_console(chunk.get("content", ""))
                console.print(
                    Panel(
                        content_preview,
                        title=f"Chunk {i} [Rerank Score: {score:.4f}] - {heading}",
                        border_style="dim"
                    )
                )
    except Exception as e:
        print(f"\nBookRAG Tutor Answer:\n{response.answer}\n")


def interactive_mode(pipeline: BookRAGPipeline, show_chunks: bool):
    """Interactive conversational chat session."""
    console.print(Panel.fit(
        "[bold cyan]Welcome to BookRAG Interactive Tutor CLI![/bold cyan]\n"
        "Ask questions about the ingested book. Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.",
        border_style="cyan"
    ))

    chat_history: List[Dict[str, str]] = []

    while True:
        try:
            query = console.input("\n[bold yellow]You > [/bold yellow]").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            with console.status("[cyan]Searching and formulating answer...[/cyan]"):
                response = pipeline.query(query, chat_history=chat_history)

            print_response(response, show_chunks=show_chunks)

            # Update history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response.answer})

        except KeyboardInterrupt:
            console.print("\n[dim]Session closed.[/dim]")
            break


def main():
    parser = argparse.ArgumentParser(description="Ask questions about the ingested book.")
    parser.add_argument("question", nargs="?", type=str, help="Single question to ask.")
    parser.add_argument("--chunks", action="store_true", help="Display retrieved context chunks.")
    parser.add_argument("--filter-type", type=str, help="Filter by content type (e.g. table, text, image).")
    parser.add_argument("--filter-chapter", type=str, help="Filter by chapter name.")
    args = parser.parse_args()

    pipeline = BookRAGPipeline()

    filters = {}
    if args.filter_type:
        filters["type"] = args.filter_type
    if args.filter_chapter:
        filters["chapter"] = args.filter_chapter

    if args.question:
        with console.status("[cyan]Searching book and answering...[/cyan]"):
            response = pipeline.query(args.question, filters=filters if filters else None)
        print_response(response, show_chunks=args.chunks)
    else:
        interactive_mode(pipeline, show_chunks=args.chunks)


if __name__ == "__main__":
    main()
