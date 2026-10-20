"""Evaluation script for BookRAG: Evaluates Hit-Rate@5 on 20 QA benchmark pairs."""

import json
import argparse
import logging
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from retrieval.engine import RetrievalEngine

console = Console()

# 20 benchmark questions paired with their ground-truth page numbers
EVAL_PAIRS = [
    # Page 1: Chapter 1 - Foundations, Qubits, Superposition, Entanglement, Code, Image
    {"question": "What is a qubit?", "expected_page": 1, "topic": "Qubit definition"},
    {"question": "How is a qubit defined compared to a classical bit?", "expected_page": 1, "topic": "Qubit vs classical bit"},
    {"question": "What is quantum superposition?", "expected_page": 1, "topic": "Superposition concept"},
    {"question": "What is quantum entanglement defined as?", "expected_page": 1, "topic": "Entanglement definition"},
    {"question": "What python numpy function implements a Hadamard gate?", "expected_page": 1, "topic": "Hadamard gate code"},
    {"question": "Show me the code snippet for hadamard_gate() with numpy array.", "expected_page": 1, "topic": "Code implementation"},
    {"question": "What are the key properties of quantum bits?", "expected_page": 1, "topic": "Properties list"},
    {"question": "Does quantum entanglement link particles regardless of distance?", "expected_page": 1, "topic": "Entanglement distance"},
    {"question": "Where is the Bloch Sphere representation shown?", "expected_page": 1, "topic": "Bloch sphere figure"},
    {"question": "What image illustrates the Bloch sphere on the page?", "expected_page": 1, "topic": "Bloch sphere image caption"},
    {"question": "What is the fundamental unit of quantum information?", "expected_page": 1, "topic": "Fundamental unit"},

    # Page 2: Chapter 2 - Algorithms, Complexity Comparison Table, RSA Impact
    {"question": "What is the classical complexity of Shor's algorithm?", "expected_page": 2, "topic": "Shor classical complexity"},
    {"question": "What is the quantum polynomial complexity of Shor's algorithm?", "expected_page": 2, "topic": "Shor quantum complexity"},
    {"question": "What is Grover's search complexity in quantum vs classical?", "expected_page": 2, "topic": "Grover search complexity"},
    {"question": "Which algorithm achieves O(1) quantum complexity in the comparison table?", "expected_page": 2, "topic": "Deutsch-Jozsa complexity"},
    {"question": "Show the table comparing quantum vs classical algorithm complexity.", "expected_page": 2, "topic": "Complexity table"},
    {"question": "How does Shor's algorithm challenge modern RSA cryptographic infrastructure?", "expected_page": 2, "topic": "RSA cryptography threat"},
    {"question": "What is the classical complexity of Deutsch-Jozsa?", "expected_page": 2, "topic": "Deutsch-Jozsa classical"},
    {"question": "What quantum algorithms deliver quadratic to exponential speedups?", "expected_page": 2, "topic": "Algorithm speedups"},
    {"question": "What performance comparison table is in Section 2.1?", "expected_page": 2, "topic": "Section 2.1 table"}
]


def evaluate_retrieval(top_k: int = 5, verbose: bool = True) -> Dict[str, Any]:
    """Run retrieval evaluation on 20 QA benchmark pairs and report hit-rate@5."""
    engine = RetrievalEngine()
    total_pairs = len(EVAL_PAIRS)
    hits_count = 0
    results_breakdown: List[Dict[str, Any]] = []

    for idx, item in enumerate(EVAL_PAIRS, 1):
        q = item["question"]
        expected_page = item["expected_page"]

        res = engine.retrieve(query=q, top_k=top_k)

        # Extract pages returned in top-k
        retrieved_pages = [int(h.get("metadata", {}).get("page", 0)) for h in res.hits]
        is_hit = expected_page in retrieved_pages
        if is_hit:
            hits_count += 1

        top_score = res.hits[0].get("rerank_score", 0.0) if res.hits else 0.0

        results_breakdown.append({
            "idx": idx,
            "question": q,
            "expected_page": expected_page,
            "retrieved_pages": retrieved_pages,
            "is_hit": is_hit,
            "top_rerank_score": top_score,
            "topic": item["topic"]
        })

    hit_rate = (hits_count / total_pairs) * 100.0

    if verbose:
        console.print("\n" + "=" * 75)
        console.print("[bold cyan]BookRAG Retrieval Evaluation Benchmark (Hit-Rate@5)[/bold cyan]")
        console.print("=" * 75)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", justify="center", width=4)
        table.add_column("Question", style="dim", width=34)
        table.add_column("Expected", justify="center", width=8)
        table.add_column("Retrieved Pages", justify="center", width=16)
        table.add_column("Result", justify="center", width=8)
        table.add_column("Top Score", justify="right", width=10)

        for row in results_breakdown:
            status_text = "[bold green]HIT[/bold green]" if row["is_hit"] else "[bold red]MISS[/bold red]"
            pages_str = ", ".join(map(str, row["retrieved_pages"][:3]))
            table.add_row(
                str(row["idx"]),
                row["question"],
                f"Page {row['expected_page']}",
                f"[{pages_str}]",
                status_text,
                f"{row['top_rerank_score']:.4f}"
            )

        console.print(table)

        summary_panel = (
            f"[bold]Total Benchmark Questions:[/bold] {total_pairs}\n"
            f"[bold]Successful Hits (in Top-5):[/bold] [green]{hits_count}[/green]\n"
            f"[bold]Misses:[/bold] [red]{total_pairs - hits_count}[/red]\n"
            f"[bold cyan]Hit-Rate@5 Metric:[/bold cyan] [bold green]{hit_rate:.2f}%[/bold green]"
        )
        console.print(Panel(summary_panel, title="Final Evaluation Metrics", border_style="green"))

    return {
        "total_pairs": total_pairs,
        "hits_count": hits_count,
        "hit_rate_at_5": hit_rate,
        "results": results_breakdown
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BookRAG retrieval hit-rate@5.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks to consider (default: 5).")
    parser.add_argument("--json", action="store_true", help="Output JSON results to stdout.")
    args = parser.parse_args()

    results = evaluate_retrieval(top_k=args.top_k, verbose=not args.json)

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
