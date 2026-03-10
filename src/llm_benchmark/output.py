"""Output handlers: CSV writing and summary printing."""

import csv
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .benchmark import BenchmarkResult

console = Console()

CSV_FIELDNAMES = [
    "timestamp",
    "model",
    "machine",
    "run",
    "prompt_tokens",
    "completion_tokens",
    "total_time_seconds",
    "tokens_per_second",
    "time_to_first_token",
]


def save_to_csv(results: list[BenchmarkResult], csv_path: Path) -> None:
    """Append benchmark results to a CSV file.

    Args:
        results: List of BenchmarkResult objects to write.
        csv_path: Path to the output CSV file.
    """
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows([r.to_dict() for r in results])

    console.print(f"\n[bold green]✓ Results appended to[/bold green] [cyan]{csv_path}[/cyan]")


def print_summary(results: list[BenchmarkResult], model: str) -> None:
    """Print summary statistics for a completed benchmark run.

    Args:
        results: List of BenchmarkResult objects.
        model: Model name for display.
    """
    if not results:
        console.print("[yellow]No results to summarize.[/yellow]")
        return

    # Calculate basic statistics
    avg_tokens = sum(r.completion_tokens for r in results) / len(results)
    avg_time = sum(r.total_time_seconds for r in results) / len(results)
    
    # Calculate tokens/sec statistics
    tps_values = [r.tokens_per_second for r in results]
    avg_tps = statistics.mean(tps_values)
    min_tps = min(tps_values)
    max_tps = max(tps_values)
    
    # Calculate standard deviation only if we have multiple runs
    std_tps = statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0

    # Create a rich table for the summary
    table = Table(title="[bold cyan]BENCHMARK SUMMARY[/bold cyan]", show_header=False, box=None)
    table.add_column(style="cyan", width=25)
    table.add_column(style="green")

    table.add_row("Model:", model)
    table.add_row("Runs:", str(len(results)))
    table.add_row("", "")  # Blank row for spacing
    table.add_row("Avg tokens generated:", f"{avg_tokens:.0f}")
    table.add_row("Avg time:", f"{avg_time:.2f}s")
    table.add_row("", "")  # Blank row for spacing
    table.add_row("Avg tokens/sec:", f"[bold]{avg_tps:.2f}[/bold]")
    table.add_row("Min tokens/sec:", f"{min_tps:.2f}")
    table.add_row("Max tokens/sec:", f"{max_tps:.2f}")
    
    if len(results) > 1:
        table.add_row("Std deviation:", f"{std_tps:.2f}")

    console.print("\n" + "─" * 50)
    console.print(table)
    console.print("─" * 50)
