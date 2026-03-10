"""Output handlers: CSV writing and summary printing."""

import csv
from pathlib import Path

from .benchmark import BenchmarkResult

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

    print(f"\nResults appended to {csv_path}")


def print_summary(results: list[BenchmarkResult], model: str) -> None:
    """Print summary statistics for a completed benchmark run.

    Args:
        results: List of BenchmarkResult objects.
        model: Model name for display.
    """
    if not results:
        print("No results to summarize.")
        return

    avg_tokens = sum(r.completion_tokens for r in results) / len(results)
    avg_time = sum(r.total_time_seconds for r in results) / len(results)
    avg_tps = sum(r.tokens_per_second for r in results) / len(results)

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Model: {model}")
    print(f"Runs: {len(results)}")
    print(f"Avg tokens generated: {avg_tokens:.0f}")
    print(f"Avg time: {avg_time:.2f}s")
    print(f"Avg tokens/sec: {avg_tps:.2f}")
    print("=" * 50)
