"""CLI entry point for llm-benchmark."""

import os
import socket
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from .benchmark import DEFAULT_PROMPT, run_benchmark
from .output import print_summary, save_to_csv

console = Console()

load_dotenv()


@click.command()
@click.option(
    "--runs",
    default=3,
    show_default=True,
    help="Number of benchmark iterations.",
)
@click.option(
    "--model",
    default=lambda: os.getenv("MODEL_NAME", ""),
    show_default="MODEL_NAME env var",
    help="Model name to benchmark.",
)
@click.option(
    "--url",
    default=lambda: os.getenv("API_URL", ""),
    show_default="API_URL env var",
    help="OpenAI-compatible API base URL.",
)
@click.option(
    "--apikey",
    default=lambda: os.getenv("API_KEY", "ollama"),
    show_default="API_KEY env var",
    help="API key (use any value for local endpoints).",
)
@click.option(
    "--machine",
    default=lambda: socket.gethostname(),
    show_default="system hostname",
    help="Machine name label to include in results.",
)
@click.option(
    "--output",
    default="benchmark_results.csv",
    show_default=True,
    type=click.Path(),
    help="Output CSV file path.",
)
@click.option(
    "--max-tokens",
    default=500,
    show_default=True,
    help="Maximum completion tokens per run.",
)
@click.option(
    "--temperature",
    default=0.7,
    show_default=True,
    type=float,
    help="Sampling temperature (0.0 to 2.0).",
)
@click.version_option()
def main(
    runs: int,
    model: str,
    url: str,
    apikey: str,
    machine: str,
    output: str,
    max_tokens: int,
    temperature: float,
) -> None:
    """Benchmark an LLM inference endpoint and record tokens/sec performance."""
    if not url:
        raise click.UsageError("API URL is required. Set --url or API_URL in .env")
    if not model:
        raise click.UsageError("Model name is required. Set --model or MODEL_NAME in .env")

    if runs <= 0:
        raise click.UsageError("--runs must be a positive integer (> 0)")
    if max_tokens <= 0:
        raise click.UsageError("--max-tokens must be a positive integer (> 0)")
    if not 0.0 <= temperature <= 2.0:
        raise click.UsageError("--temperature must be between 0.0 and 2.0")

    # Display configuration in a styled panel
    config_text = (
        f"[cyan]API URL:[/cyan]  {url}\n"
        f"[cyan]Model:[/cyan]      {model}\n"
        f"[cyan]Machine:[/cyan]    {machine}"
    )
    console.print(
        Panel(
            config_text,
            title="[bold cyan]LLM Benchmark[/bold cyan]",
            border_style="cyan",
        )
    )

    results = run_benchmark(
        api_url=url,
        api_key=apikey,
        model=model,
        machine=machine,
        num_runs=runs,
        prompt=DEFAULT_PROMPT,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if results:
        save_to_csv(results, Path(output))
        print_summary(results, model=model)
    else:
        console.print("[bold red]✗ No successful runs completed.[/bold red]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
