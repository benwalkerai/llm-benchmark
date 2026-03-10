"""Core benchmarking logic for LLM inference endpoints."""

import time
import threading
import itertools
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from openai import OpenAI

# Standard benchmark prompt
DEFAULT_PROMPT = """
Write a detailed explanation of how neural networks learn through backpropagation.
Include the mathematical concepts and provide examples.
"""


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""

    timestamp: str
    model: str
    machine: str
    run: int
    prompt_tokens: int
    completion_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    time_to_first_token: str = "N/A"

    def to_dict(self) -> dict:
        """Convert result to dictionary for CSV writing."""
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "machine": self.machine,
            "run": self.run,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_time_seconds": self.total_time_seconds,
            "tokens_per_second": self.tokens_per_second,
            "time_to_first_token": self.time_to_first_token,
        }


def spinner(stop_event: threading.Event) -> None:
    """Display a spinner animation while benchmarking."""
    spinner_chars = itertools.cycle(["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"])
    while not stop_event.is_set():
        sys.stdout.write(f"\r {next(spinner_chars)} Processing...")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r")
    sys.stdout.flush()


def run_single(
    client: OpenAI,
    model: str,
    machine: str,
    run_number: int,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> Optional[BenchmarkResult]:
    """Execute a single benchmark run and return a BenchmarkResult.

    Args:
        client: Configured OpenAI client.
        model: Model name to benchmark.
        machine: Machine identifier for the result.
        run_number: Current run index (1-based).
        prompt: The prompt to send to the model.
        max_tokens: Maximum tokens for the completion.
        temperature: Sampling temperature.

    Returns:
        BenchmarkResult on success, None on failure.
    """
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(stop_spinner,))
    spinner_thread.start()

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        end_time = time.time()
    except Exception as e:
        stop_spinner.set()
        spinner_thread.join()
        raise RuntimeError(f"API call failed on run {run_number}: {e}") from e
    finally:
        stop_spinner.set()
        spinner_thread.join()

    total_time = end_time - start_time
    tokens_generated = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    tokens_per_second = tokens_generated / total_time if total_time > 0 else 0.0

    return BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        model=model,
        machine=machine,
        run=run_number,
        prompt_tokens=prompt_tokens,
        completion_tokens=tokens_generated,
        total_time_seconds=round(total_time, 2),
        tokens_per_second=round(tokens_per_second, 2),
    )


def run_benchmark(
    api_url: str,
    api_key: str,
    model: str,
    machine: str,
    num_runs: int = 3,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 500,
) -> list[BenchmarkResult]:
    """Run the full benchmark suite and return all results.

    Args:
        api_url: Base URL for the OpenAI-compatible API.
        api_key: API key (can be a dummy value for local endpoints).
        model: Model name to benchmark.
        machine: Machine identifier.
        num_runs: Number of benchmark iterations.
        prompt: Prompt to use for benchmarking.
        max_tokens: Maximum completion tokens per run.

    Returns:
        List of BenchmarkResult objects for successful runs.
    """
    client = OpenAI(base_url=api_url, api_key=api_key)
    results = []

    for run in range(1, num_runs + 1):
        print(f"\nRun {run}/{num_runs}...")
        try:
            result = run_single(
                client=client,
                model=model,
                machine=machine,
                run_number=run,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            results.append(result)
            print(f" Tokens generated: {result.completion_tokens}")
            print(f" Time: {result.total_time_seconds:.2f}s")
            print(f" Speed: {result.tokens_per_second:.2f} tokens/sec")
        except RuntimeError as e:
            print(f" Error: {e}")
            continue

    return results
