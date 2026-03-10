# GitHub Copilot Instructions for LLM-Benchmark

## Project Overview
This is a CLI tool for benchmarking LLM inference endpoints (Ollama, OpenAI-compatible APIs).
It measures tokens/second, latency, and outputs results to CSV.

## Code Style & Conventions
- Python 3.12+, using `uv` for package management
- Follow PEP 8; use type hints on all function signatures
- Prefer `click` for CLI argument handling (replacing raw `argparse`)
- Use `pathlib.Path` over `os.path` wherever possible
- All public functions must have docstrings
- Error handling: catch specific exceptions, not bare `except Exception`

## Architecture
- `src/llm_benchmark/benchmark.py` — core benchmarking logic (pure functions, no I/O side effects)
- `src/llm_benchmark/cli.py` — Click CLI entry point, wires together benchmark + output
- `src/llm_benchmark/output.py` — CSV writing and summary printing
- `tests/` — pytest unit and integration tests

## Testing
- Use `pytest` with `pytest-mock` for mocking OpenAI API calls
- Every public function in `benchmark.py` and `output.py` should have at least one unit test
- Use `tmp_path` fixture for any file I/O tests
- Mock `OpenAI` client; never make real API calls in tests
- Target ≥80% code coverage

## Copilot Suggestions
- When adding new CLI flags, register them in `src/llm_benchmark/cli.py` using Click decorators
- When adding new output formats (JSON, Markdown), add a handler in `output.py`
- Suggest streaming support (`stream=True`) as an improvement for TTFT measurement
- Flag any hardcoded values that should be configurable via env or CLI args
- Suggest `rich` library for improved terminal output formatting
