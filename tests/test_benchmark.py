"""Tests for the core benchmarking module."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from llm_benchmark.benchmark import BenchmarkResult, run_benchmark, run_single


@pytest.fixture
def mock_client():
    """Return a mocked OpenAI client."""
    client = MagicMock()
    response = MagicMock()
    response.usage.completion_tokens = 120
    response.usage.prompt_tokens = 50
    client.chat.completions.create.return_value = response
    return client


class TestBenchmarkResult:
    def test_to_dict_has_all_keys(self):
        result = BenchmarkResult(
            timestamp="2025-01-01T00:00:00",
            model="llama3",
            machine="test-machine",
            run=1,
            prompt_tokens=50,
            completion_tokens=100,
            total_time_seconds=2.5,
            tokens_per_second=40.0,
        )
        d = result.to_dict()
        expected_keys = [
            "timestamp", "model", "machine", "run", "prompt_tokens",
            "completion_tokens", "total_time_seconds", "tokens_per_second", "time_to_first_token"
        ]
        assert list(d.keys()) == expected_keys

    def test_default_ttft_is_na(self):
        result = BenchmarkResult(
            timestamp="2025-01-01T00:00:00",
            model="llama3",
            machine="test",
            run=1,
            prompt_tokens=10,
            completion_tokens=50,
            total_time_seconds=1.0,
            tokens_per_second=50.0,
        )
        assert result.time_to_first_token == "N/A"


class TestRunSingle:
    def test_successful_run(self, mock_client):
        result = run_single(
            client=mock_client,
            model="llama3",
            machine="test-machine",
            run_number=1,
            prompt="test prompt",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.model == "llama3"
        assert result.completion_tokens == 120
        assert result.prompt_tokens == 50
        assert result.tokens_per_second > 0

    def test_api_failure_raises_runtime_error(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")
        with pytest.raises(RuntimeError, match="API call failed"):
            run_single(
                client=mock_client,
                model="llama3",
                machine="test-machine",
                run_number=1,
                prompt="test prompt",
            )


class TestRunBenchmark:
    @patch("llm_benchmark.benchmark.OpenAI")
    def test_returns_results_list(self, mock_openai):
        mock_client = MagicMock()
        response = MagicMock()
        response.usage.completion_tokens = 100
        response.usage.prompt_tokens = 40
        mock_client.chat.completions.create.return_value = response
        mock_openai.return_value = mock_client

        results = run_benchmark(
            api_url="http://localhost:11434",
            api_key="ollama",
            model="llama3",
            machine="test",
            num_runs=2,
        )
        assert len(results) == 2
        assert all(isinstance(r, BenchmarkResult) for r in results)

    @patch("llm_benchmark.benchmark.OpenAI")
    def test_skips_failed_runs(self, mock_openai):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        mock_openai.return_value = mock_client

        results = run_benchmark(
            api_url="http://localhost:11434",
            api_key="ollama",
            model="llama3",
            machine="test",
            num_runs=3,
        )
        assert results == []
