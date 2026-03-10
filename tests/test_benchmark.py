"""Tests for the core benchmarking module."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from llm_benchmark.benchmark import BenchmarkResult, run_benchmark, run_single


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
    def test_successful_run(self):
        # Create a fresh mock for this test
        client = MagicMock()
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello world test "
        chunk1.usage = None
        
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = None  
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 50
        
        client.chat.completions.create.return_value = iter([chunk1, chunk2])
        
        with patch("llm_benchmark.benchmark.time.time", side_effect=[0.0, 0.1, 2.5]):
            result = run_single(
                client=client,
                model="llama3",
                machine="test-machine",
                run_number=1,
                prompt="test prompt",
            )
            assert isinstance(result, BenchmarkResult)
            assert result.model == "llama3"
            assert result.completion_tokens > 0
            assert result.tokens_per_second > 0
            # Check TTFT was captured
            assert result.time_to_first_token != "N/A"

    def test_api_failure_raises_runtime_error(self):
        mock_client = MagicMock()
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
        
        # Create streaming mock response
        def create_stream():
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "Test response content "
            chunk1.usage = None
            
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = None
            chunk2.usage = MagicMock()
            chunk2.usage.prompt_tokens = 40
            
            return iter([chunk1, chunk2])
        
        mock_client.chat.completions.create.side_effect = [create_stream(), create_stream()]
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
