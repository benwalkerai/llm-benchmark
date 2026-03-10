"""Tests for the output module."""

import csv

import pytest

from llm_benchmark.benchmark import BenchmarkResult
from llm_benchmark.output import CSV_FIELDNAMES, print_summary, save_to_csv


@pytest.fixture
def sample_results():
    """Return a list of sample BenchmarkResult objects."""
    return [
        BenchmarkResult(
            timestamp="2025-01-01T00:00:00",
            model="llama3",
            machine="test-machine",
            run=i,
            prompt_tokens=50,
            completion_tokens=100,
            total_time_seconds=2.5,
            tokens_per_second=40.0,
        )
        for i in range(1, 4)
    ]


class TestSaveToCsv:
    def test_creates_csv_with_headers(self, tmp_path, sample_results):
        csv_path = tmp_path / "results.csv"
        save_to_csv(sample_results, csv_path)
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_FIELDNAMES

    def test_writes_correct_row_count(self, tmp_path, sample_results):
        csv_path = tmp_path / "results.csv"
        save_to_csv(sample_results, csv_path)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_appends_to_existing_file(self, tmp_path, sample_results):
        csv_path = tmp_path / "results.csv"
        save_to_csv(sample_results, csv_path)
        save_to_csv(sample_results, csv_path)  # append second batch
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6

    def test_no_duplicate_headers_on_append(self, tmp_path, sample_results):
        csv_path = tmp_path / "results.csv"
        save_to_csv(sample_results, csv_path)
        save_to_csv(sample_results, csv_path)
        with open(csv_path) as f:
            content = f.read()
        assert content.count("timestamp") == 1  # header appears once


class TestPrintSummary:
    def test_empty_results_prints_message(self, capsys):
        print_summary([], model="llama3")
        captured = capsys.readouterr()
        assert "No results" in captured.out

    def test_summary_contains_model_name(self, capsys, sample_results):
        print_summary(sample_results, model="llama3")
        captured = capsys.readouterr()
        assert "llama3" in captured.out

    def test_summary_shows_run_count(self, capsys, sample_results):
        print_summary(sample_results, model="llama3")
        captured = capsys.readouterr()
        assert "3" in captured.out
