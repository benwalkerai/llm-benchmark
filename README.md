# LLM Benchmark Tool

A simple Python tool for benchmarking local LLM performance by measuring tokens-per-second generation speed.

## Features

- 🚀 Compatible with any OpenAI-compatible API (Ollama, LM Studio, llama.cpp, vLLM, etc.)
- 📊 Tracks tokens/second, latency, time-to-first-token (TTFT), and token counts with statistical analysis
- 💾 Exports detailed results to CSV for comparison across models and configurations
- 🎯 Measures First-Token-To-User (TTFT) latency using streaming API
- 📈 Calculates min, max, and standard deviation across multiple runs
- ⚡ Real-time progress indicator with spinner animation
- 🌡️ Configurable temperature for generation behavior testing
- 🔧 Auto-detects machine name; fully configurable via CLI arguments or `.env` file

## Installation

### Option 1: Install from GitHub Release (Recommended)

```bash
# Latest release
pip install "https://github.com/benwalkerai/LLM-Benchmark/releases/download/v0.1.3/llm_benchmark-0.1.3-py3-none-any.whl"
```

### Option 2: Install from GitHub Source

```bash
# Latest main branch
pip install "git+https://github.com/benwalkerai/LLM-Benchmark.git"

# Specific release tag
pip install "git+https://github.com/benwalkerai/LLM-Benchmark.git@v0.1.3"
```

### Option 3: Local Development Setup

Clone and install in editable mode:
```bash
git clone https://github.com/benwalkerai/LLM-Benchmark.git
cd LLM-Benchmark
pip install -e .
```

### Create Configuration

Create a `.env` file in your working directory:
```env
API_URL=http://localhost:11434/v1
API_KEY=dummy-key
MODEL_NAME=llama3.2:latest
```

Required environment variables:
- `API_URL` - OpenAI-compatible endpoint (default: http://localhost:11434/v1)
- `API_KEY` - API authentication key
- `MODEL_NAME` - Model identifier to benchmark

## Usage

### Quick Start

After installation, run the benchmark with default settings:
```bash
llm-benchmark
```

Or run directly from this repo without installation:
```bash
python -m llm_benchmark.cli
```

### CLI Arguments

```bash
llm-benchmark [OPTIONS]
```

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--runs` | `-r` | Integer | 3 | Number of benchmark iterations |
| `--model` | `-m` | String | From .env | Model name to benchmark |
| `--url` | `-u` | String | From .env | OpenAI-compatible API endpoint |
| `--apikey` | `-k` | String | From .env | API authentication key |
| `--max-tokens` | | Integer | 256 | Max tokens to generate per run (1-2048) |
| `--temperature` | | Float | 1.0 | Sampling temperature (0.0-2.0) |
| `--machine` | | String | Auto-detect | Machine name tag for results |
| `--output` | `-o` | String | benchmark_results.csv | Output CSV filename |

### Example Commands

```bash
# Run 10 iterations for better statistics
llm-benchmark --runs 10

# Test different temperature settings for quality vs speed
llm-benchmark --temperature 0.1 --runs 5
llm-benchmark --temperature 1.5 --runs 5

# Limit token generation to 128 tokens
llm-benchmark --max-tokens 128 --runs 5

# Switch models and servers
llm-benchmark --model llama3.3:70b --url http://localhost:8080/v1

# Tag results with custom machine identifier
llm-benchmark --machine "RTX4090-Server" --runs 15

# Full example with all options
llm-benchmark \
  --runs 20 \
  --model llama3.3:70b \
  --url http://localhost:8080/v1 \
  --max-tokens 512 \
  --temperature 0.7 \
  --machine "GPU-Workstation" \
  --output llama3.3_results.csv
```

### Output Format

The tool displays real-time progress with a spinner, then prints a summary:

```
╭─────────────────────────────────────────────────────╮
│              Benchmark Configuration                │
│                                                     │
│ Model: llama3.2:latest                              │
│ URL: http://localhost:11434/v1                      │
│ Runs: 3                                             │
│ Max Tokens: 256                                     │
│ Temperature: 1.0                                    │
│ Machine: workstation-01                             │
╰─────────────────────────────────────────────────────╯

Run 1/3... ⠋
Run 2/3... ⠙
Run 3/3... ⠹

╭─────────────────────────────────────────────────────╮
│           BENCHMARK SUMMARY                         │
├─────────────────────────────────────────────────────┤
│ Model: llama3.2:latest                              │
│ Total Runs: 3                                       │
│ Avg Prompt Tokens: 42                               │
│ Avg Completion Tokens: 287                          │
│ Avg Total Time: 8.34s                               │
│                                                     │
│ Tokens/Second Statistics:                           │
│   Average: 34.45 tokens/sec                         │
│   Min: 32.11 tokens/sec                             │
│   Max: 36.78 tokens/sec                             │
│   Stdev: 1.89 tokens/sec                            │
│                                                     │
│ Time-to-First-Token (TTFT):                         │
│   Average: 145.2 ms                                 │
╰─────────────────────────────────────────────────────╯

Results appended to: benchmark_results.csv
```

### CSV Output Columns

When results are saved to CSV, each row contains:
- `timestamp` - ISO8601 timestamp when test ran
- `machine` - Machine identifier
- `model` - Model name tested
- `run` - Run sequence number
- `prompt_tokens` - Input tokens
- `completion_tokens` - Generated tokens
- `total_time_seconds` - Total generation time
- `tokens_per_second` - Throughput metric
- `time_to_first_token_ms` - TTFT latency in milliseconds

## Deployment Guide

### Ollama (Fastest Setup)

**1. Install and start Ollama:**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Windows: Download from https://ollama.ai/download
```

**2. Pull a model:**
```bash
ollama pull llama3.2
ollama pull neural-chat
```

**3. Configure `.env`:**
```env
API_URL=http://localhost:11434/v1
API_KEY=dummy-key
MODEL_NAME=llama3.2:latest
```

**4. Benchmark:**
```bash
llm-benchmark --runs 10
```

### llama.cpp Server

**1. Clone and build:**
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
```

**2. Start the server:**
```bash
./llama-server \
  --model your-model.gguf \
  --port 8080 \
  --api-key test \
  --n-gpu-layers 99  # Offload to GPU
```

**3. Configure `.env`:**
```env
API_URL=http://localhost:8080/v1
API_KEY=test
MODEL_NAME=your-model
```

**4. Benchmark:**
```bash
llm-benchmark --url http://localhost:8080/v1 --runs 10
```

### LM Studio

**1. Download from https://lmstudio.ai/**

**2. Load a model in the UI**

**3. Go to Local Server tab and start the server (port 1234)**

**4. Configure `.env`:**
```env
API_URL=http://localhost:1234/v1
API_KEY=lm-studio
MODEL_NAME=your-model-name
```

**5. Benchmark:**
```bash
llm-benchmark --url http://localhost:1234/v1 --runs 10
```

### vLLM (High-Performance Serving)

**1. Install:**
```bash
pip install vllm
```

**2. Start the server:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --dtype float16 \
  --tensor-parallel-size 1
```

**3. Configure `.env`:**
```env
API_URL=http://localhost:8000/v1
API_KEY=token-abc123
MODEL_NAME=meta-llama/Llama-2-7b-hf
```

**4. Benchmark with GPU offloading:**
```bash
llm-benchmark --url http://localhost:8000/v1 --runs 20 --max-tokens 512
```

## Comparing Models & Settings

## Comparing Models & Settings

### Benchmarking Multiple Models

Test different models on the same server:
```bash
#!/bin/bash
# benchmark_all_models.sh

MODELS=("llama3.2:1b" "llama3.2:3b" "llama3.3:70b" "neural-chat:latest")

for MODEL in "${MODELS[@]}"; do
  echo "Benchmarking $MODEL..."
  llm-benchmark --model "$MODEL" --runs 10 --output results_${MODEL}.csv
done

echo "All benchmarks complete!"
```

### Testing Temperature Impact on Speed

Compare how temperature affects throughput:
```bash
# Conservative (faster, more deterministic)
llm-benchmark --temperature 0.1 --runs 5 --output temp_0.1.csv

# Balanced
llm-benchmark --temperature 1.0 --runs 5 --output temp_1.0.csv

# Creative (slower, more varied)
llm-benchmark --temperature 2.0 --runs 5 --output temp_2.0.csv
```

### Token Length Analysis

Measure performance across different output lengths:
```bash
llm-benchmark --max-tokens 64 --runs 5 --output tokens_64.csv
llm-benchmark --max-tokens 256 --runs 5 --output tokens_256.csv
llm-benchmark --max-tokens 1024 --runs 5 --output tokens_1024.csv
```

### Analyzing Results

CSV files are easily analyzed with Python or spreadsheet tools:
```python
import pandas as pd

# Load all results
df = pd.read_csv('benchmark_results.csv')

# Group by model and show statistics
stats = df.groupby('model')['tokens_per_second'].agg(['mean', 'min', 'max', 'std'])
print(stats)

# Find fastest model
fastest = df.loc[df['tokens_per_second'].idxmax()]
print(f"Fastest: {fastest['model']} at {fastest['tokens_per_second']:.2f} tokens/sec")
```

## Troubleshooting

### Connection Issues

**Error: "Connection refused" or "Cannot connect"**
- Verify your LLM server is running
- Check the port number matches (e.g., 11434 for Ollama, 8080 for llama.cpp)
- Test with curl: `curl http://localhost:11434/v1/models`
- Check firewall rules if connecting to remote server

**Error: "Invalid API key"**
- Verify API_KEY matches your server configuration
- Some local servers (Ollama, llama.cpp) may use dummy keys

### Model Issues

**Error: "Model not found" or "Model does not exist"**

For Ollama:
```bash
ollama list          # See available models
ollama pull llama3.2 # Download a model
```

For llama.cpp:
- Ensure the `.gguf` file path is correct
- Check file permissions

For LM Studio:
- Verify model is loaded in the UI
- Check the exact model name displayed in the app

### Configuration Issues

**Error: "Missing environment variables"**
- Ensure `.env` file exists in your working directory
- Check variable names: `API_URL`, `API_KEY`, `MODEL_NAME` (uppercase)
- No extra spaces around `=` signs:
  ```env
  # ✅ Good
  API_URL=http://localhost:11434/v1
  
  # ❌ Bad (extra spaces)
  API_URL = http://localhost:11434/v1
  ```

**Error: "Invalid temperature" or "Invalid max-tokens"**
- Temperature must be between 0.0 and 2.0
- Max tokens must be between 1 and 2048
- Both must be positive numbers

### Performance Issues

**Slow benchmarks or timeouts**

1. Check server load: `top` (Linux/Mac) or Task Manager (Windows)
2. Reduce `--max-tokens` to get faster feedback
3. Reduce `--runs` for testing, increase later for accuracy
4. On GPU systems, ensure `--n-gpu-layers` is set to offload computation

**Very high TTFT (time-to-first-token)**
- Server may be overloaded or model not GPU-accelerated
- Check if model is loaded in VRAM: `nvidia-smi` or `watch nvidia-smi`
- Use smaller model for faster TTFT

## Development

### Local Setup

Repository structure:
```
llm-benchmark/
├── src/llm_benchmark/
│   ├── __init__.py
│   ├── cli.py          # Click CLI entry point
│   ├── benchmark.py    # Core benchmarking logic
│   └── output.py       # CSV export and display
├── tests/
│   ├── test_benchmark.py
│   └── test_output.py
├── pyproject.toml
└── README.md
```

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/benwalkerai/LLM-Benchmark.git
cd LLM-Benchmark

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src/llm_benchmark

# Run specific test file
pytest tests/test_benchmark.py -v

# Run with detailed output
pytest tests/ -vv --tb=short
```

### Code Quality

```bash
# Run linter (Ruff)
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Build distribution
uv build
```

### Architecture Notes

- **cli.py**: Click CLI interface, argument validation, configuration loading
- **benchmark.py**: Pure functions for running benchmarks with streaming API, measuring TTFT
- **output.py**: Rich formatting for terminal output, CSV export with statistics
- **Tests**: Mocked OpenAI API calls, pure function testing with pytest

Metrics calculated:
- **Tokens/Second**: `(prompt_tokens + completion_tokens) / total_time_seconds`
- **TTFT**: Time from API call to first token received (streaming)
- **Statistics**: Min, max, standard deviation of tokens/sec across runs

## Interpreting Results

### Key Metrics Explained

**Tokens/Second (Throughput)**
- Measure of how many tokens the model generates per second
- Higher is faster
- Affected by: model size, GPU capability, batch size, temperature

**Time-to-First-Token (TTFT)**
- Latency until first token arrives from the model
- Given in milliseconds
- Important for user experience (perceived responsiveness)
- Affected by: model loading, prompt length, server load

**Standard Deviation**
- Measure of consistency across runs
- Lower stdev = more consistent performance
- High stdev may indicate: server contention, thermal throttling, or variable load

### Example Analysis

```
Results for llama3.3:70b:
  Avg Tokens/Sec: 45.2 (min: 43.1, max: 47.8, stdev: 1.8)
  Avg TTFT: 280 ms

Results for llama3.2:1b:
  Avg Tokens/Sec: 156.3 (min: 152.1, max: 159.4, stdev: 2.3)
  Avg TTFT: 85 ms
```

Interpretation:
- Smaller model (1b) is ~3.5x faster for throughput
- Smaller model has much better latency (1/3 the TTFT)
- Both models show consistent performance (low stdev)
- Choose based on your requirements: speed vs quality

## License

MIT License - Feel free to modify and use as needed!

## Contributing

Contributions are welcome! Please feel free to:

1. **Report Issues**: Found a bug? Open an issue on GitHub
2. **Submit PRs**: Have an improvement? We'd love to review it
3. **Suggest Features**: Ideas for new capabilities? Let us know

### What We're Looking For

- Bug fixes and error handling improvements
- New server/framework integrations
- Better statistical analysis options
- Performance optimizations
- Documentation improvements

Thanks for helping improve LLM Benchmark!