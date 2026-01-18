# EvalScope Experiment Runner

A parallel experiment runner for benchmarking LLM models using vLLM and EvalScope.

## Features

- **Parallel Execution**: Run multiple models simultaneously on different GPUs
- **Smart GPU Scheduler**: 
  - Detects free GPUs via `nvidia-smi`
  - Supports non-consecutive GPU allocation
  - Best-fit bin packing for optimal utilization
  - Handles models with different tensor parallelism (TP) sizes
- **Auto-Resume**: Automatically skips completed models on restart
- **Model Type Detection**: Automatically detects chat vs base models
- **Configurable Token Limits**: Reads `max_tokens` from model's `generation_config.json`
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Error Tracking**: Separate error logs for failed models
- **Model Download Tools**: Scripts to download and verify HuggingFace models

## Requirements

- Python 3.8+
- NVIDIA GPUs with CUDA support
- vLLM
- EvalScope

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/evalscope-experiment-runner.git
cd evalscope-experiment-runner

# Install dependencies
pip install -r requirements.txt

# Install EvalScope (if not already installed)
pip install evalscope
```

## Usage

### Basic Usage

```bash
# Run with default settings (parallel mode, all GPUs)
./run.sh

# Or directly with Python
python run_experiment.py
```

### Command Line Options

```bash
python run_experiment.py [OPTIONS]

Options:
  -f, --file FILE          JSON file with model data (default: found_models_full.json)
  -d, --model-dir DIR      Model directory (default: /mnt/baai_cp_perf/hf_models)
  -p, --port PORT          Base vLLM port (default: 8010)
  --parallel N             Number of parallel workers (default: 8, use 1 for sequential)
  --tp N                   Only test models with specific TP size (1, 2, 4, or 8)
  --start N                Start index (0-based)
  --end N                  End index (exclusive)
  --fresh                  Ignore cache, rerun all models
  --retry-failed           Retry previously failed models
  --dry-run                Preview without running
  --list                   List models and exit
  --status                 Show experiment status and exit
  -v, --verbose            Verbose logging
```

### Examples

```bash
# Run all models in parallel (default)
./run.sh

# Run models 10-20 only
./run.sh --start 10 --end 20

# Run only TP=1 models
./run.sh --tp 1

# Retry failed models
./run.sh --retry-failed

# Sequential mode (1 GPU at a time)
./run.sh --parallel 1

# Preview what would run
./run.sh --dry-run --list

# Check current status
./run.sh --status
```

## Model JSON Format

Create a JSON file with your models:

```json
[
  {
    "model_id": "organization/model-name",
    "model_name": "Model Display Name",
    "size_bytes": 14000000000,
    "tensor_parallel_size": 1
  },
  {
    "model_id": "organization/large-model",
    "model_name": "Large Model",
    "size_bytes": 70000000000,
    "tensor_parallel_size": 4
  }
]
```

## Output Structure

```
experiment_results/
├── experiment.log          # Main experiment log
├── results.json            # All model results
├── errors.json             # Error details
├── vllm_logs/              # vLLM server logs per model
│   ├── model-name_stdout.log
│   └── model-name_stderr.log
└── error_logs/             # Detailed error logs
    └── model-name_timestamp.txt
```

## GPU Scheduler

The scheduler automatically:

1. **Detects free GPUs** at startup using `nvidia-smi`
2. **Allocates GPUs** based on model's TP requirement
3. **Supports non-consecutive GPUs** (vLLM remaps via CUDA_VISIBLE_DEVICES)
4. **Uses best-fit bin packing** to maximize utilization

Example scheduling:
```
[Scheduler] large-model (TP=4) → GPUs [0,1,2,3], port 8010 | Free: 4/8
[Scheduler] medium-model (TP=2) → GPUs [4,5], port 8011 | Free: 2/8
[Scheduler] small-model-1 (TP=1) → GPUs [6], port 8012 | Free: 1/8
[Scheduler] small-model-2 (TP=1) → GPUs [7], port 8013 | Free: 0/8
```

## Benchmark Datasets

Default datasets (configurable in code):
- MMLU (elementary_mathematics, high_school_european_history, nutrition)

Additional supported datasets:
- MMLU Pro, CEval, CMMLU, BBH, GPQA Diamond, Math 500, and more

## Model Download Tools

### Check Downloaded Models

Verify which models are fully downloaded:

```bash
# Check all models in the JSON file
python check_models_download.py -f models.json -d /path/to/hf_models

# Check specific TP size
python check_models_download.py --tp 1

# Generate list of incomplete models
python check_models_download.py --list-file incomplete_models.txt
```

### Download Models

Download models from HuggingFace:

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token

# Download models from a list file
./download.sh --list incomplete_models.txt --dir /path/to/hf_models

# Or use Python directly
python local_download.py --list models.txt --dir /path/to/hf_models

# For faster downloads in China, use mirror:
export HF_ENDPOINT=https://hf-mirror.com
./download.sh --list models.txt --dir /path/to/hf_models
```

Features:
- Automatic retry on transient errors
- Handles partial downloads with `force_download`
- Skips gated/permission errors (no useless retries)
- Logs failures to `local_download.log`

## Notes

- **Chat models only**: Base models (without `chat_template`) are skipped
- **Auto token limits**: Reads `max_tokens` from model config, capped at 4096
- **Graceful shutdown**: Ctrl+C saves state for resume

## License

MIT License

