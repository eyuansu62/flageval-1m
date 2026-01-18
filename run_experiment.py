#!/usr/bin/env python3
"""
Experiment Runner: Test all models with vLLM and evalscope benchmarks.

Features:
- Starts vLLM server for each model
- Waits for server readiness
- Runs benchmark tests
- Comprehensive error tracking and logging
- Graceful shutdown and cleanup
- AUTO-RESUME: Automatically skips completed models (default behavior)
- Efficient GPU usage (respects TP size <= 8 GPUs)

Usage:
    ./run_experiment.py [OPTIONS]

Examples:
    ./run_experiment.py                          # Run all models (auto-resume)
    ./run_experiment.py --start 10 --end 20      # Run models 10-19
    ./run_experiment.py --tp 4                   # Only TP=4 models
    ./run_experiment.py --retry-failed           # Retry previously failed models
    ./run_experiment.py --fresh                  # Start fresh, ignore cache
    ./run_experiment.py --dry-run                # Preview without running
    ./run_experiment.py --status                 # Show current status
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import requests

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_JSON = "found_models_full.json"
DEFAULT_MODEL_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
DEFAULT_PORT = 8010
DEFAULT_TIMEOUT_SERVER = 600  # 10 minutes to start server
DEFAULT_TIMEOUT_HEALTH = 10   # 10 seconds for health check
MAX_GPUS = 8
GB = 1073741824.0

RESULTS_DIR = Path("experiment_results")
LOG_FILE = RESULTS_DIR / "experiment.log"
RESULTS_FILE = RESULTS_DIR / "results.json"
ERRORS_FILE = RESULTS_DIR / "errors.json"
SKIP_MODELS_FILE = "models_incomplete_10.txt"

# Thread-safe lock for state updates in parallel mode
STATE_LOCK = threading.Lock()

# ============================================================================
# Data Classes
# ============================================================================

class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    VLLM_START_ERROR = "vllm_start_error"
    VLLM_TIMEOUT = "vllm_timeout"
    VLLM_CRASH = "vllm_crash"
    TEST_ERROR = "test_error"
    TEST_TIMEOUT = "test_timeout"
    SKIPPED = "skipped"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ModelResult:
    model_id: str
    model_name: str
    size_bytes: int
    tp_size: int
    status: Status = Status.PENDING
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    vllm_pid: Optional[int] = None
    vllm_version: Optional[str] = None
    vllm_error: Optional[str] = None
    vllm_stderr: Optional[str] = None
    test_error: Optional[str] = None
    test_stderr: Optional[str] = None
    test_stdout: Optional[str] = None
    traceback: Optional[str] = None
    # Evaluation results
    datasets_tested: Optional[list] = None
    eval_scores: Optional[dict] = None  # {dataset: {subset: score, ...}, ...}
    eval_output_dir: Optional[str] = None
    # Efficiency metrics
    timing: Optional[dict] = None  # Detailed timing breakdown
    gpu_memory_gb: Optional[float] = None  # Peak GPU memory used
    throughput: Optional[dict] = None  # Throughput metrics
    
    def to_dict(self):
        d = asdict(self)
        d['status'] = self.status.value
        return d


@dataclass
class ExperimentState:
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_models: int = 0
    completed: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    results: dict = field(default_factory=dict)  # model_id -> ModelResult
    # Global metadata
    vllm_version: Optional[str] = None
    datasets_config: Optional[list] = None
    python_version: Optional[str] = None
    hostname: Optional[str] = None
    
    def to_dict(self):
        return {
            'started_at': self.started_at,
            'vllm_version': self.vllm_version,
            'python_version': self.python_version,
            'hostname': self.hostname,
            'datasets_config': self.datasets_config,
            'total_models': self.total_models,
            'completed': self.completed,
            'success': self.success,
            'failed': self.failed,
            'skipped': self.skipped,
            'results': {k: v.to_dict() for k, v in self.results.items()}
        }


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_file: Path, verbose: bool = False):
    """Setup logging to file and console."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# Utility Functions
# ============================================================================

def model_id_to_dir(model_id: str) -> str:
    """Convert model_id (org/model) to directory format (models--org--model)."""
    return f"models--{model_id.replace('/', '--')}"


def resolve_hf_model_path(base_path: Path) -> Path:
    """
    Resolve the actual model path from HuggingFace cache directory structure.
    
    HuggingFace cache structure:
        models--org--model-name/
        ├── blobs/
        ├── refs/
        │   └── main (contains the commit hash)
        └── snapshots/
            └── <commit_hash>/
                ├── config.json
                └── ...
    
    Returns the snapshot directory containing config.json, or base_path if not HF cache format.
    """
    snapshots_dir = base_path / "snapshots"
    
    # Check if this is HF cache format
    if not snapshots_dir.exists():
        # Not HF cache format, return as-is (might be direct model path)
        return base_path
    
    # Try to read the commit hash from refs/main
    refs_main = base_path / "refs" / "main"
    if refs_main.exists():
        commit_hash = refs_main.read_text().strip()
        snapshot_path = snapshots_dir / commit_hash
        if snapshot_path.exists() and (snapshot_path / "config.json").exists():
            return snapshot_path
    
    # Fallback: find any snapshot directory with config.json
    for snapshot in snapshots_dir.iterdir():
        if snapshot.is_dir():
            if (snapshot / "config.json").exists():
                return snapshot
            # Also check for Mistral format (params.json)
            if (snapshot / "params.json").exists():
                return snapshot
    
    # No valid snapshot found, return base path
    logging.warning(f"No valid model snapshot found in {base_path}")
    return base_path


def load_models(json_file: str) -> list:
    """Load models from JSON file."""
    with open(json_file) as f:
        return json.load(f)


def load_skip_models(skip_file: str) -> set:
    """
    Load model IDs to skip (one per line). Lines starting with '#' are ignored.
    """
    skip_path = Path(skip_file)
    if not skip_path.exists():
        return set()
    
    skip_models = set()
    for line in skip_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        skip_models.add(line)
    return skip_models


def save_state(state: ExperimentState):
    """Save experiment state to file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)


def load_state() -> Optional[ExperimentState]:
    """Load experiment state from file if exists."""
    if not RESULTS_FILE.exists():
        return None
    try:
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        state = ExperimentState(
            started_at=data.get('started_at', datetime.now().isoformat()),
            total_models=data.get('total_models', 0),
            completed=data.get('completed', 0),
            success=data.get('success', 0),
            failed=data.get('failed', 0),
            skipped=data.get('skipped', 0),
        )
        for model_id, result_data in data.get('results', {}).items():
            result = ModelResult(
                model_id=result_data['model_id'],
                model_name=result_data['model_name'],
                size_bytes=result_data['size_bytes'],
                tp_size=result_data['tp_size'],
                status=Status(result_data['status']),
                start_time=result_data.get('start_time'),
                end_time=result_data.get('end_time'),
                duration_seconds=result_data.get('duration_seconds'),
                vllm_pid=result_data.get('vllm_pid'),
                vllm_error=result_data.get('vllm_error'),
                vllm_stderr=result_data.get('vllm_stderr'),
                test_error=result_data.get('test_error'),
                test_stderr=result_data.get('test_stderr'),
                test_stdout=result_data.get('test_stdout'),
                traceback=result_data.get('traceback'),
            )
            state.results[model_id] = result
        return state
    except Exception as e:
        logging.warning(f"Failed to load state: {e}")
        return None


def save_error(model_id: str, error_type: str, error_msg: str, details: dict = None):
    """Append error to errors file and save detailed error log."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create error logs directory
    error_logs_dir = RESULTS_DIR / "error_logs"
    error_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed error to individual file
    safe_model_name = model_id.replace("/", "__").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_log_file = error_logs_dir / f"{safe_model_name}_{timestamp}.txt"
    
    # Write detailed error log
    with open(error_log_file, 'w') as f:
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Error Type: {error_type}\n")
        f.write(f"Error Message: {error_msg}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        if details:
            for key, value in details.items():
                f.write(f"--- {key} ---\n")
                f.write(str(value) + "\n\n")
    
    # Load existing errors
    errors = []
    if ERRORS_FILE.exists():
        try:
            with open(ERRORS_FILE) as f:
                errors = json.load(f)
        except:
            pass
    
    # Append summary to errors.json (truncate large fields for JSON)
    error_entry = {
        'timestamp': datetime.now().isoformat(),
        'model_id': model_id,
        'error_type': error_type,
        'error_msg': error_msg,
        'details_file': str(error_log_file),
    }
    
    # Add summary of details (truncated for JSON readability)
    if details:
        summary = {}
        for key, value in details.items():
            if isinstance(value, str) and len(value) > 500:
                summary[key] = value[:500] + f"... [truncated, see {error_log_file}]"
            else:
                summary[key] = value
        error_entry['details_summary'] = summary
    
    errors.append(error_entry)
    
    with open(ERRORS_FILE, 'w') as f:
        json.dump(errors, f, indent=2)


def get_vllm_version() -> str:
    """Get vLLM version string."""
    try:
        import vllm
        return vllm.__version__
    except ImportError:
        pass
    
    # Try pip show
    try:
        result = subprocess.run(
            ["pip", "show", "vllm"],
            capture_output=True,
            text=True,
            timeout=10
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':', 1)[1].strip()
    except:
        pass
    
    # Try vllm command
    try:
        result = subprocess.run(
            ["vllm", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout.strip() or result.stderr.strip()
        if output:
            return output
    except:
        pass
    
    return "unknown"


def get_gpu_memory_usage() -> dict:
    """Get current GPU memory usage in GB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        gpus = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    idx = int(parts[0])
                    gpus[f"gpu_{idx}"] = {
                        "memory_used_gb": float(parts[1]) / 1024,
                        "memory_total_gb": float(parts[2]) / 1024,
                        "utilization_pct": float(parts[3])
                    }
        
        # Calculate totals
        if gpus:
            gpus["total_memory_used_gb"] = sum(g["memory_used_gb"] for g in gpus.values() if isinstance(g, dict))
            gpus["total_memory_total_gb"] = sum(g["memory_total_gb"] for g in gpus.values() if isinstance(g, dict))
        
        return gpus
    except Exception as e:
        logging.debug(f"Error getting GPU memory: {e}")
        return {}


def wait_for_server(port: int, timeout: int = DEFAULT_TIMEOUT_SERVER) -> bool:
    """Wait for vLLM server to be ready."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=DEFAULT_TIMEOUT_HEALTH)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    
    return False


def kill_process(pid: int):
    """Kill a process and its children."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(2)
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        logging.warning(f"Error killing process {pid}: {e}")


def cleanup_gpu():
    """Try to clean up GPU memory."""
    try:
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints"],
            capture_output=True,
            timeout=10
        )
        time.sleep(5)
    except:
        pass


# ============================================================================
# Main Experiment Logic
# ============================================================================

def run_vllm_server(model_path: str, model_id: str, tp_size: int, port: int, extra_args: str = "", gpu_ids: list = None) -> subprocess.Popen:
    """Start vLLM server as background process.
    
    Args:
        model_path: Path to the model
        model_id: Model identifier
        tp_size: Tensor parallelism size
        port: Port to run the server on
        extra_args: Extra arguments for vLLM
        gpu_ids: List of GPU IDs to use (e.g., [0, 1] for GPUs 0 and 1). If None, uses all GPUs.
    """
    cmd = [
        "vllm", "serve",
        model_path,
        "--tensor-parallel-size", str(tp_size),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--trust-remote-code",
    ]
    
    if extra_args:
        cmd.extend(extra_args.split())
    
    # Set up environment with GPU assignment
    env = os.environ.copy()
    if gpu_ids is not None:
        cuda_devices = ",".join(str(g) for g in gpu_ids)
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        logging.info(f"Assigning GPUs {gpu_ids} (CUDA_VISIBLE_DEVICES={cuda_devices}) to model on port {port}")
    
    logging.debug(f"Starting vLLM: {' '.join(cmd)}")
    
    # Create log files for this run - use model_id for readable names
    log_dir = RESULTS_DIR / "vllm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert model_id to safe filename: org/model-name -> org__model-name
    safe_model_name = model_id.replace("/", "__").replace(" ", "_")
    stdout_file = open(log_dir / f"{safe_model_name}_stdout.log", 'w')
    stderr_file = open(log_dir / f"{safe_model_name}_stderr.log", 'w')
    
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_file,
        stderr=stderr_file,
        preexec_fn=os.setsid,  # Create new process group for cleanup
        env=env,
    )
    
    return proc


# ============================================================================
# Test Configuration
# ============================================================================

# Load environment variables once at module level
try:
    from dotenv import dotenv_values
    ENV = dotenv_values('.env')  # Load from current directory
except:
    ENV = os.environ  # Fallback to environment variables

# DATASETS = [
#     'iquiz', 'ifeval', 'mmlu', 'mmlu_pro', 'musr', 'process_bench', 'race',
#     'cmmlu', 'humaneval', 'gsm8k', 'bbh', 'competition_math', 'math_500',
#     'aime24', 'gpqa_diamond', 'arc', 'ceval', 'hellaswag', 'general_mcq',
#     'general_qa', 'super_gpqa', 'mmlu_redux', 'simple_qa', 'chinese_simpleqa',
#     'alpaca_eval', 'arena_hard', 'maritime_bench', 'drop', 'winogrande',
#     'tool_bench', 'frames', 'docmath', 'needle_haystack', 'bfcl_v3', 'hle', 'tau_bench',
# ]

DATASETS = ["mmlu"]
DATASET_ARGS = {
    'mmlu': {'subset_list': ['elementary_mathematics', 'high_school_european_history', 'nutrition'], 'few_shot_num': 0},
    'mmlu_pro': {'subset_list': ['math', 'health'], 'few_shot_num': 4},
    'ceval': {'subset_list': ['computer_network', 'operating_system', 'computer_architecture'], 'few_shot_num': 0},
    'cmmlu': {'subset_list': ['elementary_chinese'], 'few_shot_num': 0},
    'bbh': {'subset_list': ['word_sorting', 'movie_recommendation']},
    'gpqa_diamond': {'few_shot_num': 0},
    'competition_math': {'subset_list': ['Level 1']},
    'math_500': {'subset_list': ['Level 1']},
    'process_bench': {'subset_list': ['gsm8k']},
    'musr': {'subset_list': ['murder_mysteries']},
    'super_gpqa': {'subset_list': ['Philosophy', 'Education'], 'few_shot_num': 0},
    'chinese_simpleqa': {'subset_list': ['中华文化']},
    'mmlu_redux': {'subset_list': ['abstract_algebra']},
    'docmath': {'subset_list': ['simpshort_testmini']},
    'bfcl_v3': {'subset_list': ['simple', 'multiple']},
    'hle': {'subset_list': ['Math', 'Other']},
    'general_mcq': {'local_path': 'custom_eval/text/mcq', 'subset_list': ['example']},
    'general_qa': {'local_path': 'custom_eval/text/qa', 'subset_list': ['example']},
    'tau_bench': {
        'extra_params': {
            'user_model': 'qwen-plus',
            'api_key': ENV.get('DASHSCOPE_API_KEY'),
            'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        },
        'subset_list': ['airline'],
    },
}


def parse_eval_results(output_dir: str, model_id: str) -> dict:
    """Parse evaluation results from evalscope output directory."""
    scores = {}
    
    try:
        # Find the latest output directory for this model
        output_path = Path(output_dir)
        if not output_path.exists():
            return scores
        
        # Look for report JSON files
        model_name = model_id.replace("/", "_")
        reports_dir = output_path / "reports" / model_name
        
        if reports_dir.exists():
            for report_file in reports_dir.glob("*.json"):
                try:
                    with open(report_file) as f:
                        report_data = json.load(f)
                    
                    dataset_name = report_file.stem
                    scores[dataset_name] = {}
                    
                    # Extract scores from report
                    if isinstance(report_data, list):
                        for item in report_data:
                            subset = item.get('subset', 'overall')
                            score = item.get('score', item.get('accuracy', item.get('mean_acc')))
                            if score is not None:
                                scores[dataset_name][subset] = score
                    elif isinstance(report_data, dict):
                        for subset, data in report_data.items():
                            if isinstance(data, dict):
                                score = data.get('score', data.get('accuracy', data.get('mean_acc')))
                                if score is not None:
                                    scores[dataset_name][subset] = score
                            elif isinstance(data, (int, float)):
                                scores[dataset_name][subset] = data
                except Exception as e:
                    logging.debug(f"Error parsing {report_file}: {e}")
                    continue
    except Exception as e:
        logging.debug(f"Error parsing eval results: {e}")
    
    return scores


def find_latest_output_dir() -> Optional[str]:
    """Find the latest evalscope output directory."""
    outputs_path = Path("./outputs")
    if not outputs_path.exists():
        return None
    
    # Get latest directory by name (format: YYYYMMDD_HHMMSS)
    dirs = sorted([d for d in outputs_path.iterdir() if d.is_dir()], reverse=True)
    if dirs:
        return str(dirs[0])
    return None


def get_model_max_tokens(model_path: str, default: int = 2048, max_cap: int = 4096) -> int:
    """
    Read max_tokens from model's generation_config.json if available.
    
    Checks for these fields in order: max_new_tokens.
    Also reads context length from config.json to ensure we leave room for input tokens.
    
    Args:
        model_path: Path to the model directory
        default: Default max_tokens if not found in config (default: 2048)
        max_cap: Maximum cap for max_tokens to leave room for input (default: 4096)
    
    Returns:
        A safe max_tokens value that leaves room for input tokens.
    """
    max_tokens = default
    context_length = None
    
    # First, try to get context length from config.json
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check common field names for context length
            for field in ['max_position_embeddings', 'n_positions', 'max_seq_len', 'seq_length', 'model_max_length']:
                if field in config and config[field] is not None:
                    context_length = config[field]
                    logging.debug(f"Found context length {field}={context_length} in config.json")
                    break
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to read config.json: {e}")
    
    # Then, try to get max_new_tokens from generation_config.json
    gen_config_path = Path(model_path) / "generation_config.json"
    if gen_config_path.exists():
        try:
            with open(gen_config_path, 'r') as f:
                gen_config = json.load(f)
            
            # Prefer max_new_tokens as it's specifically for generation
            if 'max_new_tokens' in gen_config and gen_config['max_new_tokens'] is not None:
                max_tokens = gen_config['max_new_tokens']
                logging.debug(f"Found max_new_tokens={max_tokens} in generation_config.json")
            
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to read generation_config.json: {e}")
    else:
        logging.debug(f"No generation_config.json found at {gen_config_path}")
    
    # Apply safety cap: if we have context length, use at most half of it (leave room for input)
    # Otherwise, use the max_cap
    if context_length:
        # Leave at least 25% of context for input tokens
        safe_max = int(context_length * 0.75)
        max_tokens = min(max_tokens, safe_max, max_cap)
        logging.debug(f"Context length={context_length}, capping max_tokens to {max_tokens}")
    else:
        max_tokens = min(max_tokens, max_cap)
    
    logging.debug(f"Final max_tokens={max_tokens}")
    return max_tokens


def is_chat_model(model_path: str) -> bool:
    """
    Determine if a model is a chat model by checking for chat_template in tokenizer_config.json.
    
    Returns:
        True if the model has a chat_template (chat model), False otherwise (base model).
    """
    tokenizer_config_path = Path(model_path) / "tokenizer_config.json"
    
    if tokenizer_config_path.exists():
        try:
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)
            
            # Check if chat_template exists and is not empty
            if 'chat_template' in tokenizer_config:
                chat_template = tokenizer_config['chat_template']
                if chat_template and (isinstance(chat_template, str) or isinstance(chat_template, list)):
                    logging.debug(f"Found chat_template in tokenizer_config.json - this is a chat model")
                    return True
            
            logging.debug(f"No chat_template found in tokenizer_config.json - this is a base model")
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to read tokenizer_config.json: {e}, assuming base model")
    else:
        logging.debug(f"No tokenizer_config.json found at {tokenizer_config_path}, assuming base model")
    
    return False


def run_quick_test(model_id: str, model_path: str, port: int, num_samples: int = 5) -> tuple:
    """
    Run a quick sanity test with a small number of samples to verify API works.
    
    Args:
        model_id: The model identifier
        model_path: Path to the model
        port: The port the vLLM server is running on
        num_samples: Number of samples to test (default: 5)
    
    Returns:
        tuple: (success: bool, error_message: str)
    """
    try:
        from evalscope.config import TaskConfig
        from evalscope.constants import EvalType
        from evalscope.run import run_task
        
        # Get max_tokens from model's generation config
        max_tokens = get_model_max_tokens(model_path)
        
        # Determine if this is a chat model or base model
        chat_model = is_chat_model(model_path)
        if chat_model:
            api_url = f'http://localhost:{port}/v1/chat/completions'
        else:
            api_url = f'http://localhost:{port}/v1/completions'
        
        logging.info(f"Running quick sanity test ({num_samples} samples from mmlu)...")
        
        # Create a minimal task config for quick test
        task_cfg = TaskConfig(
            model=model_path,
            model_id=f"{model_id.replace('/', '_')}_quicktest",
            api_url=api_url,
            api_key='EMPTY',
            eval_type=EvalType.SERVICE,
            datasets=['mmlu'],
            dataset_args={
                'mmlu': {'subset_list': ['elementary_mathematics'], 'few_shot_num': 0}
            },
            eval_batch_size=num_samples,
            limit=num_samples,  # Only test a few samples
            stream=False,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': min(max_tokens, 512),  # Use smaller max_tokens for quick test
            },
        )
        
        # Run the quick test
        run_task(task_cfg=task_cfg)
        
        logging.info(f"Quick sanity test PASSED!")
        return True, ""
        
    except Exception as e:
        error_msg = f"Quick sanity test FAILED: {str(e)}"
        logging.error(error_msg)
        return False, error_msg


def run_test(model_id: str, model_path: str, port: int) -> tuple:
    """
    Run the benchmark test directly in Python.
    
    Returns:
        tuple: (returncode, stdout, stderr, eval_scores, output_dir)
    """
    eval_scores = {}
    output_dir = None
    
    try:
        # Import evalscope components
        from evalscope.config import TaskConfig
        from evalscope.constants import EvalType
        from evalscope.run import run_task
        
        # Get max_tokens from model's generation config
        max_tokens = get_model_max_tokens(model_path)
        logging.info(f"Using max_tokens={max_tokens} for model {model_id}")
        
        # Determine if this is a chat model or base model
        chat_model = is_chat_model(model_path)
        if chat_model:
            api_url = f'http://localhost:{port}/v1/chat/completions'
            logging.info(f"Model {model_id} detected as CHAT model, using chat/completions endpoint")
        else:
            api_url = f'http://localhost:{port}/v1/completions'
            logging.info(f"Model {model_id} detected as BASE model, using completions endpoint")
        
        # Create task config
        task_cfg = TaskConfig(
            model=model_path,
            model_id=model_id.replace("/", "_"),
            api_url=api_url,
            api_key='EMPTY',
            eval_type=EvalType.SERVICE,
            datasets=DATASETS,
            dataset_args=DATASET_ARGS,
            eval_batch_size=16,
            stream=False,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': max_tokens,
            },
        )
        
        # Run the task
        run_task(task_cfg=task_cfg)
        
        # Find output directory and parse results
        output_dir = find_latest_output_dir()
        if output_dir:
            eval_scores = parse_eval_results(output_dir, model_id)
        
        return 0, "TEST_COMPLETED_SUCCESSFULLY", "", eval_scores, output_dir
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        return 1, "", f"{error_msg}\n{tb}", eval_scores, output_dir


def test_single_model(
    model: dict,
    model_dir: str,
    port: int,
    extra_args: str,
    dry_run: bool,
    state: ExperimentState,
    idx: int,
    total: int,
    gpu_ids: list = None
) -> ModelResult:
    """Test a single model end-to-end.
    
    Args:
        gpu_ids: List of GPU IDs to use for this model. If None, uses all available GPUs.
    """
    model_id = model['model_id']
    model_name = model.get('model_name', model_id)
    size_bytes = model.get('size_bytes')
    tp_size = model.get('tensor_parallel_size', 1)
    
    result = ModelResult(
        model_id=model_id,
        model_name=model_name,
        size_bytes=size_bytes,
        tp_size=tp_size,
        start_time=datetime.now().isoformat()
    )
    
    size_gb = size_bytes / GB
    logging.info("=" * 70)
    logging.info(f"[{idx}/{total}] Testing: {model_id}")
    logging.info(f"Size: {size_gb:.2f} GB | TP: {tp_size} | Port: {port}")
    
    if dry_run:
        logging.info("[DRY-RUN] Skipping execution")
        result.status = Status.SKIPPED
        result.end_time = datetime.now().isoformat()
        return result
    
    # Check TP size
    if tp_size > MAX_GPUS:
        logging.warning(f"Skipping: TP={tp_size} > {MAX_GPUS} available GPUs")
        result.status = Status.SKIPPED
        result.vllm_error = f"TP size {tp_size} exceeds available GPUs ({MAX_GPUS})"
        result.end_time = datetime.now().isoformat()
        return result
    
    # Build model path
    dir_name = model_id_to_dir(model_id)
    base_model_path = Path(model_dir) / dir_name
    
    if not base_model_path.exists():
        logging.error(f"Model directory not found: {base_model_path}")
        result.status = Status.SKIPPED
        result.vllm_error = f"Model directory not found: {base_model_path}"
        result.end_time = datetime.now().isoformat()
        save_error(model_id, "MODEL_NOT_FOUND", str(result.vllm_error))
        return result
    
    # Resolve HuggingFace cache path to actual model directory
    model_path = resolve_hf_model_path(base_model_path)
    logging.info(f"Resolved model path: {model_path}")
    
    if not (model_path / "config.json").exists() and not (model_path / "params.json").exists():
        logging.error(f"No config.json or params.json found in: {model_path}")
        result.status = Status.SKIPPED
        result.vllm_error = f"No valid model config found in: {model_path}"
        result.end_time = datetime.now().isoformat()
        save_error(model_id, "MODEL_CONFIG_NOT_FOUND", str(result.vllm_error))
        return result
    
    # Skip base models (non-chat models) - we only support chat models for now
    if not is_chat_model(str(model_path)):
        logging.warning(f"Skipping base model (no chat_template): {model_id}")
        result.status = Status.SKIPPED
        result.vllm_error = "Base model not supported (no chat_template in tokenizer_config.json)"
        result.end_time = datetime.now().isoformat()
        save_error(model_id, "BASE_MODEL_NOT_SUPPORTED", result.vllm_error)
        return result
    
    vllm_proc = None
    timing = {
        "server_start_seconds": None,
        "test_execution_seconds": None,
        "total_seconds": None,
    }
    
    try:
        # Step 1: Start vLLM server
        logging.info("Starting vLLM server...")
        result.status = Status.RUNNING
        server_start_time = time.time()
        
        # Safe model name for log files
        safe_model_name = model_id.replace("/", "__").replace(" ", "_")
        
        vllm_proc = run_vllm_server(str(model_path), model_id, tp_size, port, extra_args, gpu_ids=gpu_ids)
        result.vllm_pid = vllm_proc.pid
        gpu_info = f" on GPU(s) {gpu_ids}" if gpu_ids else ""
        logging.info(f"vLLM started with PID: {vllm_proc.pid}{gpu_info}")
        
        # Step 2: Wait for server to be ready
        logging.info("Waiting for server to be ready...")
        server_ready = False
        
        while time.time() - server_start_time < DEFAULT_TIMEOUT_SERVER:
            # Check if process crashed
            if vllm_proc.poll() is not None:
                # Process ended - read full stderr for error details
                log_dir = RESULTS_DIR / "vllm_logs"
                stderr_file = log_dir / f"{safe_model_name}_stderr.log"
                stderr_content = ""
                if stderr_file.exists():
                    stderr_content = stderr_file.read_text()
                
                result.status = Status.VLLM_CRASH
                result.vllm_error = f"vLLM process crashed with exit code: {vllm_proc.returncode}"
                result.vllm_stderr = stderr_content  # Store full content
                logging.error(f"vLLM crashed: {result.vllm_error}")
                # Save full error to errors.json (no truncation)
                save_error(model_id, "VLLM_CRASH", result.vllm_error, {
                    'stderr': stderr_content,
                    'log_file': str(stderr_file)
                })
                break
            
            # Check health endpoint
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=5)
                if resp.status_code == 200:
                    server_ready = True
                    timing["server_start_seconds"] = round(time.time() - server_start_time, 2)
                    logging.info(f"Server ready after {timing['server_start_seconds']}s")
                    break
            except:
                pass
            
            time.sleep(5)
        
        if not server_ready and result.status == Status.RUNNING:
            result.status = Status.VLLM_TIMEOUT
            result.vllm_error = f"Server failed to start within {DEFAULT_TIMEOUT_SERVER}s"
            logging.error(result.vllm_error)
            save_error(model_id, "VLLM_TIMEOUT", result.vllm_error)
        
        # Step 3: Run tests if server is ready
        if server_ready:
            logging.info("Running benchmark tests...")
            # Store vLLM version and datasets info
            result.vllm_version = get_vllm_version()
            result.datasets_tested = DATASETS.copy()
            
            # Get GPU memory usage before test
            gpu_before = get_gpu_memory_usage()
            peak_gpu_memory = gpu_before.get("total_memory_used_gb", 0)
            
            # Step 3a: Run quick sanity test first
            quick_test_passed, quick_test_error = run_quick_test(model_id, str(model_path), port)
            
            if not quick_test_passed:
                result.status = Status.TEST_ERROR
                result.test_error = f"Quick sanity test failed: {quick_test_error}"
                logging.error(f"Skipping full benchmark due to failed quick test")
                save_error(model_id, "QUICK_TEST_FAILED", quick_test_error)
                # Skip to cleanup
                result.end_time = datetime.now().isoformat()
                if result.start_time:
                    start_dt = datetime.fromisoformat(result.start_time)
                    end_dt = datetime.fromisoformat(result.end_time)
                    result.duration_seconds = round((end_dt - start_dt).total_seconds(), 2)
                return result
            
            # Step 3b: Run full benchmark tests
            test_start_time = time.time()
            
            try:
                returncode, stdout, stderr, eval_scores, output_dir = run_test(model_id, str(model_path), port)
                
                timing["test_execution_seconds"] = round(time.time() - test_start_time, 2)
                
                # Get GPU memory after test
                gpu_after = get_gpu_memory_usage()
                peak_gpu_memory = max(peak_gpu_memory, gpu_after.get("total_memory_used_gb", 0))
                result.gpu_memory_gb = round(peak_gpu_memory, 2)
                
                result.test_stdout = stdout if stdout else ""
                result.test_stderr = stderr if stderr else ""
                result.eval_scores = eval_scores
                result.eval_output_dir = output_dir
                
                # Calculate throughput if we have timing info
                if timing["test_execution_seconds"] and timing["test_execution_seconds"] > 0:
                    # Count total samples from eval_scores
                    total_samples = 0
                    if eval_scores:
                        for dataset_scores in eval_scores.values():
                            if isinstance(dataset_scores, dict):
                                for subset, score_info in dataset_scores.items():
                                    if isinstance(score_info, dict) and 'num' in score_info:
                                        total_samples += score_info['num']
                    
                    result.throughput = {
                        "samples_per_second": round(total_samples / timing["test_execution_seconds"], 2) if total_samples else None,
                        "total_samples": total_samples if total_samples else None,
                    }
                
                if returncode == 0:
                    result.status = Status.SUCCESS
                    logging.info(f"Tests completed successfully! (test took {timing['test_execution_seconds']}s)")
                    if eval_scores:
                        logging.info(f"Evaluation scores: {json.dumps(eval_scores, indent=2)}")
                else:
                    result.status = Status.TEST_ERROR
                    result.test_error = f"Test failed with exit code: {returncode}"
                    logging.error(f"Test failed: {result.test_error}")
                    if stderr:
                        logging.debug(f"Stderr: {stderr[:500]}")
                    save_error(model_id, "TEST_ERROR", result.test_error, {
                        'stderr': stderr or '',
                        'stdout': stdout or ''
                    })
                    
            except Exception as e:
                result.status = Status.TEST_ERROR
                result.test_error = str(e)
                result.traceback = traceback.format_exc()
                logging.error(f"Test exception: {e}")
                save_error(model_id, "TEST_EXCEPTION", str(e), {
                    'traceback': result.traceback,
                    'exception_type': type(e).__name__
                })
    
    except Exception as e:
        result.status = Status.UNKNOWN_ERROR
        result.vllm_error = str(e)
        result.traceback = traceback.format_exc()
        logging.error(f"Unexpected error: {e}")
        logging.debug(traceback.format_exc())
        save_error(model_id, "UNKNOWN_ERROR", str(e), {
            'traceback': result.traceback,
            'exception_type': type(e).__name__
        })
    
    finally:
        # Cleanup: kill vLLM process
        if vllm_proc and vllm_proc.poll() is None:
            logging.info("Stopping vLLM server...")
            kill_process(vllm_proc.pid)
        
        # Extra cleanup
        cleanup_gpu()
        
        result.end_time = datetime.now().isoformat()
        if result.start_time and result.end_time:
            start = datetime.fromisoformat(result.start_time)
            end = datetime.fromisoformat(result.end_time)
            result.duration_seconds = round((end - start).total_seconds(), 2)
            timing["total_seconds"] = result.duration_seconds
        
        # Save timing info
        result.timing = timing
    
    # Log summary
    timing_summary = f"total={timing.get('total_seconds', '?')}s"
    if timing.get('server_start_seconds'):
        timing_summary += f", server_start={timing['server_start_seconds']}s"
    if timing.get('test_execution_seconds'):
        timing_summary += f", test={timing['test_execution_seconds']}s"
    if result.gpu_memory_gb:
        timing_summary += f", gpu_mem={result.gpu_memory_gb}GB"
    
    logging.info(f"Result: {result.status.value} ({timing_summary})")
    return result


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments on all models (auto-resume by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("-f", "--file", default=DEFAULT_JSON,
                        help=f"JSON file with model data (default: {DEFAULT_JSON})")
    parser.add_argument("-d", "--model-dir", default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
                        help=f"Model directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("-p", "--port", type=int, default=int(os.environ.get("VLLM_PORT", DEFAULT_PORT)),
                        help=f"vLLM port (default: {DEFAULT_PORT})")
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8],
                        help="Only test models with this TP size")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (0-based)")
    parser.add_argument("--end", type=int,
                        help="End index (exclusive)")
    
    # Cache control options
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--fresh", action="store_true",
                        help="Start fresh: ignore cache, rerun all models")
    cache_group.add_argument("--retry-failed", action="store_true",
                        help="Retry failed models (skip only successful ones)")
    
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without running")
    parser.add_argument("--extra-args", default="",
                        help="Extra arguments to pass to vLLM")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--list", action="store_true",
                        help="List models and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show experiment status and exit")
    parser.add_argument("--parallel", type=int, default=MAX_GPUS,
                        help=f"Number of models to run in parallel (default: {MAX_GPUS}, use 1 for sequential)")
    return parser.parse_args()


def show_status(state: Optional[ExperimentState]):
    """Display current experiment status."""
    print("\n" + "=" * 70)
    print("EXPERIMENT STATUS")
    print("=" * 70)
    
    if not state or not state.results:
        print("No previous runs found.")
        print(f"Results file: {RESULTS_FILE}")
        return
    
    # Count by status
    status_counts = {}
    for result in state.results.values():
        status = result.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\nStarted at: {state.started_at}")
    print(f"Total models in cache: {len(state.results)}")
    print(f"\nStatus breakdown:")
    
    for status, count in sorted(status_counts.items()):
        pct = count / len(state.results) * 100
        bar = "█" * int(pct / 2)
        print(f"  {status:<20} {count:>5} ({pct:>5.1f}%) {bar}")
    
    # Show failed models
    failed = [r for r in state.results.values() 
              if r.status not in (Status.SUCCESS, Status.SKIPPED, Status.PENDING)]
    
    if failed:
        print(f"\nFailed models ({len(failed)}):")
        for r in failed[:10]:
            error = r.vllm_error or r.test_error or "Unknown error"
            print(f"  • {r.model_id[:45]:<45} [{r.status.value}] {error[:30]}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    print(f"\nResults file: {RESULTS_FILE}")
    print(f"Errors file: {ERRORS_FILE}")
    print("=" * 70 + "\n")


def should_skip_model(model_id: str, state: ExperimentState, fresh: bool, retry_failed: bool) -> tuple:
    """
    Determine if a model should be skipped based on cache.
    Returns: (should_skip: bool, reason: str)
    """
    # Fresh mode: never skip
    if fresh:
        return False, None
    
    # Not in cache: don't skip
    if model_id not in state.results:
        return False, None
    
    prev_result = state.results[model_id]
    prev_status = prev_result.status
    
    # Always skip successful models
    if prev_status == Status.SUCCESS:
        return True, f"cached: {prev_status.value}"
    
    # Skip DRY_RUN skipped models only if not retrying
    if prev_status == Status.SKIPPED:
        # Check if it was skipped due to DRY_RUN vs actual skip (e.g., model not found)
        if prev_result.vllm_error:
            # Was skipped for a real reason (model not found, TP too large)
            return True, f"cached: {prev_status.value} ({prev_result.vllm_error[:30]})"
        # DRY_RUN skip - don't skip on real run
        return False, None
    
    # Failed models: skip unless --retry-failed
    if prev_status in (Status.VLLM_CRASH, Status.VLLM_TIMEOUT, Status.VLLM_START_ERROR,
                       Status.TEST_ERROR, Status.TEST_TIMEOUT, Status.UNKNOWN_ERROR):
        if retry_failed:
            return False, None  # Retry
        else:
            return True, f"cached: {prev_status.value} (use --retry-failed to retry)"
    
    # RUNNING or PENDING: don't skip (incomplete)
    return False, None


def parallel_worker(
    model: dict,
    model_dir: str,
    port: int,
    extra_args: str,
    dry_run: bool,
    state: ExperimentState,
    idx: int,
    total: int,
    gpu_ids: list,
    skip_models: set,
    fresh: bool,
    retry_failed: bool
) -> tuple:
    """
    Worker function for parallel model testing.
    
    Returns:
        tuple: (model_id, result, should_update_state)
    """
    model_id = model['model_id']
    
    # Check if should skip (in skip list)
    if model_id in skip_models:
        logging.info(f"[Worker GPU {gpu_ids}] Skipping {model_id} (in skip list)")
        result = ModelResult(
            model_id=model_id,
            model_name=model.get('model_name', model_id),
            size_bytes=model.get('size_bytes'),
            tp_size=model.get('tensor_parallel_size', 1),
            status=Status.SKIPPED,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration_seconds=0.0,
            vllm_error=f"Marked incomplete in skip list"
        )
        return model_id, result, True
    
    # Check cache
    should_skip, skip_reason = should_skip_model(model_id, state, fresh, retry_failed)
    if should_skip:
        logging.debug(f"[Worker GPU {gpu_ids}] Skipping {model_id} ({skip_reason})")
        return model_id, None, False  # Don't update state for cached models
    
    logging.info(f"[Worker GPU {gpu_ids}] Starting: {model_id}")
    
    # Run the test
    result = test_single_model(
        model=model,
        model_dir=model_dir,
        port=port,
        extra_args=extra_args,
        dry_run=dry_run,
        state=state,
        idx=idx,
        total=total,
        gpu_ids=gpu_ids
    )
    
    logging.info(f"[Worker GPU {gpu_ids}] Completed: {model_id} -> {result.status.value}")
    return model_id, result, True


def update_state_with_result(state: ExperimentState, model_id: str, result: ModelResult):
    """Thread-safe state update."""
    with STATE_LOCK:
        # Handle re-runs properly
        old_result = state.results.get(model_id)
        if old_result:
            if old_result.status == Status.SUCCESS:
                state.success = max(0, state.success - 1)
            elif old_result.status == Status.SKIPPED:
                state.skipped = max(0, state.skipped - 1)
            else:
                state.failed = max(0, state.failed - 1)
            state.completed = max(0, state.completed - 1)
        
        # Update with new result
        state.results[model_id] = result
        state.completed += 1
        
        if result.status == Status.SUCCESS:
            state.success += 1
        elif result.status == Status.SKIPPED:
            state.skipped += 1
        else:
            state.failed += 1
        
        # Save state
        save_state(state)


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(LOG_FILE, args.verbose)
    
    logging.info("=" * 70)
    logging.info("EXPERIMENT RUNNER (auto-resume enabled)")
    logging.info("=" * 70)
    
    # Load cached state (always, unless --fresh)
    cached_state = None
    if not args.fresh:
        cached_state = load_state()
        if cached_state:
            cached_count = len(cached_state.results)
            success_count = sum(1 for r in cached_state.results.values() if r.status == Status.SUCCESS)
            failed_count = sum(1 for r in cached_state.results.values() 
                             if r.status not in (Status.SUCCESS, Status.SKIPPED, Status.PENDING))
            logging.info(f"Loaded cache: {cached_count} models ({success_count} success, {failed_count} failed)")
    else:
        logging.info("Fresh mode: ignoring cache")
    
    # Status mode
    if args.status:
        show_status(cached_state)
        return
    
    # Load models
    try:
        models = load_models(args.file)
        logging.info(f"Loaded {len(models)} models from {args.file}")
    except FileNotFoundError:
        logging.error(f"File not found: {args.file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON: {e}")
        sys.exit(1)
    
    # Filter by TP size
    if args.tp:
        models = [m for m in models if m.get('tensor_parallel_size') == args.tp]
        logging.info(f"Filtered to {len(models)} models with TP={args.tp}")
    
    # Load skip list of incomplete models
    skip_models = load_skip_models(SKIP_MODELS_FILE)
    if skip_models:
        logging.info(f"Skip list: {len(skip_models)} models from {SKIP_MODELS_FILE}")
    else:
        logging.info(f"No skip list found at {SKIP_MODELS_FILE}")
    
    # Filter out models with TP > MAX_GPUS
    original_count = len(models)
    models = [m for m in models if m.get('tensor_parallel_size', 1) <= MAX_GPUS]
    if len(models) < original_count:
        logging.info(f"Excluded {original_count - len(models)} models with TP > {MAX_GPUS}")
    
    # Apply range
    end_idx = args.end if args.end else len(models)
    models = models[args.start:end_idx]
    
    if not models:
        logging.info("No models to test")
        return
    
    # List mode - show cache status for each model
    if args.list:
        print(f"\n{'Model ID':<45} {'Size':>8} {'TP':>3} {'Status':<15}")
        print("-" * 75)
        for m in models:
            model_id = m['model_id']
            size_gb = m.get('size_bytes', 0) / GB
            tp = m.get('tensor_parallel_size', '?')
            
            # Check cache status
            status = "pending"
            if cached_state and model_id in cached_state.results:
                status = cached_state.results[model_id].status.value
            
            print(f"{model_id[:45]:<45} {size_gb:>7.1f}G {tp:>3} {status:<15}")
        return
    
    # Initialize or reuse state
    import platform
    import socket
    
    if args.fresh or not cached_state:
        state = ExperimentState(total_models=len(models))
    else:
        state = cached_state
        state.total_models = len(models)
    
    # Update global metadata
    state.vllm_version = get_vllm_version()
    state.python_version = platform.python_version()
    state.hostname = socket.gethostname()
    state.datasets_config = DATASETS.copy()
    
    # Count how many will be skipped
    skip_count = 0
    skip_list_count = 0
    run_count = 0
    for m in models:
        if m['model_id'] in skip_models:
            skip_count += 1
            skip_list_count += 1
            continue
        should_skip, _ = should_skip_model(m['model_id'], state, args.fresh, args.retry_failed)
        if should_skip:
            skip_count += 1
        else:
            run_count += 1
    
    logging.info(f"Models to test: {len(models)} total, {run_count} to run, {skip_count} skipped ({skip_list_count} from skip list, {skip_count - skip_list_count} cached)")
    if args.retry_failed:
        logging.info("Retry mode: will retry previously failed models")
    logging.info(f"Results will be saved to: {RESULTS_DIR}")
    
    # Determine parallel settings
    use_parallel = args.parallel > 1
    if use_parallel:
        logging.info(f"PARALLEL MODE: Using up to {MAX_GPUS} GPUs with dynamic allocation based on model TP size")
    
    # Run experiments
    run_idx = 0
    try:
        if use_parallel:
            # ============================================================
            # PARALLEL EXECUTION MODE WITH DYNAMIC GPU ALLOCATION
            # ============================================================
            # Filter models that need to run (not cached/skipped)
            models_to_run = []
            for idx, model in enumerate(models, args.start + 1):
                model_id = model['model_id']
                if model_id in skip_models:
                    # Handle skip list models
                    result = ModelResult(
                        model_id=model_id,
                        model_name=model.get('model_name', model_id),
                        size_bytes=model.get('size_bytes'),
                        tp_size=model.get('tensor_parallel_size', 1),
                        status=Status.SKIPPED,
                        start_time=datetime.now().isoformat(),
                        end_time=datetime.now().isoformat(),
                        duration_seconds=0.0,
                        vllm_error=f"Marked incomplete in skip list"
                    )
                    update_state_with_result(state, model_id, result)
                    continue
                    
                should_skip, _ = should_skip_model(model_id, state, args.fresh, args.retry_failed)
                if not should_skip:
                    models_to_run.append((idx, model))
            
            # Count models by TP size
            tp_counts = {}
            for _, m in models_to_run:
                tp = m.get('tensor_parallel_size', 1)
                tp_counts[tp] = tp_counts.get(tp, 0) + 1
            logging.info(f"Queued {len(models_to_run)} models for parallel execution: {tp_counts}")
            
            # ================================================================
            # OPTIMIZED GPU SCHEDULER
            # - Supports non-consecutive GPU IDs (vLLM remaps via CUDA_VISIBLE_DEVICES)
            # - Best-fit bin packing: prioritizes jobs that best use available GPUs
            # - Prevents GPU fragmentation by preferring larger jobs when possible
            # - Detects actually free GPUs via nvidia-smi at startup
            # ================================================================
            
            def detect_free_gpus(memory_threshold_mb: int = 500) -> set:
                """
                Detect GPUs that are actually free by checking nvidia-smi.
                A GPU is considered free if its memory usage is below threshold.
                
                Args:
                    memory_threshold_mb: Max memory (MB) to consider GPU as free (default: 500MB)
                
                Returns:
                    Set of free GPU IDs
                """
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10
                    )
                    
                    free_gpus = set()
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 2:
                                gpu_id = int(parts[0])
                                memory_used_mb = float(parts[1])
                                if memory_used_mb < memory_threshold_mb:
                                    free_gpus.add(gpu_id)
                                else:
                                    logging.warning(f"GPU {gpu_id} already in use ({memory_used_mb:.0f}MB), excluding from pool")
                    
                    return free_gpus
                except Exception as e:
                    logging.warning(f"Failed to detect free GPUs via nvidia-smi: {e}, assuming all {MAX_GPUS} GPUs free")
                    return set(range(MAX_GPUS))
            
            # Detect actually free GPUs at startup
            initial_free_gpus = detect_free_gpus()
            if len(initial_free_gpus) < MAX_GPUS:
                logging.info(f"Detected {len(initial_free_gpus)}/{MAX_GPUS} free GPUs: {sorted(initial_free_gpus)}")
            else:
                logging.info(f"All {MAX_GPUS} GPUs are free")
            
            if not initial_free_gpus:
                logging.error("No free GPUs available! Exiting parallel mode.")
                return
            
            available_gpus = initial_free_gpus.copy()
            gpu_lock = threading.Lock()
            base_port = args.port
            port_counter = [0]  # Mutable counter for unique ports
            
            def allocate_gpus(tp_size: int) -> tuple:
                """
                Allocate any N available GPUs for a model.
                vLLM handles non-consecutive GPUs via CUDA_VISIBLE_DEVICES remapping.
                Returns (gpu_ids, port) or (None, None) if not enough GPUs.
                """
                with gpu_lock:
                    if len(available_gpus) < tp_size:
                        return None, None
                    
                    # Simply take the first N available GPUs (sorted for determinism)
                    sorted_gpus = sorted(available_gpus)
                    allocated = sorted_gpus[:tp_size]
                    
                    for gpu in allocated:
                        available_gpus.remove(gpu)
                    
                    port = base_port + port_counter[0]
                    port_counter[0] += 1
                    return allocated, port
            
            def release_gpus(gpu_ids: list):
                """Return GPUs to the pool."""
                with gpu_lock:
                    for gpu in gpu_ids:
                        available_gpus.add(gpu)
            
            def get_available_gpu_count():
                """Thread-safe check of available GPU count."""
                with gpu_lock:
                    return len(available_gpus)
            
            # Dynamic executor with resource-aware scheduling
            max_workers = MAX_GPUS  # Maximum concurrent workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                worker_idx = 0
                model_queue = list(models_to_run)  # Copy to queue
                
                def try_submit_next():
                    """
                    Smart scheduler using best-fit bin packing:
                    1. If all GPUs free, prefer largest TP job (avoid fragmentation)
                    2. Otherwise, find the largest job that fits available GPUs
                    This maximizes GPU utilization and minimizes waiting.
                    """
                    nonlocal worker_idx
                    
                    if not model_queue:
                        return False
                    
                    avail_gpus = get_available_gpu_count()
                    if avail_gpus == 0:
                        return False
                    
                    # Find the best fitting model (largest TP that fits)
                    best_idx = None
                    best_tp = 0
                    
                    for i, (idx, model) in enumerate(model_queue):
                        tp_size = model.get('tensor_parallel_size', 1)
                        if tp_size <= avail_gpus and tp_size > best_tp:
                            best_tp = tp_size
                            best_idx = i
                    
                    if best_idx is None:
                        return False
                    
                    idx, model = model_queue.pop(best_idx)
                    tp_size = model.get('tensor_parallel_size', 1)
                    gpu_ids, port = allocate_gpus(tp_size)
                    
                    if gpu_ids is None:
                        # Shouldn't happen, but handle gracefully
                        model_queue.insert(best_idx, (idx, model))
                        return False
                    
                    worker_idx += 1
                    
                    logging.info(f"[Scheduler] {model['model_id'][:40]} (TP={tp_size}) → GPUs {gpu_ids}, port {port} | Free: {get_available_gpu_count()}/{MAX_GPUS}")
                    
                    future = executor.submit(
                        parallel_worker,
                        model=model,
                        model_dir=args.model_dir,
                        port=port,
                        extra_args=args.extra_args,
                        dry_run=args.dry_run,
                        state=state,
                        idx=worker_idx,
                        total=len(models_to_run),
                        gpu_ids=gpu_ids,
                        skip_models=skip_models,
                        fresh=args.fresh,
                        retry_failed=args.retry_failed
                    )
                    futures[future] = (gpu_ids, port, model['model_id'], tp_size)
                    return True
                
                # Submit initial batch
                while model_queue and try_submit_next():
                    pass
                
                # Process completed jobs and submit new ones
                while futures:
                    # Wait for any job to complete
                    done_futures = []
                    for future in as_completed(futures):
                        done_futures.append(future)
                        break  # Process one at a time to reuse GPUs immediately
                    
                    for future in done_futures:
                        gpu_ids, port, model_id, tp_size = futures.pop(future)
                        
                        try:
                            result_model_id, result, should_update = future.result()
                            if should_update and result:
                                update_state_with_result(state, result_model_id, result)
                                run_idx += 1
                                logging.info(f"Progress: {run_idx}/{len(models_to_run)} | "
                                           f"Success: {state.success} | Failed: {state.failed} | "
                                           f"Free GPUs: {len(available_gpus) + len(gpu_ids)}")
                        except Exception as e:
                            logging.error(f"Worker error for {model_id}: {e}")
                        
                        # Release GPUs back to pool
                        release_gpus(gpu_ids)
                        
                        # Try to submit more jobs with freed GPUs
                        while model_queue and try_submit_next():
                            pass
        
        else:
            # ============================================================
            # SEQUENTIAL EXECUTION MODE (original behavior)
            # ============================================================
            for idx, model in enumerate(models, args.start + 1):
                model_id = model['model_id']
                
                # Always skip models listed in the incomplete skip list
                if model_id in skip_models:
                    logging.info(f"[{idx}/{args.start + len(models)}] Skipping {model_id} (listed in {SKIP_MODELS_FILE})")
                    result = ModelResult(
                        model_id=model_id,
                        model_name=model.get('model_name', model_id),
                        size_bytes=model.get('size_bytes'),
                        tp_size=model.get('tensor_parallel_size', 1),
                        status=Status.SKIPPED,
                        start_time=datetime.now().isoformat(),
                        end_time=datetime.now().isoformat(),
                        duration_seconds=0.0,
                        vllm_error=f"Marked incomplete in {SKIP_MODELS_FILE}"
                    )
                    
                    # Update state counters (handling reruns)
                    old_result = state.results.get(model_id)
                    if old_result:
                        if old_result.status == Status.SUCCESS:
                            state.success = max(0, state.success - 1)
                        elif old_result.status == Status.SKIPPED:
                            state.skipped = max(0, state.skipped - 1)
                        else:
                            state.failed = max(0, state.failed - 1)
                        state.completed = max(0, state.completed - 1)
                    
                    state.results[model_id] = result
                    state.completed += 1
                    state.skipped += 1
                    save_state(state)
                    continue
                
                # Check if should skip (cache check)
                should_skip, skip_reason = should_skip_model(
                    model_id, state, args.fresh, args.retry_failed
                )
                
                if should_skip:
                    logging.debug(f"[{idx}/{args.start + len(models)}] Skipping {model_id} ({skip_reason})")
                    continue
                
                run_idx += 1
                logging.info(f"[{run_idx}/{run_count}] (overall {idx}/{args.start + len(models)})")
                
                # Test the model
                result = test_single_model(
                    model=model,
                    model_dir=args.model_dir,
                    port=args.port,
                    extra_args=args.extra_args,
                    dry_run=args.dry_run,
                    state=state,
                    idx=run_idx,
                    total=run_count
                )
                
                # Update state counts (handle re-runs properly)
                old_result = state.results.get(model_id)
                if old_result:
                    # Decrement old counts
                    if old_result.status == Status.SUCCESS:
                        state.success = max(0, state.success - 1)
                    elif old_result.status == Status.SKIPPED:
                        state.skipped = max(0, state.skipped - 1)
                    else:
                        state.failed = max(0, state.failed - 1)
                    state.completed = max(0, state.completed - 1)
                
                # Update with new result
                state.results[model_id] = result
                state.completed += 1
                
                if result.status == Status.SUCCESS:
                    state.success += 1
                elif result.status == Status.SKIPPED:
                    state.skipped += 1
                else:
                    state.failed += 1
                
                # Save state after each model
                save_state(state)
                
                logging.info(f"Progress: {run_idx}/{run_count} | "
                            f"Success: {state.success} | Failed: {state.failed} | Cached: {skip_count}")
    
    except KeyboardInterrupt:
        logging.warning("Interrupted by user - state saved")
        save_state(state)
    
    finally:
        cleanup_gpu()
    
    # Final summary
    logging.info("=" * 70)
    logging.info("EXPERIMENT COMPLETE")
    logging.info(f"Ran: {run_idx} | Success: {state.success} | "
                f"Failed: {state.failed} | Cached/Skipped: {skip_count}")
    logging.info(f"Results saved to: {RESULTS_FILE}")
    if state.failed > 0:
        logging.info(f"Errors saved to: {ERRORS_FILE}")
        logging.info("Tip: Use --retry-failed to retry failed models")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()

