#!/usr/bin/env python3
"""
Ray Cluster and Experiment Monitor

Monitors Ray cluster resources, running jobs, and experiment progress.
Provides real-time updates on GPU usage, task status, and experiment metrics.

Usage:
    python monitor_ray.py [--ray-address ADDRESS] [--interval SECONDS] [--watch]
    
Examples:
    python monitor_ray.py                          # One-time status check
    python monitor_ray.py --watch                   # Continuous monitoring (5s interval)
    python monitor_ray.py --interval 10             # Custom update interval
    python monitor_ray.py --ray-address 10.0.9.57:6379
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

try:
    import ray
except ImportError:
    print("Error: Ray is not installed. Please install with: pip install ray")
    sys.exit(1)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None

# Configuration
RESULTS_DIR = Path("experiment_results")
RESULTS_FILE = RESULTS_DIR / "results.json"
LOG_FILE = RESULTS_DIR / "experiment.log"
DEFAULT_RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "10.0.9.57:6379")
DEFAULT_INTERVAL = 5


def get_gpu_info_local():
    """Get GPU information from local machine using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpus.append({
                            'index': parts[0],
                            'name': parts[1],
                            'utilization': int(parts[2]) if parts[2].isdigit() else 0,
                            'memory_used': int(parts[3]) if parts[3].isdigit() else 0,
                            'memory_total': int(parts[4]) if parts[4].isdigit() else 0,
                        })
            return gpus
    except Exception as e:
        pass
    return []


def get_gpu_info_remote():
    """Remote function to get GPU info on a Ray node."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpus.append({
                            'index': parts[0],
                            'name': parts[1],
                            'utilization': int(parts[2]) if parts[2].isdigit() else 0,
                            'memory_used': int(parts[3]) if parts[3].isdigit() else 0,
                            'memory_total': int(parts[4]) if parts[4].isdigit() else 0,
                        })
            return gpus
    except Exception as e:
        return []
    return []


def get_gpu_info_all_nodes(ray_address: str):
    """Get GPU information from all nodes in the cluster."""
    all_gpu_info = {}
    
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        nodes = ray.nodes()
        
        # Get local GPU info first
        local_gpus = get_gpu_info_local()
        
        # Try to match local GPUs to a node
        # We'll assign local GPUs to the first node that matches our IP or head node
        local_node_found = False
        for node in nodes:
            node_ip = node.get('NodeManagerAddress', 'Unknown')
            node_id = node.get('NodeID', 'Unknown')[:8]
            is_head = node.get('Resources', {}).get('node:__internal_head__', 0) > 0
            
            # If we have local GPUs and haven't assigned them yet, assign to head node or first node
            if local_gpus and not local_node_found:
                if is_head or node_ip.startswith('10.0.9.57'):
                    all_gpu_info[node_ip] = {
                        'gpus': local_gpus,
                        'node_id': node_id
                    }
                    local_node_found = True
                    continue
            
            # For other nodes, try to get GPU info via remote function
            if node_ip not in all_gpu_info:
                try:
                    # Create remote function
                    remote_get_gpu = ray.remote(get_gpu_info_remote)
                    
                    # Try to schedule on this specific node
                    future = remote_get_gpu.options(
                        num_cpus=0.1,
                        resources={f"node:{node_ip}": 0.1}
                    ).remote()
                    
                    # Get result with timeout
                    gpus = ray.get(future, timeout=8)
                    if gpus:
                        all_gpu_info[node_ip] = {
                            'gpus': gpus,
                            'node_id': node_id
                        }
                except Exception:
                    # If we can't get remote GPU info, that's okay
                    # We'll just show what we have
                    pass
        
        # If we still have local GPUs but didn't assign them, assign to first node
        if local_gpus and not local_node_found and nodes:
            first_node = nodes[0]
            node_ip = first_node.get('NodeManagerAddress', 'Unknown')
            if node_ip not in all_gpu_info:
                all_gpu_info[node_ip] = {
                    'gpus': local_gpus,
                    'node_id': first_node.get('NodeID', 'Unknown')[:8]
                }
        
        ray.shutdown()
    except Exception as e:
        # Fallback: just return local GPU info
        local_gpus = get_gpu_info_local()
        if local_gpus:
            all_gpu_info['local'] = {
                'gpus': local_gpus,
                'node_id': 'local'
            }
    
    return all_gpu_info


def load_experiment_state() -> Optional[Dict]:
    """Load experiment state from results file."""
    if not RESULTS_FILE.exists():
        return None
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except Exception as e:
        return None


def get_experiment_progress(state: Optional[Dict]) -> Dict:
    """Extract progress information from experiment state."""
    if not state:
        return {
            'total': 0,
            'completed': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'progress_pct': 0.0,
            'remaining': 0
        }
    
    total = state.get('total_models', 0)
    completed = state.get('completed', 0)
    success = state.get('success', 0)
    failed = state.get('failed', 0)
    skipped = state.get('skipped', 0)
    
    progress_pct = (completed / total * 100) if total > 0 else 0.0
    remaining = total - completed
    
    return {
        'total': total,
        'completed': completed,
        'success': success,
        'failed': failed,
        'skipped': skipped,
        'progress_pct': progress_pct,
        'remaining': remaining
    }


def get_recent_log_entries(limit: int = 5) -> list:
    """Get recent log entries from experiment log."""
    if not LOG_FILE.exists():
        return []
    
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            # Get last N lines that contain "Progress"
            recent = []
            for line in reversed(lines[-100:]):  # Check last 100 lines
                if 'Progress:' in line or 'INFO' in line:
                    recent.append(line.strip())
                    if len(recent) >= limit:
                        break
            return list(reversed(recent))
    except Exception:
        return []


def find_model_in_outputs(model_id: str, outputs_path: Path) -> Optional[Path]:
    """Find the model's report directory in outputs, trying different name formats."""
    if not outputs_path.exists():
        return None
    
    # Try different name formats
    sanitized_model_id = model_id.replace("/", "_")
    name_variants = [
        sanitized_model_id,
        model_id.replace("/", "__"),
        model_id.split("/")[-1] if "/" in model_id else model_id,
    ]
    
    # Check simple path first (outputs/ModelName) - Direct and fast
    simple_path = outputs_path / sanitized_model_id / "reports" / sanitized_model_id
    if simple_path.exists():
        return simple_path
        
    # Check simple path with quicktest
    simple_path_qt = outputs_path / sanitized_model_id / "reports" / f"{sanitized_model_id}_quicktest"
    if simple_path_qt.exists():
        return simple_path_qt
    
    # Fallback: Search directories sorted by time (for backward compatibility or complex cases)
    try:
        all_dirs = sorted([d for d in outputs_path.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    except Exception:
        return None
    
    for output_dir in all_dirs[:100]:
        dir_name = output_dir.name
        
        # Format 1: New format (outputs/timestamp_model_name)
        # Check if the directory name contains the model name
        if sanitized_model_id in dir_name:
            reports_path = output_dir / "reports" / sanitized_model_id
            if reports_path.exists():
                return reports_path
            
            # Check for quicktest variant
            reports_path = output_dir / "reports" / f"{sanitized_model_id}_quicktest"
            if reports_path.exists():
                return reports_path

        # Format 2: Old format (outputs/timestamp/reports/model_name)
        reports_dir = output_dir / "reports"
        if not reports_dir.exists():
            continue
        
        for name_variant in name_variants:
            model_reports_dir = reports_dir / name_variant
            if model_reports_dir.exists():
                return model_reports_dir
            
            # Also try with _quicktest suffix
            model_reports_dir = reports_dir / f"{name_variant}_quicktest"
            if model_reports_dir.exists():
                return model_reports_dir
        
        # Format 3: Partial match in reports dir
        for model_dir in reports_dir.iterdir():
            if model_dir.is_dir():
                # Check if model_id components match
                model_parts = model_id.lower().replace("/", "_").split("_")
                dir_parts = model_dir.name.lower().split("_")
                # Strict check: all significant parts of model_id should be in dir name
                if any(part in dir_parts for part in model_parts if len(part) > 3):
                    return model_dir
    
    return None


def get_eval_progress_from_output_dir(output_dir: str, model_id: str, datasets: list) -> Dict:
    """Get evaluation progress from evalscope output directory."""
    progress = {
        'datasets_completed': [],
        'datasets_in_progress': [],
        'scores': {},
        'total_datasets': len(datasets) if datasets else 0
    }
    
    try:
        # If output_dir provided, use it; otherwise search in outputs/
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path("./outputs")
        
        if not output_path.exists():
            return progress
        
        # Find the model's report directory
        if output_dir:
            # Direct path provided
            model_name = model_id.replace("/", "_")
            reports_dir = output_path / "reports" / model_name
            if not reports_dir.exists():
                # Try alternative name
                reports_dir = output_path / "reports" / f"{model_name}_quicktest"
        else:
            # Search for model in outputs
            reports_dir = find_model_in_outputs(model_id, output_path)
            if not reports_dir:
                return progress
        
        if reports_dir and reports_dir.exists():
            # Check which datasets have completed reports
            completed_datasets = set()
            for report_file in reports_dir.glob("*.json"):
                dataset_name = report_file.stem
                if dataset_name in datasets:
                    completed_datasets.add(dataset_name)
                    # Try to read score
                    try:
                        with open(report_file) as f:
                            report_data = json.load(f)
                            # Extract score
                            if isinstance(report_data, list) and len(report_data) > 0:
                                score = report_data[0].get('score', report_data[0].get('accuracy'))
                                if score is not None:
                                    progress['scores'][dataset_name] = score
                            elif isinstance(report_data, dict):
                                score = report_data.get('score', report_data.get('accuracy'))
                                if score is not None:
                                    progress['scores'][dataset_name] = score
                    except Exception:
                        pass
            
            progress['datasets_completed'] = list(completed_datasets)
            progress['datasets_in_progress'] = [d for d in datasets if d not in completed_datasets]
        else:
            # Reports dir doesn't exist yet, but evaluation might be in progress
            # Check if there are any log files or temp files indicating active evaluation
            try:
                # Check for recent activity in the output directory
                current_time = time.time()
                for item in output_path.rglob("*"):
                    if item.is_file():
                        try:
                            mtime = item.stat().st_mtime
                            # If file modified in last 10 minutes, evaluation might be active
                            if current_time - mtime < 600:
                                progress['datasets_in_progress'] = datasets
                                break
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass
    
    return progress


def match_vllm_server_to_model(server: Dict, state: Optional[Dict]) -> Optional[str]:
    """Try to match a vLLM server to a model in the experiment state."""
    if not state:
        return None
    
    server_model_name = server.get('model_name', '')
    server_model_path = server.get('model_path', '')
    
    if not server_model_name and not server_model_path:
        return None
    
    results = state.get('results', {})
    
    # Try to match by model name or path
    for model_id, result_data in results.items():
        model_name = result_data.get('model_name', model_id)
        
        # Check if names match
        if server_model_name:
            if (model_id in server_model_name or 
                server_model_name in model_id or 
                model_name in server_model_name or
                server_model_name in model_name):
                return model_id
        
        # Check if path matches
        if server_model_path:
            model_path = result_data.get('model_path', '')
            if model_path and (model_path in server_model_path or server_model_path in model_path):
                return model_id
    
    return None


def get_active_evaluations_from_outputs() -> Dict:
    """Check outputs directory for active evaluations."""
    active_evals = {}
    
    try:
        outputs_path = Path("./outputs")
        if not outputs_path.exists():
            return active_evals
        
        # Get latest output directory
        dirs = sorted([d for d in outputs_path.iterdir() if d.is_dir()], reverse=True)
        if not dirs:
            return active_evals
        
        latest_dir = dirs[0]
        reports_dir = latest_dir / "reports"
        
        if reports_dir.exists():
            # Check for recent report files (modified in last hour)
            current_time = time.time()
            for model_dir in reports_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name.replace("_", "/")
                    # Check for recently modified files
                    recent_files = []
                    for report_file in model_dir.glob("*.json"):
                        try:
                            mtime = report_file.stat().st_mtime
                            if current_time - mtime < 3600:  # Modified in last hour
                                recent_files.append(report_file)
                        except Exception:
                            pass
                    
                    if recent_files:
                        # This model might be actively evaluating
                        active_evals[model_name] = {
                            'output_dir': str(latest_dir),
                            'recent_files': len(recent_files)
                        }
    except Exception:
        pass
    
    return active_evals


def get_currently_running_models(state: Optional[Dict], vllm_servers: list) -> list:
    """Extract information about models currently being processed."""
    running_models = []
    datasets = state.get('datasets_config', ["mmlu", "mmlu_pro", "math_500", "gsm8k", "gpqa_diamond"]) if state else ["mmlu", "mmlu_pro", "math_500", "gsm8k", "gpqa_diamond"]
    
    # First, check vLLM servers and try to match them to models
    for server in vllm_servers:
        model_id = match_vllm_server_to_model(server, state)
        
        if model_id and state:
            result_data = state.get('results', {}).get(model_id)
            if result_data:
                # Model found in state
                model_info = {
                    'model_id': model_id,
                    'model_name': result_data.get('model_name', model_id),
                    'tp_size': result_data.get('tp_size', server.get('tp_size', 1)),
                    'start_time': result_data.get('start_time'),
                    'status': result_data.get('status', 'running'),
                    'vllm_pid': server.get('pid'),
                    'port': server.get('port'),
                    'host': server.get('host'),
                    'eval_progress': None
                }
            else:
                # Server running but not in state (might be manual/test server)
                model_info = {
                    'model_id': model_id,
                    'model_name': server.get('model_name', model_id),
                    'tp_size': server.get('tp_size', 1),
                    'start_time': None,
                    'status': 'serving',
                    'vllm_pid': server.get('pid'),
                    'port': server.get('port'),
                    'host': server.get('host'),
                    'eval_progress': None
                }
        else:
            # Server not matched to state - might be a test/manual server
            model_info = {
                'model_id': server.get('model_name', 'Unknown'),
                'model_name': server.get('model_name', 'Unknown'),
                'tp_size': server.get('tp_size', 1),
                'start_time': None,
                'status': 'serving',
                'vllm_pid': server.get('pid'),
                'port': server.get('port'),
                'host': server.get('host'),
                'eval_progress': None
            }
        
            # Get evaluation progress if available
            if state and model_id:
                result_data = state.get('results', {}).get(model_id)
                if result_data:
                    eval_output_dir = result_data.get('eval_output_dir')
                    if eval_output_dir:
                        model_info['eval_progress'] = get_eval_progress_from_output_dir(
                            eval_output_dir, model_id, datasets
                        )
        
        # Try to find latest output directory if no progress found
        if not model_info.get('eval_progress') or len(model_info.get('eval_progress', {}).get('datasets_completed', [])) == 0:
            try:
                outputs_path = Path("./outputs")
                if outputs_path.exists():
                    dirs = sorted([d for d in outputs_path.iterdir() if d.is_dir()], reverse=True)
                    if dirs:
                        # Try latest directory
                        latest_progress = get_eval_progress_from_output_dir(
                            None, model_info['model_id'], datasets
                        )
                        # Only use if we found completed datasets
                        if latest_progress.get('datasets_completed'):
                            model_info['eval_progress'] = latest_progress
            except Exception:
                pass
        
        # Check if datasets_tested is available
        if state and model_id:
            result_data = state.get('results', {}).get(model_id)
            if result_data:
                datasets_tested = result_data.get('datasets_tested', [])
                if datasets_tested:
                    if not model_info['eval_progress']:
                        model_info['eval_progress'] = {
                            'datasets_completed': [],
                            'datasets_in_progress': datasets_tested,
                            'scores': {},
                            'total_datasets': len(datasets_tested)
                        }
                    else:
                        # Only add datasets that aren't already completed
                        completed = model_info['eval_progress'].get('datasets_completed', [])
                        model_info['eval_progress']['datasets_in_progress'] = [
                            d for d in datasets_tested if d not in completed
                        ]
        
        running_models.append(model_info)
    
    # Also check for models marked as running in state
    if state:
        results = state.get('results', {})
        for model_id, result_data in results.items():
            status = result_data.get('status', '')
            start_time = result_data.get('start_time')
            end_time = result_data.get('end_time')
            
            # If started but not ended and status is running, add it
            if status == 'running' or (start_time and not end_time and status not in ['success', 'failed', 'skipped']):
                # Check if we already added this model from vLLM server matching
                if not any(m.get('model_id') == model_id for m in running_models):
                    model_info = {
                        'model_id': model_id,
                        'model_name': result_data.get('model_name', model_id),
                        'tp_size': result_data.get('tp_size', 1),
                        'start_time': start_time,
                        'status': status,
                        'vllm_pid': result_data.get('vllm_pid'),
                        'port': None,
                        'host': None,
                        'eval_progress': None
                    }
                    
                    # Get evaluation progress
                    eval_output_dir = result_data.get('eval_output_dir')
                    if eval_output_dir:
                        model_info['eval_progress'] = get_eval_progress_from_output_dir(
                            eval_output_dir, model_id, datasets
                        )
                    
                    running_models.append(model_info)
    
    return running_models


def parse_vllm_command(cmd: str) -> Optional[Dict]:
    """Parse vLLM command line to extract model, port, TP size, etc."""
    try:
        info = {
            'model_path': None,
            'model_name': None,
            'port': None,
            'tp_size': None,
            'host': None
        }
        
        parts = cmd.split()
        i = 0
        while i < len(parts):
            # Check if this is vllm serve (could be full path like /path/to/vllm)
            # Look for 'vllm' in the part and 'serve' as the next part
            if 'vllm' in parts[i].lower() and i + 1 < len(parts) and parts[i+1] == 'serve':
                i += 2
                # Next should be model path
                if i < len(parts):
                    model_path = parts[i]
                    info['model_path'] = model_path
                    # Extract model name from path
                    if 'models--' in model_path:
                        # HuggingFace cache format: models--org--model-name/snapshots/...
                        # Extract org and model name
                        model_part = model_path.split('models--')[-1].split('/')[0]
                        if '--' in model_part:
                            parts_model = model_part.split('--')
                            if len(parts_model) >= 2:
                                org = parts_model[0]
                                name = parts_model[1]
                                info['model_name'] = f"{org}/{name}"
                            else:
                                info['model_name'] = model_part.replace('--', '/')
                        else:
                            info['model_name'] = model_part
                    else:
                        # Try to extract from path
                        path_parts = model_path.rstrip('/').split('/')
                        if len(path_parts) >= 2:
                            info['model_name'] = '/'.join(path_parts[-2:])
                        else:
                            info['model_name'] = path_parts[-1]
            elif parts[i] == '--tensor-parallel-size' and i + 1 < len(parts):
                try:
                    info['tp_size'] = int(parts[i+1])
                except ValueError:
                    pass
                i += 1
            elif parts[i] == '--port' and i + 1 < len(parts):
                try:
                    info['port'] = int(parts[i+1])
                except ValueError:
                    pass
                i += 1
            elif parts[i] == '--host' and i + 1 < len(parts):
                info['host'] = parts[i+1]
                i += 1
            i += 1
        
        return info if info['model_path'] else None
    except Exception:
        return None


def check_vllm_health(host: str, port: int, timeout: int = 2) -> Dict:
    """Check vLLM server health and get model info."""
    health_info = {
        'healthy': False,
        'model_info': None,
        'error': None
    }
    
    if not HAS_REQUESTS:
        health_info['error'] = "requests library not available"
        return health_info
    
    try:
        # Check health endpoint
        health_url = f"http://{host}:{port}/health"
        resp = requests.get(health_url, timeout=timeout)
        if resp.status_code == 200:
            health_info['healthy'] = True
            
            # Try to get model info from /v1/models
            try:
                models_url = f"http://{host}:{port}/v1/models"
                models_resp = requests.get(models_url, timeout=timeout)
                if models_resp.status_code == 200:
                    models_data = models_resp.json()
                    if 'data' in models_data and len(models_data['data']) > 0:
                        health_info['model_info'] = models_data['data'][0]
            except Exception:
                pass
        else:
            health_info['error'] = f"HTTP {resp.status_code}"
    except requests.exceptions.ConnectionError:
        health_info['error'] = "Connection refused"
    except requests.exceptions.Timeout:
        health_info['error'] = "Timeout"
    except Exception as e:
        health_info['error'] = str(e)[:50]
    
    return health_info


def get_vllm_servers():
    """Get information about running vLLM servers."""
    vllm_servers = []
    
    try:
        # First, try to find vLLM processes by checking ps output
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                # Check for vllm serve in the command - look for actual vllm binary path
                if '/vllm' in line or ('vllm' in line.lower() and 'serve' in line.lower()):
                    parts = line.split()
                    if len(parts) >= 11:
                        pid = parts[1]
                        cpu = parts[2]
                        mem = parts[3]
                        # Command starts at index 10
                        cmd = ' '.join(parts[10:])
                        
                        # Only process if it looks like a real vLLM command (not a shell wrapper)
                        if '/vllm' in cmd or ('vllm' in cmd and 'serve' in cmd):
                            # Parse vLLM command
                            vllm_info = parse_vllm_command(cmd)
                            if vllm_info:
                                vllm_info['pid'] = pid
                                vllm_info['cpu'] = cpu
                                vllm_info['mem'] = mem
                                vllm_info['cmd'] = cmd
                                
                                # Check health
                                host = vllm_info.get('host', 'localhost')
                                port = vllm_info.get('port')
                                if port:
                                    health = check_vllm_health(host, port)
                                    vllm_info['health'] = health
                                
                                vllm_servers.append(vllm_info)
        
        # Also try to find vLLM servers by checking ports and health endpoints
        # The experiment uses ports 8010-8109 (find_free_port with max_attempts=100)
        # Get list of cluster nodes to check all nodes, not just localhost
        cluster_nodes = []
        try:
            import ray
            ray.init(address='10.0.9.57:6379', ignore_reinit_error=True)
            nodes = ray.nodes()
            for node in nodes:
                node_ip = node.get('NodeManagerAddress', 'Unknown')
                if node_ip not in ['Unknown', '127.0.0.1', 'localhost']:
                    cluster_nodes.append(node_ip)
            ray.shutdown()
        except Exception:
            pass
        
        # If no cluster nodes found, just check localhost
        if not cluster_nodes:
            cluster_nodes = ['localhost']
        
        # Check ports on all cluster nodes
        for node_ip in cluster_nodes:
            for port in range(8010, 8110):
                # Skip if we already found this port+host combination
                if any(s.get('port') == port and s.get('host') == node_ip for s in vllm_servers):
                    continue
                try:
                    health = check_vllm_health(node_ip, port)
                    if health.get('healthy'):
                        # Found a healthy vLLM server
                        vllm_info = {
                            'port': port,
                            'host': node_ip,
                            'health': health,
                            'pid': 'N/A',
                            'cpu': 'N/A',
                            'mem': 'N/A',
                            'tp_size': None
                        }
                        # Try to extract model info from health response
                        if health.get('model_info'):
                            model_id = health['model_info'].get('id', '')
                            vllm_info['model_path'] = model_id
                            # Extract model name
                            if 'models--' in model_id:
                                model_part = model_id.split('models--')[-1].split('/')[0]
                                if '--' in model_part:
                                    parts_model = model_part.split('--')
                                    if len(parts_model) >= 2:
                                        vllm_info['model_name'] = f"{parts_model[0]}/{parts_model[1]}"
                        vllm_servers.append(vllm_info)
                except Exception:
                    pass
    except Exception as e:
        pass
    
    return vllm_servers


def get_ray_processes():
    """Get information about Ray-related processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            processes = []
            for line in result.stdout.split('\n'):
                if 'ray::ray_worker' in line or 'run_experiment.py' in line:
                    # Skip vllm serve processes as they're handled separately
                    if 'vllm serve' not in line:
                        parts = line.split()
                        if len(parts) >= 11:
                            pid = parts[1]
                            cpu = parts[2]
                            mem = parts[3]
                            cmd = ' '.join(parts[10:])
                            processes.append({
                                'pid': pid,
                                'cpu': cpu,
                                'mem': mem,
                                'cmd': cmd[:80]  # Truncate long commands
                            })
            return processes
    except Exception:
        pass
    return []


def get_ray_jobs_and_tasks(ray_address: str):
    """Get information about running Ray jobs and tasks."""
    jobs_info = {
        'running_tasks': [],
        'pending_tasks': [],
        'actors': [],
        'jobs': []
    }
    
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        
        # Try to get task information using Ray's internal APIs
        try:
            # Get actors
            import ray._private.state as state_module
            actors = state_module.actors()
            for actor_id, actor_info in list(actors.items())[:20]:
                state_str = actor_info.get('state', 'unknown')
                if state_str in ['ALIVE', 'RUNNING']:
                    jobs_info['actors'].append({
                        'id': actor_id[:16],
                        'name': actor_info.get('name', 'unnamed'),
                        'class': actor_info.get('class_name', 'unknown'),
                        'state': state_str,
                        'pid': actor_info.get('pid', 'N/A')
                    })
        except Exception:
            pass
        
        # Try to get tasks
        try:
            import ray._private.state as state_module
            tasks = state_module.tasks()
            for task_id, task_info in list(tasks.items())[:50]:
                state_str = task_info.get('state', 'unknown')
                func_name = task_info.get('function_name', 'unknown')
                name = task_info.get('name', 'unnamed')
                
                task_data = {
                    'id': task_id[:16],
                    'name': name or 'unnamed',
                    'function': func_name,
                    'state': state_str
                }
                
                if state_str == 'RUNNING':
                    jobs_info['running_tasks'].append(task_data)
                elif state_str in ['PENDING', 'WAITING']:
                    jobs_info['pending_tasks'].append(task_data)
        except Exception:
            pass
        
        # Try to get job information
        try:
            import ray._private.state as state_module
            jobs = state_module.jobs()
            for job in jobs:
                entrypoint = job.get('Entrypoint', 'N/A')
                # Filter out monitor script instances
                if 'monitor_ray.py' not in entrypoint:
                    jobs_info['jobs'].append({
                        'job_id': job.get('JobId', 'Unknown')[:16],
                        'status': job.get('Status', 'Unknown'),
                        'entrypoint': entrypoint,
                        'start_time': job.get('StartTime', 'N/A')
                    })
        except Exception:
            pass
        
        ray.shutdown()
    except Exception as e:
        pass
    
    return jobs_info


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{title}")
    print("-" * 80)


def monitor_ray_cluster(ray_address: str):
    """Monitor Ray cluster status."""
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
    except Exception as e:
        print(f"Error connecting to Ray cluster at {ray_address}: {e}")
        return None
    
    try:
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        nodes = ray.nodes()
        
        return {
            'cluster_resources': cluster_resources,
            'available_resources': available_resources,
            'nodes': nodes,
            'connected': True
        }
    except Exception as e:
        print(f"Error getting Ray cluster info: {e}")
        return None
    finally:
        try:
            ray.shutdown()
        except:
            pass


def display_status(ray_address: str, watch: bool = False, interval: int = DEFAULT_INTERVAL):
    """Display current status."""
    while True:
        # Clear screen if watching
        if watch:
            os.system('clear' if os.name != 'nt' else 'cls')
        
        print_header(f"RAY CLUSTER & EXPERIMENT MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ray Cluster Status
        print_section("RAY CLUSTER STATUS")
        cluster_info = monitor_ray_cluster(ray_address)
        
        if cluster_info and cluster_info.get('connected'):
            cr = cluster_info['cluster_resources']
            ar = cluster_info['available_resources']
            nodes = cluster_info['nodes']
            
            total_gpus = cr.get('GPU', 0)
            available_gpus = ar.get('GPU', 0)
            used_gpus = total_gpus - available_gpus
            
            total_cpus = cr.get('CPU', 0)
            available_cpus = ar.get('CPU', 0)
            used_cpus = total_cpus - available_cpus
            
            print(f"Cluster Address: {ray_address}")
            print(f"Nodes: {len(nodes)}")
            for node in nodes:
                node_id = node.get('NodeID', 'Unknown')[:8]
                node_ip = node.get('NodeManagerAddress', 'Unknown')
                alive = node.get('Alive', False)
                resources = node.get('Resources', {})
                gpu_count = resources.get('GPU', 0)
                print(f"  • Node {node_id}... ({node_ip}): {gpu_count:.0f} GPUs, {'✓ Alive' if alive else '✗ Dead'}")
            
            print(f"\nResource Usage:")
            print(f"  GPUs: {used_gpus:.0f} / {total_gpus:.0f} in use ({used_gpus/total_gpus*100:.1f}%)")
            print(f"  CPUs: {used_cpus:.0f} / {total_cpus:.0f} in use ({used_cpus/total_cpus*100:.1f}%)")
        else:
            print(f"⚠ Could not connect to Ray cluster at {ray_address}")
        
        # GPU Information from all nodes
        print_section("GPU STATUS (ALL NODES)")
        all_gpu_info = get_gpu_info_all_nodes(ray_address)
        
        if all_gpu_info:
            for node_ip, node_data in all_gpu_info.items():
                node_id = node_data.get('node_id', 'Unknown')
                gpus = node_data.get('gpus', [])
                if gpus:
                    print(f"\n  Node {node_id}... ({node_ip}):")
                    for gpu in gpus:
                        mem_pct = (gpu['memory_used'] / gpu['memory_total'] * 100) if gpu['memory_total'] > 0 else 0
                        status = "🟢 Active" if gpu['utilization'] > 10 or gpu['memory_used'] > 1000 else "⚪ Idle"
                        print(f"    GPU {gpu['index']}: {status} | "
                              f"Util: {gpu['utilization']:3d}% | "
                              f"Mem: {gpu['memory_used']:6d}MB / {gpu['memory_total']:6d}MB ({mem_pct:.1f}%)")
        else:
            # Fallback to local GPU info
            local_gpus = get_gpu_info_local()
            if local_gpus:
                print("  (Local node only - could not query remote nodes)")
                for gpu in local_gpus:
                    mem_pct = (gpu['memory_used'] / gpu['memory_total'] * 100) if gpu['memory_total'] > 0 else 0
                    status = "🟢 Active" if gpu['utilization'] > 10 or gpu['memory_used'] > 1000 else "⚪ Idle"
                    print(f"  GPU {gpu['index']}: {status} | "
                          f"Util: {gpu['utilization']:3d}% | "
                          f"Mem: {gpu['memory_used']:6d}MB / {gpu['memory_total']:6d}MB ({mem_pct:.1f}%)")
            else:
                print("  ⚠ Could not get GPU information (nvidia-smi not available)")
        
        # Load experiment state (used in multiple sections)
        state = load_experiment_state()
        
        # vLLM Servers
        print_section("VLLM SERVERS")
        vllm_servers = get_vllm_servers()
        if vllm_servers:
            # Remove duplicates - normalize host (0.0.0.0, localhost, 127.0.0.1, and node IP are the same)
            # Also deduplicate by port + model_name
            seen = set()
            unique_servers = []
            for server in vllm_servers:
                host = server.get('host', 'localhost')
                port = server.get('port')
                model_name = server.get('model_name', 'Unknown')
                
                # Normalize host for deduplication
                if host in ['0.0.0.0', 'localhost', '127.0.0.1']:
                    # Check if we can determine the actual node IP
                    if cluster_info and cluster_info.get('connected'):
                        nodes = cluster_info['nodes']
                        for node in nodes:
                            node_ip = node.get('NodeManagerAddress', 'Unknown')
                            if node_ip.startswith('10.0.9.57') or node.get('Resources', {}).get('node:__internal_head__', 0) > 0:
                                host = node_ip
                                break
                
                key = (host, port, model_name)
                if key not in seen:
                    seen.add(key)
                    server['host'] = host  # Update to normalized host
                    unique_servers.append(server)
            
            print(f"  Found {len(unique_servers)} vLLM server(s):\n")
            for server in unique_servers:
                model_name = server.get('model_name', 'Unknown')
                if len(model_name) > 45:
                    model_name = model_name[:42] + '...'
                
                port = server.get('port', 'N/A')
                tp_size = server.get('tp_size', 'N/A')
                pid = server.get('pid', 'N/A')
                cpu = server.get('cpu', 'N/A')
                mem = server.get('mem', 'N/A')
                host = server.get('host', 'localhost')
                health = server.get('health', {})
                healthy = health.get('healthy', False)
                status_icon = "🟢" if healthy else "🔴"
                
                # Determine which node this is on
                node_info = ""
                if cluster_info and cluster_info.get('connected'):
                    nodes = cluster_info['nodes']
                    for node in nodes:
                        node_ip = node.get('NodeManagerAddress', 'Unknown')
                        if host == node_ip or (host in ['localhost', '0.0.0.0', '127.0.0.1'] and node_ip.startswith('10.0.9.57')):
                            node_id = node.get('NodeID', 'Unknown')[:8]
                            node_info = f" (Node {node_id}...)"
                            break
                
                print(f"  {status_icon} {model_name}{node_info}")
                print(f"    Host: {host} | Port: {port}", end="")
                if tp_size and tp_size != 'N/A':
                    print(f" | TP Size: {tp_size}", end="")
                if pid != 'N/A':
                    print(f" | PID: {pid}", end="")
                if cpu != 'N/A' and mem != 'N/A':
                    print(f" | CPU: {cpu}% | MEM: {mem}%", end="")
                print()
                
                if health.get('model_info'):
                    model_info = health['model_info']
                    model_id = model_info.get('id', 'N/A')
                    max_len = model_info.get('max_model_len')
                    if max_len:
                        print(f"    Max Context Length: {max_len:,} tokens")
                
                if not healthy and health.get('error'):
                    print(f"    Status: ⚠ {health['error']}")
        else:
            print("  No vLLM servers currently running")
        
        # Currently Running Models with Evaluation Progress
        print_section("CURRENTLY RUNNING MODELS & EVALUATION PROGRESS")
        running_models = get_currently_running_models(state, vllm_servers)
        
        # Remove duplicates by model_id + host + port
        seen = set()
        unique_models = []
        for model in running_models:
            key = (model.get('model_id'), model.get('host'), model.get('port'))
            if key not in seen:
                seen.add(key)
                unique_models.append(model)
        
        if unique_models:
            for model in unique_models[:5]:  # Show up to 5
                model_name = model['model_name']
                if len(model_name) > 50:
                    model_name = model_name[:47] + '...'
                
                tp_size = model.get('tp_size')
                tp_str = f"TP={tp_size}" if tp_size and tp_size != 'N/A' else ""
                print(f"  • {model_name} {tp_str}".strip())
                
                # Show server info if matched
                if model.get('port'):
                    print(f"    Server: {model.get('host', 'localhost')}:{model['port']}")
                
                # Show runtime
                if model.get('start_time'):
                    try:
                        start = datetime.fromisoformat(model['start_time'].replace('Z', '+00:00'))
                        elapsed = datetime.now() - start.replace(tzinfo=None)
                        print(f"    Running for: {format_duration(elapsed.total_seconds())}")
                    except:
                        pass
                
                # Show evaluation progress
                eval_progress = model.get('eval_progress')
                if eval_progress:
                    total = eval_progress.get('total_datasets', 0)
                    completed = len(eval_progress.get('datasets_completed', []))
                    in_progress = eval_progress.get('datasets_in_progress', [])
                    
                    if total > 0:
                        progress_pct = (completed / total * 100) if total > 0 else 0
                        bar_width = 30
                        filled = int(bar_width * progress_pct / 100)
                        bar = '█' * filled + '░' * (bar_width - filled)
                        print(f"    Evaluation: [{bar}] {completed}/{total} datasets ({progress_pct:.0f}%)")
                        
                        # Show completed datasets with scores
                        completed_ds = eval_progress.get('datasets_completed', [])
                        if completed_ds:
                            scores = eval_progress.get('scores', {})
                            score_strs = []
                            for ds in completed_ds[:3]:  # Show up to 3
                                score = scores.get(ds)
                                if score is not None:
                                    score_strs.append(f"{ds}: {score:.2f}")
                                else:
                                    score_strs.append(ds)
                            print(f"      ✓ Completed: {', '.join(score_strs)}")
                        
                        # Show in-progress datasets
                        if in_progress:
                            print(f"      ⟳ In Progress: {', '.join(in_progress[:3])}")
                            if len(in_progress) > 3:
                                print(f"      ... and {len(in_progress) - 3} more")
                    else:
                        print(f"    Status: {model.get('status', 'serving')} (no evaluation data)")
                else:
                    status = model.get('status', 'serving')
                    # Check if it has been running for a while without progress
                    is_stale = False
                    if model.get('start_time'):
                        try:
                            start = datetime.fromisoformat(model['start_time'].replace('Z', '+00:00'))
                            elapsed = (datetime.now() - start.replace(tzinfo=None)).total_seconds()
                            if elapsed > 1800: # 30 minutes without output
                                is_stale = True
                        except:
                            pass
                    
                    if status == 'serving':
                        msg = "vLLM server running"
                        if is_stale:
                            msg += " (no output found - possibly stuck?)"
                        else:
                            msg += " (initializing or waiting for tests)"
                        print(f"    Status: {msg}")
                    else:
                        print(f"    Status: {status}")
        else:
            print("  No models currently running (checking experiment state)")
        
        # Ray Jobs and Tasks
        print_section("RAY JOBS & TASKS")
        jobs_info = get_ray_jobs_and_tasks(ray_address)
        
        if jobs_info['running_tasks']:
            print(f"\n  Running Tasks ({len(jobs_info['running_tasks'])}):")
            for task in jobs_info['running_tasks'][:10]:  # Show up to 10
                print(f"    • {task['name'] or 'unnamed'} ({task['function']}) - {task['id']}")
        else:
            print("  No running tasks found in Ray state")
        
        if jobs_info['pending_tasks']:
            print(f"\n  Pending Tasks ({len(jobs_info['pending_tasks'])}):")
            for task in jobs_info['pending_tasks'][:5]:  # Show up to 5
                print(f"    • {task['name'] or 'unnamed'} ({task['function']}) - {task['id']}")
        
        if jobs_info['actors']:
            print(f"\n  Active Actors ({len(jobs_info['actors'])}):")
            for actor in jobs_info['actors'][:10]:  # Show up to 10
                print(f"    • {actor['name'] or 'unnamed'} ({actor['class']}) - PID: {actor['pid']} - {actor['id']}")
        
        if jobs_info['jobs']:
            print(f"\n  Active Jobs ({len(jobs_info['jobs'])}):")
            for job in jobs_info['jobs'][:5]:  # Show up to 5 most recent
                entrypoint = job['entrypoint']
                # Truncate long paths
                if len(entrypoint) > 60:
                    entrypoint = '...' + entrypoint[-57:]
                print(f"    • Job {job['job_id']}: {job['status']}")
                print(f"      {entrypoint}")
        else:
            print("  No active jobs found (excluding monitor instances)")
        
        # Experiment Progress
        print_section("EXPERIMENT PROGRESS")
        progress = get_experiment_progress(state)
        
        if progress['total'] > 0:
            bar_width = 50
            filled = int(bar_width * progress['progress_pct'] / 100)
            bar = '█' * filled + '░' * (bar_width - filled)
            
            print(f"  Total Models: {progress['total']}")
            print(f"  Progress: [{bar}] {progress['progress_pct']:.1f}%")
            print(f"  Completed: {progress['completed']} | "
                  f"Success: {progress['success']} | "
                  f"Failed: {progress['failed']} | "
                  f"Skipped: {progress['skipped']}")
            print(f"  Remaining: {progress['remaining']} models")
            
            if state and 'started_at' in state:
                try:
                    start_time = datetime.fromisoformat(state['started_at'].replace('Z', '+00:00'))
                    elapsed = datetime.now() - start_time.replace(tzinfo=None)
                    if progress['completed'] > 0:
                        avg_time = elapsed.total_seconds() / progress['completed']
                        eta_seconds = avg_time * progress['remaining']
                        eta = format_duration(eta_seconds)
                        print(f"  Elapsed: {format_duration(elapsed.total_seconds())} | "
                              f"ETA: ~{eta}")
                except:
                    pass
        else:
            print("  ⚠ No experiment state found. Is the experiment running?")
        
        # Recent Activity
        print_section("RECENT ACTIVITY")
        recent_logs = get_recent_log_entries(limit=3)
        if recent_logs:
            for log in recent_logs:
                # Extract timestamp and message
                if '|' in log:
                    parts = log.split('|', 2)
                    if len(parts) >= 3:
                        timestamp = parts[0].strip()
                        level = parts[1].strip()
                        message = parts[2].strip()
                        print(f"  [{timestamp}] {message}")
                else:
                    print(f"  {log[:100]}")
        else:
            print("  No recent activity logged")
        
        # Running Processes
        print_section("RAY PROCESSES")
        processes = get_ray_processes()
        if processes:
            for proc in processes:
                print(f"  PID {proc['pid']:>8} | CPU: {proc['cpu']:>5}% | MEM: {proc['mem']:>5}% | {proc['cmd']}")
        else:
            print("  No Ray worker processes found")
        
        # Footer
        if watch:
            print(f"\n{'='*80}")
            print(f"Monitoring... (Press Ctrl+C to stop, updates every {interval}s)")
            time.sleep(interval)
        else:
            print(f"\n{'='*80}")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Ray cluster and experiment progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--ray-address",
        default=DEFAULT_RAY_ADDRESS,
        help=f"Ray cluster address (default: {DEFAULT_RAY_ADDRESS})"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Update interval in seconds when watching (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor and update display"
    )
    
    args = parser.parse_args()
    
    try:
        display_status(args.ray_address, watch=args.watch, interval=args.interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
