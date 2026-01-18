#!/usr/bin/env python3
"""
Quick checker for locally cached HuggingFace models listed in a JSON file.

It follows the same path rules as `run_experiment.py`:
- `model_id` like `org/model`  -> directory name `models--org--model`
- base directory defaults to `/mnt/baai_cp_perf/hf_models` (or $MODEL_DIR)
- inside that directory, it resolves HF cache layout:
    models--org--model/
    ├── blobs/
    ├── refs/main              # commit hash
    └── snapshots/<hash>/      # actual model files (config.json, weights, etc.)

For each model it will:
- locate the base directory
- resolve the snapshot directory (if using HF cache layout)
- check for `config.json` or `params.json`
- check that at least one weight file exists (*.bin, *.safetensors, model-*-of-*.bin)

It prints a human-readable summary and can optionally dump a JSON report.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


DEFAULT_JSON = "found_models_full.json"
DEFAULT_MODEL_DIR = "/mnt/baai_cp_perf/hf_models"


def model_id_to_dir(model_id: str) -> str:
    """Convert model_id (org/model) to directory format (models--org--model)."""
    return f"models--{model_id.replace('/', '--')}"


def resolve_hf_model_path(base_path: Path) -> Path:
    """
    Resolve the actual model path from HuggingFace cache directory structure.

    This mirrors the logic in `run_experiment.py`.
    """
    snapshots_dir = base_path / "snapshots"

    # Not HF cache format, return as-is (might be direct model path)
    if not snapshots_dir.exists():
        return base_path

    # Try to read the commit hash from refs/main
    refs_main = base_path / "refs" / "main"
    if refs_main.exists():
        try:
            commit_hash = refs_main.read_text().strip()
        except OSError:
            commit_hash = ""
        if commit_hash:
            snapshot_path = snapshots_dir / commit_hash
            if snapshot_path.exists() and (
                (snapshot_path / "config.json").exists()
                or (snapshot_path / "params.json").exists()
            ):
                return snapshot_path

    # Fallback: find any snapshot directory with config or params
    for snapshot in snapshots_dir.iterdir():
        if snapshot.is_dir():
            if (snapshot / "config.json").exists() or (snapshot / "params.json").exists():
                return snapshot

    # No valid snapshot found, return base path
    return base_path


def load_models(json_file: str) -> List[Dict[str, Any]]:
    with open(json_file) as f:
        return json.load(f)


def has_weight_files(model_path: Path) -> bool:
    """
    Heuristic check for presence of *any* weight files in the resolved model directory.
    NOTE: This does not guarantee completeness for sharded checkpoints.
    """
    if not model_path.exists():
        return False

    exts = {".bin", ".safetensors"}
    for p in model_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix in exts:
            return True
        name = p.name
        if name.startswith("pytorch_model-") and name.endswith(".bin"):
            return True
        if name.startswith("model-") and name.endswith(".safetensors"):
            return True
    return False


def check_single_model(
    model: Dict[str, Any],
    base_dir: Path,
) -> Dict[str, Any]:
    model_id = model.get("model_id") or model.get("model_name")
    dir_name = model_id_to_dir(model_id)
    base_model_path = base_dir / dir_name

    info: Dict[str, Any] = {
        "model_id": model_id,
        "model_name": model.get("model_name", model_id),
        "base_model_path": str(base_model_path),
        "resolved_model_path": None,
        "exists": base_model_path.exists(),
        "config_found": False,
        "weights_found": False,
        "status": "",
        "reason": "",
    }

    if not base_model_path.exists():
        info["status"] = "missing"
        info["reason"] = "base directory not found"
        return info

    model_path = resolve_hf_model_path(base_model_path)
    info["resolved_model_path"] = str(model_path)

    config_exists = (model_path / "config.json").exists() or (model_path / "params.json").exists()
    weights_exist = has_weight_files(model_path)
    info["config_found"] = bool(config_exists)
    info["weights_found"] = bool(weights_exist)

    # Default classification based on simple presence
    if not config_exists and not weights_exist:
        info["status"] = "incomplete"
        info["reason"] = "no config or weights found in resolved directory"
        return info

    if config_exists and not weights_exist:
        info["status"] = "incomplete"
        info["reason"] = "config found but no weight files detected"
        return info

    if not config_exists and weights_exist:
        info["status"] = "incomplete"
        info["reason"] = "weights found but no config.json/params.json"
        return info

    # At this point, we have both config and at least one weight file.
    # For sharded checkpoints like model-00001-of-00014.safetensors, verify all shards exist.
    shard_pattern_safetensors = re.compile(r"model-(\d+)-of-(\d+)\.safetensors$")
    shard_pattern_bin = re.compile(r"pytorch_model-(\d+)-of-(\d+)\.bin$")

    shard_info = []
    try:
        for p in model_path.iterdir():
            if not p.is_file():
                continue
            m = shard_pattern_safetensors.match(p.name)
            if m:
                idx = int(m.group(1))
                total = int(m.group(2))
                shard_info.append(("safetensors", idx, total))
                continue
            m = shard_pattern_bin.match(p.name)
            if m:
                idx = int(m.group(1))
                total = int(m.group(2))
                shard_info.append(("bin", idx, total))
    except FileNotFoundError:
        # Directory disappeared between checks; treat as incomplete.
        info["status"] = "incomplete"
        info["reason"] = "model directory disappeared during inspection"
        return info

    if shard_info:
        # Group by format and expected total
        by_key = {}
        for fmt, idx, total in shard_info:
            key = (fmt, total)
            by_key.setdefault(key, set()).add(idx)

        # Choose the group with the largest total shard count as the primary checkpoint
        fmt, total = max(by_key.keys(), key=lambda k: k[1])
        present = by_key[(fmt, total)]
        expected = set(range(1, total + 1))
        missing = sorted(expected - present)

        if missing:
            info["status"] = "incomplete"
            info["reason"] = f"missing {len(missing)} of {total} {fmt} weight shard files (e.g., indices: {missing[:5]})"
            return info

    # Either unsharded weights or all shards accounted for
    info["status"] = "complete"

    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check which models in a JSON list are downloaded in the local HF cache.",
    )
    parser.add_argument(
        "-f",
        "--file",
        default=DEFAULT_JSON,
        help=f"JSON file with model data (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "-d",
        "--model-dir",
        default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
        help=f"Base model directory (default: {DEFAULT_MODEL_DIR} or $MODEL_DIR)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        help="Only check models with this tensor_parallel_size",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based) in the JSON list",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End index (exclusive) in the JSON list",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of models to check (after filtering)",
    )
    parser.add_argument(
        "--list-file",
        type=str,
        help=(
            "Optional path to write a plain-text list of model_ids that are not "
            "fully downloaded (statuses: incomplete, missing)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    json_path = Path(args.file)
    if not json_path.exists():
        raise SystemExit(f"JSON file not found: {json_path}")

    try:
        models = load_models(str(json_path))
    except Exception as e:
        raise SystemExit(f"Failed to load JSON {json_path}: {e}")

    # Filter by TP if requested
    if args.tp is not None:
        models = [m for m in models if m.get("tensor_parallel_size") == args.tp]

    # Slice by start/end
    end_idx: Optional[int] = args.end if args.end is not None else len(models)
    models = models[args.start:end_idx]

    # Apply limit if provided
    if args.limit is not None and args.limit < len(models):
        models = models[: args.limit]

    if not models:
        print("No models to check (after filtering).")
        return

    base_dir = Path(args.model_dir)
    print(f"Base model directory: {base_dir}")
    print(f"Models to check: {len(models)}")
    print("-" * 80)

    results: List[Dict[str, Any]] = []
    counts = {"complete": 0, "incomplete": 0, "missing": 0}

    for idx, model in enumerate(models, start=1):
        info = check_single_model(model, base_dir)
        results.append(info)
        status = info["status"]
        counts[status] = counts.get(status, 0) + 1

        model_id = info["model_id"]
        reason = f" ({info['reason']})" if info.get("reason") else ""
        print(f"[{idx}/{len(models)}] {model_id:<50} -> {status.upper()}{reason}")

    print("-" * 80)
    total = len(results)
    print(f"Total models checked: {total}")
    for k in ("complete", "incomplete", "missing"):
        v = counts.get(k, 0)
        pct = (v / total * 100) if total > 0 else 0.0
        print(f"{k.capitalize():<10}: {v:>5} ({pct:5.1f}%)")

    # Always write JSON output
    default_out = Path("experiment_results") / "model_cache_report.json"
    out_path = default_out
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "base_model_dir": str(base_dir),
                    "json_file": str(json_path),
                    "counts": counts,
                    "models": results,
                },
                f,
                indent=2,
            )
        print(f"\nDetailed report saved to: {out_path}")
    except Exception as e:
        print(f"\nFailed to write JSON report to {out_path}: {e}")

    # Optionally write a list of model_ids for redownloading
    if args.list_file:
        list_path = Path(args.list_file)
        try:
            list_path.parent.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            # If parent cannot be created, fall through to error when opening file
            pass

        try:
            with open(list_path, "w") as f:
                for info in results:
                    if info["status"] in ("incomplete", "missing"):
                        f.write(f"{info['model_id']}\n")
            print(f"Model id list for incomplete/missing downloads saved to: {list_path}")
        except Exception as e:
            print(f"Failed to write model list to {list_path}: {e}")


if __name__ == "__main__":
    main()


