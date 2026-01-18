#!/usr/bin/env python3
"""
Local HF model downloader with retries, meant to be used in a user-writable directory.

Usage example:

    python local_download.py \
        --list /home/secure/models_incomplete_10.txt \
        --dir  /home/secure/hf_models_local

This script:
- Reads model_ids from a text file (one per line)
- Downloads them with huggingface_hub.snapshot_download into a cache dir you own
- Retries transient / partial-download errors automatically
- Avoids useless retries on gated / permission errors

Tokens:
- You can pass --token, or rely on env vars:
    HF_TOKEN or HUGGING_FACE_HUB_TOKEN
"""

import os
import time
import argparse
import datetime
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


# Time to wait between requests to avoid hitting rate limits too quickly
INTER_MODEL_DELAY = 1.5  # seconds
LOG_FILE = "local_download.log"

# Retry settings for transient errors or partial downloads
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds between retries


def log_failure(model_id: str, error_msg: str):
    """
    Appends failure details to the local log file with a timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_error = str(error_msg).replace("\n", " ")
    log_entry = f"[{timestamp}] FAILED: {model_id} | Reason: {clean_error}\n"

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


def classify_error(e: Exception) -> tuple[str, bool]:
    """
    Classify an exception to decide whether to retry and whether to use force_download.

    Returns:
        (category, use_force_download)
        category:
            - 'non_retryable_auth'  -> don't retry (gated/private, bad auth, 401/403)
            - 'retry'               -> retry is useful
        use_force_download:
            - True  -> next retry should use force_download=True
            - False -> keep force_download=False
    """
    msg = str(e)

    # Auth / gated / permission issues: don't retry automatically
    if isinstance(e, RepositoryNotFoundError):
        return "non_retryable_auth", False

    if isinstance(e, HfHubHTTPError):
        if e.response is not None and e.response.status_code in (401, 403):
            return "non_retryable_auth", False

    lowered = msg.lower()
    auth_markers = [
        "gated repo",
        "restricted",
        "repository not found",
        "access denied",
        "invalid username or password",
        "make sure your token has the correct permissions",
    ]
    if any(m in lowered for m in auth_markers):
        return "non_retryable_auth", False

    # Partial / inconsistent downloads: suggest force_download
    if "consistency check failed" in lowered or "please retry with `force_download=true`" in lowered:
        return "retry", True

    if "peer closed connection without sending complete message body" in lowered:
        return "retry", True

    # Timeouts / transient network / SSL errors: retry without forcing by default
    timeout_markers = [
        "timed out",
        "handshake operation timed out",
        "the read operation timed out",
        "ssl",
        "connection aborted",
        "connection reset",
    ]
    if any(m in lowered for m in timeout_markers):
        return "retry", False

    # Fallback: allow retry without force_download
    return "retry", False


def download_models_to_cache(list_file: str, cache_dir: str, token: str | None = None):
    """
    Downloads models sequentially to the specified cache directory.
    Adds retries for transient / partial failures.
    """
    # 1. Validate input file
    if not os.path.exists(list_file):
        print(f"[ERROR] List file not found: {list_file}")
        return

    # 2. Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # 3. Read model list
    with open(list_file, "r", encoding="utf-8") as f:
        # Filter out empty lines and comments
        models = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    total = len(models)
    print("Task Summary:")
    print(f"  - Input List: {list_file}")
    print(f"  - Cache Dir:  {os.path.abspath(cache_dir)}")
    print(f"  - Log File:   {os.path.abspath(LOG_FILE)}")
    print(f"  - Total Models: {total}")
    print("-" * 60)

    success_count = 0
    fail_count = 0

    # 4. Main Download Loop
    for index, model_id in enumerate(models, 1):
        print(f"\n[{index}/{total}] Syncing: {model_id}")

        attempt = 1
        last_error: Exception | None = None

        while attempt <= MAX_RETRIES:
            start_time = time.time()
            try:
                # Decide whether to use force_download based on previous error (if any)
                use_force = False
                if last_error is not None:
                    category, use_force_suggestion = classify_error(last_error)
                    if category == "non_retryable_auth":
                        print("  -> Non-retryable auth/gated error detected, skipping further attempts.")
                        break
                    use_force = use_force_suggestion

                if use_force:
                    print(f"  -> Retry attempt {attempt}/{MAX_RETRIES} with force_download=True")
                elif attempt > 1:
                    print(f"  -> Retry attempt {attempt}/{MAX_RETRIES}")

                snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    token=token,
                    force_download=use_force,
                )

                elapsed = time.time() - start_time
                print(f"OK - Done ({elapsed:.1f}s)")
                success_count += 1
                last_error = None
                break

            except Exception as e:  # noqa: BLE001
                elapsed = time.time() - start_time
                last_error = e
                category, use_force_suggestion = classify_error(e)

                print(f"!!! Error on attempt {attempt}/{MAX_RETRIES} ({elapsed:.1f}s)")
                print(f"    -> {e}")

                if category == "non_retryable_auth":
                    print("    -> Non-retryable auth/gated error, skipping further attempts.")
                    break

                if attempt >= MAX_RETRIES:
                    print(f"    -> Reached max retries ({MAX_RETRIES}), giving up and logging failure.")
                    break

                attempt += 1
                print(f"    -> Will retry in {RETRY_DELAY}s (force_download={use_force_suggestion})")
                time.sleep(RETRY_DELAY)

        # If we exited retry loop without success, record failure
        if last_error is not None:
            log_failure(model_id, str(last_error))
            fail_count += 1

        # Pause briefly to be gentle on the API / Network between models
        time.sleep(INTER_MODEL_DELAY)

    # 5. Final Summary
    print("\n" + "=" * 60)
    print("Job Finished")
    print(f"Successful: {success_count}")
    print(f"Failed:     {fail_count}")
    if fail_count > 0:
        print(f"See {LOG_FILE} for failure details.")
    print(f"Models stored in: {os.path.abspath(cache_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local HuggingFace batch downloader with retries (user-writable dir)."
    )
    parser.add_argument(
        "--list",
        "-l",
        type=str,
        required=True,
        help="Path to the text file containing model IDs.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        required=True,
        help="Target directory for HF cache (must be writable by current user).",
    )
    parser.add_argument(
        "--token",
        "-t",
        type=str,
        default=None,
        help="HuggingFace Access Token (optional, falls back to HF_TOKEN/HUGGING_FACE_HUB_TOKEN).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Prioritize command line token, fallback to environment variables
    hf_token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    download_models_to_cache(args.list, args.dir, hf_token)


if __name__ == "__main__":
    main()


