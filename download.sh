#!/bin/bash
# Download models from HuggingFace
# 
# Usage:
#   HF_TOKEN=your_token ./download.sh --list models.txt --dir /path/to/models
#
# Or set HF_TOKEN in your environment:
#   export HF_TOKEN=your_token
#   ./download.sh --list models.txt --dir /path/to/models

# Optional: Use HuggingFace mirror for faster downloads in China
# export HF_ENDPOINT=https://hf-mirror.com

python local_download.py "$@"

