#!/usr/bin/env bash
# Create the three model environments and the lightweight CPU controller.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v conda >/dev/null

# Override with a compatible official PyTorch wheel index when needed.
torch_index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
for spec in cxr-fit:3.11 cxr-qwen:3.11 cxr-text:3.12 cxr-image:3.11; do
  env_name="${spec%%:*}"
  python_version="${spec##*:}"
  conda create -n "$env_name" "python=$python_version" pip -y
  conda run -n "$env_name" python -m pip install --upgrade pip
  if [[ "$env_name" != cxr-fit ]]; then
    conda run -n "$env_name" python -m pip install \
      torch==2.6.0 torchvision==0.21.0 --index-url "$torch_index"
  fi
  case "$env_name" in
    cxr-fit) requirements=requirements-fit.txt ;;
    cxr-qwen) requirements=requirements-qwen.txt ;;
    cxr-text) requirements=requirements-text.txt ;;
    cxr-image) requirements=requirements-biovilt.txt ;;
  esac
  conda run -n "$env_name" python -m pip install -r "$requirements"
  conda run -n "$env_name" python -m pip check
done
