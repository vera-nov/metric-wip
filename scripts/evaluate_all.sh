#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/evaluate_all.sh [path/to/reports.json]
# Relative input paths are resolved from the directory where you run the script.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
input="${1:-$repo_dir/reports.json}"
if [[ "$input" != /* ]]; then
  input="$PWD/$input"
fi
cd -- "$repo_dir"

conda run --no-capture-output -n cxr-fit python score_component.py russian "$input" \
  --output russian_scores.json
conda run --no-capture-output -n cxr-fit python score_component.py text "$input" \
  --output text_scores.json
conda run --no-capture-output -n cxr-fit python score_component.py image "$input" \
  --output image_scores.json
conda run --no-capture-output -n cxr-fit python merge_scores.py \
  russian_scores.json text_scores.json image_scores.json --output final_scores.json
