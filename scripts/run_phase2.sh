#!/usr/bin/env bash
# Phase-2 train on /mnt/sn34 (prefer chunked if disk-limited).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

DATA="${DATA:-${SN34_DATA_ROOT:-/mnt/sn34}}"
export SN34_DATA_ROOT="$DATA"
mkdir -p "$DATA" "$DATA/runs/vit_phase2" "$DATA/submission"

python -m src.train_chunked --config configs/train_vit_phase2.yaml \
  --data-dir "$DATA" \
  --output-dir "$DATA/runs/vit_phase2" \
  --submission-dir "$DATA/submission" \
  --zip-path "$DATA/submission/image_detector.zip" \
  --budget-tib "${BUDGET_TIB:-2.5}" "$@"

python -m src.eval_local --config configs/eval_local.yaml \
  --data-dir "$DATA" --mode small --model-dir "$DATA/submission" \
  --out "$DATA/runs/vit_phase2/local_sn34_small.json" || true
