#!/usr/bin/env bash
# Chunked Phase-1 train under a 2.5 TiB data budget (for ~3 TB disks).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

DATA="${DATA:-$ROOT/.data}"
mkdir -p "$DATA" "$DATA/runs/vit_phase1" "$DATA/submission"

python scripts/plan_chunks.py --config configs/train_vit_phase1.yaml \
  --data-dir "$DATA" --output-dir "$DATA/runs/vit_phase1" \
  --budget-tib "${BUDGET_TIB:-2.5}" "$@"

python -m src.train_chunked --config configs/train_vit_phase1.yaml \
  --data-dir "$DATA" \
  --output-dir "$DATA/runs/vit_phase1" \
  --submission-dir "$DATA/submission" \
  --zip-path "$DATA/submission/image_detector.zip" \
  --budget-tib "${BUDGET_TIB:-2.5}" "$@"
