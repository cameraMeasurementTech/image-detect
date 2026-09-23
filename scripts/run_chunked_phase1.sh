#!/usr/bin/env bash
# Chunked Phase-1 train on external storage under /mnt/sn34 (≤2.5 TiB data / chunk).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

# External volume — override with SN34_DATA_ROOT or DATA
DATA="${DATA:-${SN34_DATA_ROOT:-/mnt/sn34}}"
export SN34_DATA_ROOT="$DATA"
mkdir -p "$DATA" "$DATA/runs/vit_phase1" "$DATA/submission"

echo "Using external data root: $DATA"
df -h "$DATA" || true

python scripts/plan_chunks.py --config configs/train_vit_phase1.yaml \
  --data-dir "$DATA" --output-dir "$DATA/runs/vit_phase1" \
  --budget-tib "${BUDGET_TIB:-2.5}" "$@"

python -m src.train_chunked --config configs/train_vit_phase1.yaml \
  --data-dir "$DATA" \
  --output-dir "$DATA/runs/vit_phase1" \
  --submission-dir "$DATA/submission" \
  --zip-path "$DATA/submission/image_detector.zip" \
  --budget-tib "${BUDGET_TIB:-2.5}" "$@"
