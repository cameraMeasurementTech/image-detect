#!/usr/bin/env bash
# Phase-3 train on /mnt/sn34.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

DATA="${DATA:-${SN34_DATA_ROOT:-/mnt/sn34}}"
export SN34_DATA_ROOT="$DATA"
mkdir -p "$DATA" "$DATA/runs/convnext_phase3" "$DATA/submission"

python -m src.train_chunked --config configs/train_ensemble_phase3.yaml \
  --data-dir "$DATA" \
  --output-dir "$DATA/runs/convnext_phase3" \
  --submission-dir "$DATA/submission" \
  --zip-path "$DATA/submission/image_detector.zip" \
  --budget-tib "${BUDGET_TIB:-2.5}" "$@"

python -m src.eval_local --config configs/eval_local.yaml \
  --data-dir "$DATA" --mode small --model-dir "$DATA/submission" \
  --out "$DATA/runs/convnext_phase3/local_sn34_small.json" || true
echo "Set model.king_sn34_score in config, then:"
echo "  python scripts/push_model.py --require-push-ready --train-result $DATA/runs/convnext_phase3/train_result.json"
echo "  # or zip: $DATA/submission/image_detector.zip"
