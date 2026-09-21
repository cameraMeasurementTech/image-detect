#!/usr/bin/env bash
# Phase-3 train: stronger backbone + optional king-margin gated push.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
python -m src.train --config configs/train_ensemble_phase3.yaml "$@"
python -m src.eval_local --config configs/train_ensemble_phase3.yaml --mode both --model-dir submission || true
echo "Set model.king_sn34_score in config, retrain, then:"
echo "  python scripts/push_model.py --require-push-ready --train-result runs/convnext_phase3/train_result.json"
