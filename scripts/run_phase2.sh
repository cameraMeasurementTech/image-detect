#!/usr/bin/env bash
# Phase-2 train: public registries + GAS-Station + robustness augs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
python -m src.train --config configs/train_vit_phase2.yaml "$@"
python -m src.eval_local --config configs/train_vit_phase2.yaml --mode small --model-dir submission "$@" || true
