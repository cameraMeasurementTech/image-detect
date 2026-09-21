# BitMind SN34 Image Discriminator — Train Pipeline

Fine-tune and package an image provenance classifier for [Bittensor SN34 (GAS)](https://github.com/BitMind-AI/bitmind-subnet) to compete for the **image King-of-the-Hill** lane (40% of emissions).

Reference strategy: [bitmind-subnet/docs/Image-Discriminator-Gates.md](../bitmind-subnet/docs/Image-Discriminator-Gates.md), [arXiv:2607.13234](https://arxiv.org/pdf/2607.13234).

## What this adapts from the notebook

[`ai-vs-human-generated-images-prediction-vit.ipynb`](ai-vs-human-generated-images-prediction-vit.ipynb) is a Kaggle binary ViT fine-tune (human vs AI, accuracy-only). This pipeline:

| Notebook | This repo |
|----------|-----------|
| Kaggle CSV binary labels | Gasbench `real` / `synthetic` / `semisynthetic` (3-class) |
| Accuracy | Multiclass MCC + Brier → `sn34_score` |
| HF Hub export | Safetensors zip (`model_config.yaml` + `model.py` + weights) |
| Random image split | Source-stratified holdout by `generator_family` |
| Mild geometric augs | Optional JPEG/WebP/resize robustness chain (Phase 2) |

## GPU requirements (Phase-1 base model)

Base model is `google/vit-base-patch16-224-in21k` (**~85.8M** parameters, ViT-B/16). Phase-1 config is `224×224`, train batch **32**, eval batch **64**, AdamW, **fp32** (no mixed precision in `src/train.py`), 5 epochs. The trainer uses **one GPU** (`cuda` if present, else CPU). Multi-GPU is not required and is not wired up.

| Role | GPU | VRAM | Notes |
|------|-----|------|--------|
| Minimum for Phase 1 as written | **1× 16 GB** (RTX 4060 Ti 16GB, RTX 4080, T4, A4000) | ~10–14 GB peak | Batch 32 fp32 + Adam states |
| Recommended | **1× 24 GB** (RTX 3090 / 4090, A5000, L4) | headroom to ~20 GB | Same config; faster throughput; room for Phase-2 JPEG/WebP augs |
| Tight fallback | **1× 12 GB** | ~7–9 GB if you cut batch | Set `train.train_batch_size: 16` and `eval_batch_size: 32` |
| Not needed | 40–80 GB, multi-GPU, multi-node | — | 85.8M does not fill an A100; extra GPUs only shorten wall-clock if you add DDP later |

Memory sketch at batch 32, fp32:

- Weights + gradients + AdamW (m, v): about **1.4 GB** (`85.8e6 × 4 bytes × 4`)
- Activations (12 layers, 197 tokens, hidden 768) dominate the rest and push the peak into the **10–14 GB** range

Wall-clock (order of magnitude, one GPU, cap `max_per_source: 2000` on ~220 sources, up to ~4e5 images, 5 epochs):

- 24 GB consumer GPU (4090-class): roughly **half a day to a day**
- 16 GB datacenter GPU (T4-class): roughly **1–2 days**

Disk is separate from VRAM: full Hugging Face registry downloads can be **hundreds of GB**. Cache lives under `~/.cache/sn34-miner/image`.

Phase 3 (`facebook/convnext-base-224-22k`, batch 16) is a similar single **16–24 GB** GPU. Enabling the two-branch ensemble roughly doubles activation memory; use a **24 GB** card or train branches one at a time.

## Quick start

```bash
cd /home/ai-image-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pick where everything lands (downloads, index, runs, submission zip)
DATA=/mnt/data/sn34          # change to any path you own
mkdir -p "$DATA" "$DATA/runs" "$DATA/submission"

# Registry summary (no download)
python scripts/build_index.py --summarize-only --data-dir "$DATA"

# Smoke tests (metrics / split / export)
python scripts/smoke_test.py

# Phase 1 — smoke train on a few datasets
python -m src.train --config configs/train_vit_phase1.yaml \
  --data-dir "$DATA" \
  --output-dir "$DATA/runs/vit_phase1" \
  --submission-dir "$DATA/submission" \
  --zip-path "$DATA/submission/image_detector.zip" \
  --dataset-limit 4 --max-per-source 64 --rebuild-index

# Phase 1 — full public registries (long; large disk/bandwidth)
python -m src.train --config configs/train_vit_phase1.yaml \
  --data-dir "$DATA" --output-dir "$DATA/runs/vit_phase1" --rebuild-index

# Phase 2 — + GAS-Station + robustness augs
python -m src.train --config configs/train_vit_phase2.yaml \
  --data-dir "$DATA" --output-dir "$DATA/runs/vit_phase2" --rebuild-index

# Phase 3 — ConvNeXt / stronger backbone
python -m src.train --config configs/train_ensemble_phase3.yaml \
  --data-dir "$DATA" --output-dir "$DATA/runs/convnext_phase3" --rebuild-index

# Local gasbench gates (same index / model paths)
python -m src.eval_local --config configs/eval_local.yaml \
  --data-dir "$DATA" --model-dir "$DATA/submission" --mode small \
  --out "$DATA/runs/local_sn34_small.json"
python -m src.eval_local --config configs/eval_local.yaml \
  --data-dir "$DATA" --model-dir "$DATA/submission" --mode full \
  --out "$DATA/runs/local_sn34_full.json"

# Optional: the upstream gasbench CLI, if installed
python -m src.eval_local --use-gasbench --mode small --model-dir "$DATA/submission"

# Push when local proxy clears king + margin (set king_sn34_score in config)
python scripts/push_model.py --zip "$DATA/submission/image_detector.zip" --require-push-ready
```

## Where data is saved

Every download / index / train / eval command accepts path flags (CLI overrides YAML):

| Flag | What it controls |
|------|------------------|
| `--data-dir DIR` | Root for HF downloads + default `DIR/index.jsonl` |
| `--cache-dir DIR` | Download cache only (overrides `--data-dir` for cache) |
| `--index-path FILE` | Sample index JSONL read/write path |
| `--output-dir DIR` | Checkpoints + `train_result.json` |
| `--submission-dir DIR` | Exported `model_config.yaml` / `model.py` / weights |
| `--zip-path FILE` | Packed `image_detector.zip` |
| `--out FILE` | Eval report JSON (`eval_local` only) |

`--data-dir /path` alone is enough for most workflows:

```text
/path/
  hf/ ...          # HuggingFace dataset cache
  yaml/ ...        # registry YAML cache
  index.jsonl      # sample index
```

Then point train/eval outputs separately with `--output-dir`, `--submission-dir`, `--zip-path`, and `--out`.
## Layout

```
configs/           train_vit_phase1.yaml, phase2, phase3
src/data/          YAML index, HF download, split, augs, dataset
src/metrics/       sn34_score (Gorodkin MCC × Brier)
src/models/        ViT/ConvNeXt/ensemble + safetensors export
src/train.py       main training loop
src/eval_local.py  gasbench --small/--full + king margin helper
scripts/           build_index, smoke_test, push_model
submission/        built zip contents after train
```

## Local eval (same scoring as the network bench)

Validators do not run your model. GAS/gasbench does, then validators copy the resulting kings onto the chain. `src/eval/score_model.py` reproduces that scorer on your machine:

- Input is uint8 RGB `[B,3,H,W]` from `model_config.yaml` resize; the scorer applies softmax to your logits.
- Image head is 3-class. Score is multiclass Gorodkin MCC and multiclass Brier, combined as gasbench does: `sqrt(M_norm^1.2 * B_norm^1.8)`.
- `--mode small` keeps at most 100 images per dataset and treats accuracy ≥ 0.80 as the entrance exam.
- `--mode full` scores every indexed image.
- `aug_weight` blends `(1-w)*base + w*aug` using the JPEG/WebP/resize chain.
- `shares` splits the score across `public` and `gas_station`. A missing bucket is dropped and the rest renormalized. Private holdouts are not in this index, so the local number is an upper bound on the network full bench.
- `king_sn34` checks the dethrone rule: local `sn34_score` ≥ king + 0.01.

```bash
python scripts/build_index.py --config configs/train_vit_phase1.yaml --data-dir "$DATA"
python -m src.eval_local --config configs/eval_local.yaml \
  --data-dir "$DATA" --model-dir "$DATA/submission" --mode small \
  --out "$DATA/runs/local_sn34_small.json"
python scripts/test_local_eval.py
```

Report JSON defaults to `runs/local_sn34_report.json` unless `--out` is set.
## Gates reminder

1. 3-class logits, uint8 `[B,3,H,W]` in  
2. Sandbox allow-listed imports only  
3. `gasbench --small` ≥ **80%** accuracy  
4. KOTH: `sn34_score ≥ king + 0.01`  

Do **not** burn 0.5 TAO SN34 α to resubmit until local public+aug proxy clears king + ~0.015 safety margin.

## Eval data note

Validators do not score your zip. They upload verified synthetics to `gasstation/gs-images-v4`. GAS/gasbench mixes public YAML + private holdouts + GAS-Station for `sn34_score`, then validators set the image-lane weights from the kings API.
