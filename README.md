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

Disk is separate from VRAM: a **naive full Hub mirror of all registries is ~15 TiB**.
On a **3 TB** volume use **chunked training** (download ≤2.5 TiB → train → delete → next chunk).
See [Chunked training (3 TB disk)](#chunked-training-3-tb-disk) below.

Phase 3 (`facebook/convnext-base-224-22k`, batch 16) is a similar single **16–24 GB** GPU. Enabling the two-branch ensemble roughly doubles activation memory; use a **24 GB** card or train branches one at a time.

## External storage (`/mnt/sn34`)

**All train datasets, indexes, runs, and submission zips default to `/mnt/sn34`.**
Override with `SN34_DATA_ROOT` or `--data-dir` if your volume is mounted elsewhere under `/mnt`.

```text
/mnt/sn34/
  hf/                  # downloaded HF dataset snapshots (chunked; deleted after each chunk)
  yaml/                # gasbench registry YAMLs
  hf_datasets/         # HuggingFace datasets cache
  .hf_home/            # hub / transformers weights cache
  index.jsonl          # current chunk sample index
  repo_sizes.json      # HF size catalog
  runs/vit_phase1/     # checkpoints + chunk_state.json
  submission/          # model_config.yaml, model.py, image_detector.zip
```

```bash
# Mount your 3 TB disk at /mnt (or bind-mount to /mnt/sn34), then:
export SN34_DATA_ROOT=/mnt/sn34   # optional; this is already the default
mkdir -p /mnt/sn34
df -h /mnt/sn34

# No --data-dir needed — configs and CLI default to /mnt/sn34
python scripts/plan_chunks.py --config configs/train_vit_phase1.yaml --budget-tib 2.5 --refresh-sizes
python -m src.train_chunked --config configs/train_vit_phase1.yaml --budget-tib 2.5
# Or: bash scripts/run_chunked_phase1.sh
```

## Chunked training (3 TB disk)

Public gasbench image repos sum to **~15 TiB**. This pipeline packs them into ordered chunks of **≤ 2.5 TiB** of data so you keep ~0.5 TiB free for checkpoints, the model zip, and OS overhead on a 3 TB disk.

Repos larger than 2.5 TiB alone (e.g. OpenFake ~6 TiB) are never fully mirrored: they use **stream_cap** (extract ≤ `max_per_source` images).

```bash
# Preview plan (sizes cached at /mnt/sn34/repo_sizes.json)
python scripts/plan_chunks.py --config configs/train_vit_phase1.yaml \
  --budget-tib 2.5 --refresh-sizes

# Download chunk → train (resume) → delete data → next (all under /mnt/sn34)
python -m src.train_chunked --config configs/train_vit_phase1.yaml --budget-tib 2.5

# Optional controls
python -m src.train_chunked ... --dry-run
python -m src.train_chunked ... --max-chunks 1
python -m src.train_chunked ... --only-chunk 0
python -m src.train_chunked ... --start-chunk 2
python -m src.train_chunked ... --keep-data
```

State files under `/mnt/sn34/runs/vit_phase1/`: `chunk_plan.json`, `chunk_state.json`, rolling `best.pt`.

## Quick start

```bash
cd /home/ai-image-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# External 3 TB volume must be available at /mnt/sn34 (or set SN34_DATA_ROOT)
mkdir -p /mnt/sn34 && df -h /mnt/sn34

# Registry summary (no download)
python scripts/build_index.py --summarize-only

# Smoke tests / gate check
python scripts/smoke_test.py
python scripts/probe_base_inference.py --config configs/train_vit_phase1.yaml
python scripts/check_gates.py --config configs/train_vit_phase1.yaml \
  --keep-probe-dir /mnt/sn34/submission_probe

# Phase 1 — chunked train on /mnt/sn34 (2.5 TiB / chunk)
python -m src.train_chunked --config configs/train_vit_phase1.yaml --budget-tib 2.5
# Or: bash scripts/run_chunked_phase1.sh

# Local eval (reads index + submission from /mnt/sn34)
python -m src.eval_local --config configs/eval_local.yaml --mode small
python -m src.eval_local --config configs/eval_local.yaml --mode full

# Push
python scripts/push_model.py --zip /mnt/sn34/submission/image_detector.zip --require-push-ready
```

If your disk is mounted at another path:

```bash
export SN34_DATA_ROOT=/mnt/mydisk/sn34
# or pass --data-dir /mnt/mydisk/sn34 on every command
```

## Where data is saved

Default root: **`/mnt/sn34`** (`SN34_DATA_ROOT` or `--data-dir` to override).

| Flag | What it controls |
|------|------------------|
| `--data-dir DIR` | Root for HF downloads, index, runs, submission (default `/mnt/sn34`) |
| `--cache-dir DIR` | Download cache only (defaults to `--data-dir`) |
| `--index-path FILE` | Sample index JSONL |
| `--output-dir DIR` | Checkpoints / `train_result.json` |
| `--submission-dir DIR` | Export dir |
| `--zip-path FILE` | Submission zip |
| `--out FILE` | Eval report JSON |

HF hub / transformers / datasets caches are also redirected under `/mnt/sn34/.hf_home` and `/mnt/sn34/hf_datasets` so nothing large lands in `~/.cache`.
## Layout

```
configs/           train_vit_phase1.yaml (+ chunk:), phase2, phase3
src/data/          YAML index, HF download, chunk_plan, split, augs
src/train.py       single-shot training loop
src/train_chunked.py  download→train→delete under disk budget
scripts/           plan_chunks, build_index, smoke_test, push_model, check_gates
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
