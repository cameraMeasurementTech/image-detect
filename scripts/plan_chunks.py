#!/usr/bin/env python3
"""Preview or rebuild the disk-budget chunk plan (no training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.chunk_plan import build_chunk_plan_from_config, save_plan, summarize_plan
from src.paths import add_path_arguments, apply_path_overrides
from src.train import load_config


def main():
    p = argparse.ArgumentParser(description="Plan 2.5 TiB download/train chunks")
    p.add_argument("--config", default=str(ROOT / "configs/train_vit_phase1.yaml"))
    p.add_argument("--budget-tib", type=float, default=2.5)
    p.add_argument("--refresh-sizes", action="store_true")
    p.add_argument("--out", default=None, help="Write plan JSON (default: <output-dir>/chunk_plan.json)")
    p.add_argument("--max-per-source", type=int, default=None)
    p.add_argument("--dataset-limit", type=int, default=None)
    add_path_arguments(p)
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    apply_path_overrides(
        cfg,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        index_path=args.index_path,
        output_dir=args.output_dir,
        submission_dir=args.submission_dir,
        zip_path=args.zip_path,
    )
    cfg.setdefault("chunk", {})["budget_tib"] = float(args.budget_tib)
    if args.max_per_source is not None:
        cfg.setdefault("data", {})["max_per_source"] = args.max_per_source
    if args.dataset_limit is not None:
        cfg.setdefault("data", {})["dataset_limit"] = args.dataset_limit

    out = Path(
        args.out
        or (
            Path(cfg.get("train", {}).get("output_dir", "runs/exp")).expanduser()
            / "chunk_plan.json"
        )
    )
    catalog = Path(cfg["cache_dir"]).expanduser() / "repo_sizes.json"
    chunks, _ = build_chunk_plan_from_config(
        cfg, catalog_path=catalog, refresh_sizes=args.refresh_sizes
    )
    save_plan(chunks, out)
    summary = summarize_plan(chunks)
    print(json.dumps(summary, indent=2))
    print(f"Plan written: {out}")
    print(f"Size catalog: {catalog}")


if __name__ == "__main__":
    main()
