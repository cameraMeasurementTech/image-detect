#!/usr/bin/env python3
"""Build / refresh the local image index from gasbench registries (+ optional GAS-Station)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.download import build_public_index, index_gas_station, merge_indices, write_index
from src.data.gasbench_index import load_registry_entries, summarize_entries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_index")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/train_vit_phase1.yaml"))
    p.add_argument("--dataset-limit", type=int, default=None)
    p.add_argument("--max-per-source", type=int, default=None)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--with-gas-station", action="store_true")
    p.add_argument("--summarize-only", action="store_true", help="Only print registry summary")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cache = cfg["cache_dir"]
    entries = load_registry_entries(
        cfg.get("registries"),
        cache_dir=str(Path(cache).expanduser() / "yaml"),
        skip_paths=["gasstation/"],
    )
    summary = summarize_entries(entries)
    print(json.dumps(summary, indent=2))
    if args.summarize_only:
        return

    max_per = args.max_per_source or int(cfg.get("data", {}).get("max_per_source", 2000))
    limit = args.dataset_limit or cfg.get("data", {}).get("dataset_limit")
    samples = build_public_index(
        cache_dir=cache,
        index_path=cfg["index_path"],
        registries=cfg.get("registries"),
        max_per_source=max_per,
        seed=int(cfg.get("seed", 42)),
        download=not args.no_download,
        dataset_limit=limit,
    )

    gs = cfg.get("gas_station") or {}
    if args.with_gas_station or gs.get("enabled"):
        gs_samples = index_gas_station(
            cache_dir=cache,
            repo_id=gs.get("path", "gasstation/gs-images-v4"),
            media_type=gs.get("media_type", "synthetic"),
            max_samples=int(gs.get("max_samples", 20000)),
            weeks=gs.get("weeks"),
            download=not args.no_download,
            seed=int(cfg.get("seed", 42)),
        )
        samples = merge_indices(samples, gs_samples, index_path=cfg["index_path"])
    else:
        write_index(samples, cfg["index_path"])

    logger.info("Done: %d samples -> %s", len(samples), cfg["index_path"])


if __name__ == "__main__":
    main()
