#!/usr/bin/env python3
"""Chunked download → train → delete loop for disks with a ~3 TB limit.

Workflow per chunk (default budget 2.5 TiB of *data*):
  1. Download/index only the repos assigned to this chunk
  2. Fine-tune (resume from previous chunk's best.pt when present)
  3. Export submission zip
  4. Delete that chunk's HF dataset cache to free space
  5. Advance to the next chunk

Repos larger than the budget (e.g. OpenFake ~6 TiB) are never fully mirrored;
they use stream_cap (≤ max_per_source images extracted).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.chunk_plan import (
    build_chunk_plan_from_config,
    load_plan,
    save_plan,
    summarize_plan,
)
from src.data.download import delete_repo_caches, dir_size_bytes, expand_cache_dir
from src.paths import add_path_arguments, apply_path_overrides
from src.train import load_config, train_loop

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("train_chunked")


def _state_path(cfg: Dict[str, Any]) -> Path:
    chunk_cfg = cfg.get("chunk") or {}
    out = Path(cfg.get("train", {}).get("output_dir", "runs/exp")).expanduser()
    return Path(chunk_cfg.get("state_path") or (out / "chunk_state.json")).expanduser()


def _plan_path(cfg: Dict[str, Any]) -> Path:
    chunk_cfg = cfg.get("chunk") or {}
    out = Path(cfg.get("train", {}).get("output_dir", "runs/exp")).expanduser()
    return Path(chunk_cfg.get("plan_path") or (out / "chunk_plan.json")).expanduser()


def load_or_build_plan(
    cfg: Dict[str, Any], *, refresh_sizes: bool = False, force_rebuild_plan: bool = False
):
    plan_file = _plan_path(cfg)
    if plan_file.exists() and not force_rebuild_plan and not refresh_sizes:
        logger.info("Loading existing chunk plan %s", plan_file)
        return load_plan(plan_file)

    catalog = Path(
        (cfg.get("chunk") or {}).get("catalog_path")
        or (expand_cache_dir(cfg["cache_dir"]) / "repo_sizes.json")
    )
    chunks, _sizes = build_chunk_plan_from_config(
        cfg, catalog_path=catalog, refresh_sizes=refresh_sizes or force_rebuild_plan
    )
    save_plan(chunks, plan_file)
    logger.info("Chunk plan summary:\n%s", json.dumps(summarize_plan(chunks), indent=2))
    return chunks


def load_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "completed_chunk_ids": [],
        "next_chunk_id": 0,
        "last_checkpoint": None,
        "history": [],
    }


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_one_chunk(
    cfg: Dict[str, Any],
    chunk,
    *,
    resume_from: Optional[str],
    delete_after: bool,
) -> Dict[str, Any]:
    data = cfg.setdefault("data", {})
    train = cfg.setdefault("train", {})
    chunk_cfg = cfg.get("chunk") or {}

    data["dataset_allowlist"] = list(chunk.entry_names)
    data["repo_allowlist"] = list(chunk.repo_ids)
    data["stream_only_repos"] = list(chunk.stream_repo_ids)
    # Force rebuild index for this chunk's allowlist
    index_path = Path(cfg["index_path"]).expanduser()
    if index_path.exists():
        index_path.unlink()

    epochs = chunk_cfg.get("epochs_per_chunk")
    if epochs is not None:
        train["num_epochs"] = int(epochs)
    if resume_from:
        train["resume_from"] = resume_from
    else:
        train.pop("resume_from", None)

    # Per-chunk output subdir keeps history; best.pt also copied to rolling path
    base_out = Path(train.get("output_dir", "runs/exp")).expanduser()
    chunk_out = base_out / f"chunk_{chunk.chunk_id:03d}"
    chunk_out.mkdir(parents=True, exist_ok=True)
    # Temporarily point train output at chunk dir, but keep export at configured paths
    train["output_dir"] = str(chunk_out)

    cache = expand_cache_dir(cfg["cache_dir"])
    before = dir_size_bytes(cache / "hf")
    logger.info(
        "=== Chunk %d: %d repos (%d full, %d stream_cap), est %.2f TiB, hf_cache=%.2f GiB ===",
        chunk.chunk_id,
        len(chunk.repos),
        len(chunk.full_repo_ids),
        len(chunk.stream_repo_ids),
        chunk.estimated_bytes / (1024**4),
        before / (1024**3),
    )

    result = train_loop(cfg, rebuild_index=True)
    best = result.get("best_checkpoint")
    rolling = base_out / "best.pt"
    if best and Path(best).is_file():
        shutil.copy2(best, rolling)
        result["rolling_checkpoint"] = str(rolling)

    after = dir_size_bytes(cache / "hf")
    result["chunk_id"] = chunk.chunk_id
    result["hf_cache_gib_before_delete"] = after / (1024**3)
    result["entry_names"] = chunk.entry_names
    result["repo_ids"] = chunk.repo_ids

    if delete_after:
        n = delete_repo_caches(cfg["cache_dir"], chunk.repo_ids)
        freed = after - dir_size_bytes(cache / "hf")
        result["deleted_repos"] = n
        result["freed_gib"] = freed / (1024**3)
        logger.info("Deleted %d repo caches, freed %.2f GiB", n, result["freed_gib"])
        # Drop chunk index so next chunk cannot see stale paths
        if index_path.exists():
            index_path.unlink()

    (chunk_out / "chunk_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_chunked(
    cfg: Dict[str, Any],
    *,
    refresh_sizes: bool = False,
    rebuild_plan: bool = False,
    max_chunks: Optional[int] = None,
    only_chunk: Optional[int] = None,
    start_chunk: Optional[int] = None,
    delete_after: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    chunks = load_or_build_plan(cfg, refresh_sizes=refresh_sizes, force_rebuild_plan=rebuild_plan)
    state_path = _state_path(cfg)
    state = load_state(state_path)
    completed = set(state.get("completed_chunk_ids") or [])

    if dry_run:
        summary = summarize_plan(chunks)
        summary["completed_chunk_ids"] = sorted(completed)
        summary["next_chunk_id"] = state.get("next_chunk_id", 0)
        print(json.dumps(summary, indent=2))
        return summary

    # Restore base output_dir (run_one_chunk mutates it)
    base_out = str(
        Path((cfg.get("train") or {}).get("output_dir", "runs/exp")).expanduser()
    )
    cfg.setdefault("train", {})["output_dir"] = base_out

    resume = state.get("last_checkpoint")
    ran: List[Dict[str, Any]] = []
    n_run = 0

    for chunk in chunks:
        if only_chunk is not None and chunk.chunk_id != only_chunk:
            continue
        if start_chunk is not None and chunk.chunk_id < start_chunk:
            continue
        if chunk.chunk_id in completed and only_chunk is None:
            logger.info("Skipping completed chunk %d", chunk.chunk_id)
            continue
        if max_chunks is not None and n_run >= max_chunks:
            break

        cfg["train"]["output_dir"] = base_out  # reset before each chunk
        result = run_one_chunk(cfg, chunk, resume_from=resume, delete_after=delete_after)
        ran.append({k: v for k, v in result.items() if k != "history"})
        completed.add(chunk.chunk_id)
        resume = result.get("rolling_checkpoint") or result.get("best_checkpoint") or resume
        state = {
            "completed_chunk_ids": sorted(completed),
            "next_chunk_id": chunk.chunk_id + 1,
            "last_checkpoint": resume,
            "history": (state.get("history") or []) + ran[-1:],
        }
        save_state(state_path, state)
        n_run += 1

    final = {
        "completed_chunk_ids": sorted(completed),
        "last_checkpoint": resume,
        "chunks_run_this_session": n_run,
        "results": ran,
    }
    print(json.dumps({k: v for k, v in final.items() if k != "results"}, indent=2))
    return final


def parse_args():
    p = argparse.ArgumentParser(
        description="Chunked SN34 training under a disk budget (default 2.5 TiB / chunk)"
    )
    p.add_argument("--config", default=str(ROOT / "configs/train_vit_phase1.yaml"))
    p.add_argument(
        "--budget-tib",
        type=float,
        default=None,
        help="Max data disk per chunk in TiB (default 2.5)",
    )
    p.add_argument("--refresh-sizes", action="store_true", help="Re-query HF repo sizes")
    p.add_argument("--rebuild-plan", action="store_true", help="Rebuild chunk plan JSON")
    p.add_argument("--dry-run", action="store_true", help="Print plan only, no download/train")
    p.add_argument("--max-chunks", type=int, default=None, help="Run at most N new chunks")
    p.add_argument("--only-chunk", type=int, default=None, help="Run a single chunk id")
    p.add_argument("--start-chunk", type=int, default=None, help="Skip chunks before this id")
    p.add_argument(
        "--keep-data",
        action="store_true",
        help="Do not delete HF caches after a chunk (debug)",
    )
    p.add_argument("--max-per-source", type=int, default=None)
    p.add_argument("--dataset-limit", type=int, default=None, help="Smoke: limit registry rows before packing")
    add_path_arguments(p)
    return p.parse_args()


def main():
    args = parse_args()
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
    chunk_cfg = cfg.setdefault("chunk", {})
    if args.budget_tib is not None:
        chunk_cfg["budget_tib"] = float(args.budget_tib)
    if args.max_per_source is not None:
        cfg.setdefault("data", {})["max_per_source"] = args.max_per_source
    if args.dataset_limit is not None:
        cfg.setdefault("data", {})["dataset_limit"] = args.dataset_limit

    run_chunked(
        cfg,
        refresh_sizes=args.refresh_sizes,
        rebuild_plan=args.rebuild_plan,
        max_chunks=args.max_chunks,
        only_chunk=args.only_chunk,
        start_chunk=args.start_chunk,
        delete_after=not args.keep_data,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
