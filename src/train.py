#!/usr/bin/env python3
"""Train an SN34 image discriminator (Phase 1–3)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

# Allow running as `python -m src.train` or `python src/train.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ImageIndexDataset, collate_batch
from src.data.download import (
    build_public_index,
    index_gas_station,
    merge_indices,
    read_index,
    write_index,
)
from src.data.split import source_stratified_split, summarize_split
from src.metrics.sn34 import compute_sn34_metrics, fit_temperature
from src.models.export_bundle import export_submission
from src.models.vit_detector import build_export_wrapper, build_train_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("train")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_index(cfg: Dict[str, Any], force_rebuild: bool = False) -> Path:
    index_path = Path(cfg["index_path"]).expanduser()
    cache_dir = cfg["cache_dir"]
    if index_path.exists() and not force_rebuild:
        logger.info("Using existing index %s", index_path)
        return index_path

    max_per = int(cfg.get("data", {}).get("max_per_source", 2000))
    samples = build_public_index(
        cache_dir=cache_dir,
        index_path=index_path,
        registries=cfg.get("registries"),
        max_per_source=max_per,
        seed=int(cfg.get("seed", 42)),
        download=True,
        skip_gasstation_in_registry=True,
        dataset_allowlist=cfg.get("data", {}).get("dataset_allowlist"),
        dataset_limit=cfg.get("data", {}).get("dataset_limit"),
        repo_allowlist=cfg.get("data", {}).get("repo_allowlist"),
        stream_only_repos=cfg.get("data", {}).get("stream_only_repos"),
    )

    gs = cfg.get("gas_station") or {}
    if gs.get("enabled") and not cfg.get("data", {}).get("repo_allowlist"):
        gs_samples = index_gas_station(
            cache_dir=cache_dir,
            repo_id=gs.get("path", "gasstation/gs-images-v4"),
            media_type=gs.get("media_type", "synthetic"),
            max_samples=int(gs.get("max_samples", 20000)),
            weeks=gs.get("weeks"),
            download=True,
            seed=int(cfg.get("seed", 42)),
        )
        samples = merge_indices(samples, gs_samples, index_path=index_path)
        logger.info("Merged GAS-Station: total %d samples", len(samples))
    else:
        write_index(samples, index_path)
    return index_path


def make_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=max(labels) + 1).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = [1.0 / counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 3,
) -> Dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        out = model(pixel_values=pv)
        logits = out.logits if hasattr(out, "logits") else out
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return compute_sn34_metrics(logits_np, labels_np, num_classes=num_classes)


def train_loop(cfg: Dict[str, Any], rebuild_index: bool = False) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    index_path = ensure_index(cfg, force_rebuild=rebuild_index)
    samples = read_index(index_path)
    if not samples:
        raise RuntimeError(
            f"No samples in {index_path}. Check downloads / dataset_limit / network."
        )

    data_cfg = cfg.get("data", {})
    train_rows, val_rows = source_stratified_split(
        samples,
        val_fraction=float(data_cfg.get("val_source_fraction", 0.1)),
        stratify_by=str(data_cfg.get("stratify_by", "generator_family")),
        seed=seed,
    )
    split_info = summarize_split(
        train_rows, val_rows, str(data_cfg.get("stratify_by", "generator_family"))
    )
    logger.info("Split: %s", json.dumps(split_info, indent=2))

    model_cfg = cfg["model"]
    image_size = int(model_cfg.get("image_size", 224))
    rob_prob = float(data_cfg.get("robustness_aug_prob", 0.0))
    train_ds = ImageIndexDataset(
        train_rows, image_size=image_size, train=True, robustness_aug_prob=rob_prob, seed=seed
    )
    val_ds = ImageIndexDataset(
        val_rows, image_size=image_size, train=False, robustness_aug_prob=0.0, seed=seed
    )

    train_cfg = cfg["train"]
    train_labels = [s.label for s in train_rows]
    sampler = make_balanced_sampler(train_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("train_batch_size", 32)),
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 4)),
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("eval_batch_size", 64)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_train_model(cfg).to(device)
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        resume_path = Path(str(resume_from)).expanduser()
        if resume_path.is_file():
            ckpt0 = torch.load(resume_path, map_location="cpu")
            state = ckpt0.get("model", ckpt0)
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info(
                "Resumed weights from %s (missing=%d unexpected=%d)",
                resume_path,
                len(missing),
                len(unexpected),
            )
            model.to(device)
        else:
            logger.warning("resume_from not found: %s", resume_path)

    label_smoothing = float(model_cfg.get("label_smoothing", 0.0))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    epochs = int(train_cfg.get("num_epochs", 5))
    total_steps = max(1, epochs * len(train_loader))
    warmup = int(total_steps * float(train_cfg.get("warmup_ratio", 0.05)))

    def lr_at(step: int) -> float:
        base = float(train_cfg.get("learning_rate", 2e-5))
        if step < warmup:
            return base * float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return base * 0.5 * (1.0 + math.cos(math.pi * progress))

    out_dir = Path(train_cfg.get("output_dir", "runs/exp")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    best_metric = -1.0
    best_path = out_dir / "best.pt"
    patience = int(train_cfg.get("early_stop_patience", 2))
    bad_epochs = 0
    metric_name = str(train_cfg.get("metric_for_best", "sn34_score"))
    global_step = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            lr = lr_at(global_step)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            out = model(pixel_values=pv)
            logits = out.logits if hasattr(out, "logits") else out
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
            global_step += 1
            if global_step % int(train_cfg.get("logging_steps", 50)) == 0:
                logger.info(
                    "epoch=%d step=%d loss=%.4f lr=%.2e",
                    epoch,
                    global_step,
                    running / max(1, n_batches),
                    lr,
                )

        metrics = evaluate(model, val_loader, device, int(model_cfg.get("num_classes", 3)))
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = running / max(1, n_batches)
        history.append(metrics)
        logger.info("Val epoch %d: %s", epoch, {k: round(v, 4) for k, v in metrics.items()})

        score = float(metrics.get(metric_name, metrics["sn34_score"]))
        if score > best_metric:
            best_metric = score
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "metrics": metrics,
                    "backbone_id": model_cfg.get("backbone"),
                },
                best_path,
            )
            logger.info("New best %s=%.4f -> %s", metric_name, best_metric, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Load best and optional temperature fit
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    temperature = 1.0
    if model_cfg.get("fit_temperature"):
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pv = batch["pixel_values"].to(device)
                out = model(pixel_values=pv)
                logits = out.logits if hasattr(out, "logits") else out
                all_logits.append(logits.float().cpu().numpy())
                all_labels.append(batch["labels"].numpy())
        temperature = fit_temperature(
            np.concatenate(all_logits), np.concatenate(all_labels)
        )
        logger.info("Fitted temperature T=%.4f", temperature)
        # re-score with T
        scaled = np.concatenate(all_logits) / temperature
        cal_metrics = compute_sn34_metrics(
            scaled, np.concatenate(all_labels), int(model_cfg.get("num_classes", 3))
        )
        logger.info("Calibrated val: %s", {k: round(v, 4) for k, v in cal_metrics.items()})
        ckpt["temperature"] = temperature
        ckpt["calibrated_metrics"] = cal_metrics
        torch.save(ckpt, best_path)

    # Export submission bundle
    export_cfg = cfg.get("export", {})
    wrapper = build_export_wrapper(model.cpu(), temperature=temperature)
    zip_out = export_submission(
        wrapper,
        submission_dir=export_cfg.get("submission_dir", "submission"),
        backbone_id=str(model_cfg.get("backbone")),
        num_classes=int(model_cfg.get("num_classes", 3)),
        resize=export_cfg.get("resize", [image_size, image_size]),
        temperature=temperature,
        zip_path=export_cfg.get("zip_path", "submission/image_detector.zip"),
        name=str(cfg.get("experiment_name", "sn34-image-detector")),
    )
    logger.info("Exported submission to %s", zip_out)

    # King margin check (Phase 3)
    king = train_cfg.get("king_sn34_score")
    margin = float(train_cfg.get("push_margin", 0.015))
    push_ready = None
    if king is not None:
        local_score = float(
            ckpt.get("calibrated_metrics", ckpt.get("metrics", {})).get(
                "sn34_score", best_metric
            )
        )
        push_ready = local_score >= float(king) + margin
        logger.info(
            "King check: local=%.4f king=%.4f margin=%.4f push_ready=%s",
            local_score,
            float(king),
            margin,
            push_ready,
        )

    result = {
        "best_metric": best_metric,
        "metric_name": metric_name,
        "temperature": temperature,
        "best_checkpoint": str(best_path),
        "submission": str(zip_out),
        "history": history,
        "split_info": split_info,
        "push_ready": push_ready,
    }
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Train SN34 image discriminator")
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs/train_vit_phase1.yaml"),
        help="YAML config path",
    )
    p.add_argument("--rebuild-index", action="store_true", help="Force re-download/index")
    p.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Limit number of registry datasets (smoke test)",
    )
    p.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Override max images per source",
    )
    from src.paths import add_path_arguments

    add_path_arguments(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    from src.paths import apply_path_overrides

    apply_path_overrides(
        cfg,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        index_path=args.index_path,
        output_dir=args.output_dir,
        submission_dir=args.submission_dir,
        zip_path=args.zip_path,
    )
    if args.dataset_limit is not None:
        cfg.setdefault("data", {})["dataset_limit"] = args.dataset_limit
    if args.max_per_source is not None:
        cfg.setdefault("data", {})["max_per_source"] = args.max_per_source
    logger.info(
        "Paths: cache_dir=%s index_path=%s output_dir=%s submission_dir=%s zip_path=%s",
        cfg.get("cache_dir"),
        cfg.get("index_path"),
        (cfg.get("train") or {}).get("output_dir"),
        (cfg.get("export") or {}).get("submission_dir"),
        (cfg.get("export") or {}).get("zip_path"),
    )
    result = train_loop(cfg, rebuild_index=args.rebuild_index)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
