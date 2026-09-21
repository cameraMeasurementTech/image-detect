"""Local SN34 image eval that follows the gasbench scoring contract.

Validators do not score discriminator weights. They set chain weights from the
kings API after GAS/gasbench computes sn34_score. This runner reproduces that
scoring on a local index:

- uint8 RGB NCHW input, logits out, softmax inside the scorer
- 3-class image labels (0 real, 1 synthetic, 2 semisynthetic)
- weighted Gorodkin MCC + multiclass Brier, geometric sn34 (alpha 1.2, beta 1.8)
- optional robustness pass blended as (1-w)*base + w*aug
- small mode: up to 100 images per dataset (gasbench --small archive cap)
- full mode: every indexed image

Private holdouts are not in the public registries, so a local score is an
upper bound on the network full bench.
"""

from __future__ import annotations

import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageFile

from src.data.augment import apply_robustness_augmentations
from src.data.download import IndexedSample, read_index
from src.metrics.gasbench_score import GasbenchMetrics, blend_sn34, provenance_weights

ImageFile.LOAD_TRUNCATED_IMAGES = True

ID2LABEL = {0: "real", 1: "synthetic", 2: "semisynthetic"}
EXAM_ACCURACY = 0.80
KOTH_MARGIN = 0.01
SMALL_PER_DATASET = 100


def load_submission(model_dir: Path, device: torch.device):
    model_dir = Path(model_dir)
    cfg_path = model_dir / "model_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"model_config.yaml not found in {model_dir}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_cfg = cfg.get("model") or {}
    pre = cfg.get("preprocessing") or {}
    resize = pre.get("resize") or [224, 224]
    height, width = int(resize[0]), int(resize[1])
    num_classes = int(model_cfg.get("num_classes", 3))
    weights = model_dir / model_cfg.get("weights_file", "model.safetensors")

    spec = importlib.util.spec_from_file_location("sn34_submission", model_dir / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {model_dir / 'model.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "load_model"):
        raise RuntimeError("model.py must define load_model()")

    kwargs = {
        "temperature": model_cfg.get("temperature", 1.0),
        "mean": model_cfg.get("mean", [0.485, 0.456, 0.406]),
        "std": model_cfg.get("std", [0.229, 0.224, 0.225]),
    }
    model = module.load_model(str(weights), num_classes=num_classes, **kwargs)
    model.to(device)
    model.eval()
    return model, cfg, (height, width), num_classes


def select_samples(
    samples: Sequence[IndexedSample],
    mode: str,
    per_dataset: int = SMALL_PER_DATASET,
    seed: int = 42,
) -> List[IndexedSample]:
    if mode == "full":
        return list(samples)
    if mode != "small":
        raise ValueError("mode must be 'small' or 'full'")
    rng = random.Random(seed)
    by_ds: Dict[str, List[IndexedSample]] = defaultdict(list)
    for s in samples:
        by_ds[s.dataset_name].append(s)
    chosen: List[IndexedSample] = []
    for name in sorted(by_ds):
        rows = list(by_ds[name])
        rng.shuffle(rows)
        chosen.extend(rows[:per_dataset])
    return chosen


def _prepare_uint8(path: str, size: Tuple[int, int], robust: bool) -> Optional[torch.Tensor]:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    if robust:
        img = apply_robustness_augmentations(img, seed=hash(path) % (10**9))
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8
    return tensor


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@torch.no_grad()
def run_pass(
    model: torch.nn.Module,
    samples: Sequence[IndexedSample],
    size: Tuple[int, int],
    device: torch.device,
    batch_size: int,
    num_classes: int,
    weights: np.ndarray,
    robust: bool,
) -> GasbenchMetrics:
    metrics = GasbenchMetrics(num_classes=num_classes)
    batch_tensors: List[torch.Tensor] = []
    batch_meta: List[Tuple[int, float]] = []

    def flush() -> None:
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device)
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        logits_np = logits.detach().float().cpu().numpy()
        probs = _softmax_np(logits_np)
        for i, (label, weight) in enumerate(batch_meta):
            pred = int(np.argmax(probs[i]))
            metrics.update(label, pred, probs[i], weight=float(weight))
        batch_tensors.clear()
        batch_meta.clear()

    for sample, weight in zip(samples, weights):
        tensor = _prepare_uint8(sample.path, size, robust=robust)
        if tensor is None:
            continue
        batch_tensors.append(tensor)
        batch_meta.append((int(sample.label), float(weight)))
        if len(batch_tensors) >= batch_size:
            flush()
    flush()
    return metrics


def per_dataset_accuracy(
    model: torch.nn.Module,
    samples: Sequence[IndexedSample],
    size: Tuple[int, int],
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Dict[str, Dict[str, Any]]:
    """One unweighted metrics object per dataset_name for the report."""
    grouped: Dict[str, List[IndexedSample]] = defaultdict(list)
    for s in samples:
        grouped[s.dataset_name].append(s)
    out: Dict[str, Dict[str, Any]] = {}
    for name, rows in grouped.items():
        w = np.ones(len(rows), dtype=np.float64)
        m = run_pass(model, rows, size, device, batch_size, num_classes, w, robust=False)
        out[name] = {
            "media_type": rows[0].media_type,
            "generator_family": rows[0].generator_family,
            "n": len(rows),
            "accuracy": m.accuracy(),
            "sn34_score": m.sn34_score(multiclass=True),
        }
    return out


def score_submission(
    model_dir: Path,
    index_path: Path,
    mode: str = "small",
    batch_size: int = 32,
    seed: int = 42,
    aug_weight: float = 0.0,
    shares: Optional[Dict[str, float]] = None,
    king_sn34: Optional[float] = None,
    koth_margin: float = KOTH_MARGIN,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, size, num_classes = load_submission(Path(model_dir), device)
    samples = read_index(index_path)
    selected = select_samples(samples, mode=mode, seed=seed)
    if not selected:
        raise RuntimeError(f"No samples selected from {index_path} (mode={mode})")

    source_alias = {"registry": "public", "gas_station": "gas_station"}
    sources = [source_alias.get(s.source, s.source) for s in selected]
    if shares:
        weights = provenance_weights(sources, shares)
    else:
        weights = np.ones(len(selected), dtype=np.float64)

    base = run_pass(
        model, selected, size, device, batch_size, num_classes, weights, robust=False
    )
    base_report = base.report(multiclass=True)

    aug_report = None
    final_sn34 = base_report["sn34_score"]
    if aug_weight > 0:
        aug = run_pass(
            model, selected, size, device, batch_size, num_classes, weights, robust=True
        )
        aug_report = aug.report(multiclass=True)
        final_sn34 = blend_sn34(base_report["sn34_score"], aug_report["sn34_score"], aug_weight)

    by_ds = per_dataset_accuracy(
        model, selected, size, device, batch_size, num_classes
    )
    exam_passed = None
    if mode == "small":
        exam_passed = base_report["accuracy"] >= EXAM_ACCURACY

    dethrone = None
    if king_sn34 is not None:
        dethrone = final_sn34 >= float(king_sn34) + float(koth_margin)

    return {
        "model_dir": str(model_dir),
        "index_path": str(index_path),
        "mode": mode,
        "device": str(device),
        "image_size": list(size),
        "num_classes": num_classes,
        "n_samples": len(selected),
        "n_datasets": len({s.dataset_name for s in selected}),
        "scoring": {
            "formula": "sqrt(clip((MCC+1)/2,0,1)^1.2 * max(0,(B0-B)/B0)^1.8)",
            "mode": "multiclass",
            "b0": (num_classes - 1) / num_classes,
            "aug_weight": aug_weight,
            "shares": shares,
            "holdouts_included": False,
        },
        "base": base_report,
        "aug": aug_report,
        "sn34_score": final_sn34,
        "exam_gate": {
            "applies": mode == "small",
            "min_accuracy": EXAM_ACCURACY,
            "passed": exam_passed,
        },
        "koth": {
            "king_sn34": king_sn34,
            "margin": koth_margin,
            "dethrone": dethrone,
        },
        "per_dataset": by_ds,
        "model_config_name": (cfg.get("name") if isinstance(cfg, dict) else None),
    }


def write_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # per_dataset can be large; keep it
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
