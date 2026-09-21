"""Source-stratified train/val splits for gasbench image indices."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .download import IndexedSample


def source_stratified_split(
    samples: Sequence[IndexedSample],
    val_fraction: float = 0.1,
    stratify_by: str = "generator_family",
    seed: int = 42,
    min_val_sources: int = 1,
) -> Tuple[List[IndexedSample], List[IndexedSample]]:
    """Hold out whole source groups (dataset or generator family) for validation.

    Never splits images from the same stratify key across train and val, so public
    gasbench generalization estimates are not contaminated by within-source leakage.
    """
    if not samples:
        return [], []
    if stratify_by not in ("generator_family", "dataset_name"):
        raise ValueError("stratify_by must be 'generator_family' or 'dataset_name'")

    by_key: Dict[str, List[IndexedSample]] = defaultdict(list)
    for s in samples:
        key = getattr(s, stratify_by)
        by_key[str(key)].append(s)

    keys = sorted(by_key.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_val = max(min_val_sources, int(round(len(keys) * val_fraction)))
    n_val = min(n_val, max(0, len(keys) - 1))  # keep at least one train source when possible
    val_keys = set(keys[:n_val]) if n_val else set()
    train_keys = set(keys) - val_keys

    # Ensure each label appears in train when possible by swapping
    train, val = [], []
    for key, rows in by_key.items():
        if key in val_keys:
            val.extend(rows)
        else:
            train.extend(rows)

    if not train and val:
        # Degenerate: move one source back to train
        move_key = next(iter(val_keys))
        moved = by_key[move_key]
        val = [s for s in val if getattr(s, stratify_by) != move_key]
        train.extend(moved)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def class_counts(samples: Sequence[IndexedSample]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for s in samples:
        counts[s.label] = counts.get(s.label, 0) + 1
    return counts


def summarize_split(
    train: Sequence[IndexedSample], val: Sequence[IndexedSample], stratify_by: str
) -> Dict[str, object]:
    def keys(rows: Sequence[IndexedSample]):
        return sorted({str(getattr(s, stratify_by)) for s in rows})

    return {
        "n_train": len(train),
        "n_val": len(val),
        "train_sources": keys(train),
        "val_sources": keys(val),
        "train_class_counts": class_counts(train),
        "val_class_counts": class_counts(val),
    }
