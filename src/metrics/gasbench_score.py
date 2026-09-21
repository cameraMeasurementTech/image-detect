"""Gasbench-compatible SN34 metrics (weighted MCC, Brier, geometric sn34).

Mirrors gasbench ``benchmarks/utils/metrics.py`` for image (K=3, multiclass).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class GasbenchMetrics:
    """Accumulates weighted binary and multiclass stats the way gasbench does."""

    def __init__(self, num_classes: int = 3):
        self.num_classes = max(2, int(num_classes))
        self.true_positives = 0.0
        self.true_negatives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0
        self.binary_y_true: list = []
        self.binary_probs: list = []
        self.binary_weights: list = []
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=float)
        self.mc_sq_error = 0.0
        self.mc_weight = 0.0
        self.correct_weight = 0.0
        self.total_weight = 0.0

    def update(
        self,
        label: int,
        pred: int,
        pred_probs: Optional[np.ndarray] = None,
        weight: float = 1.0,
    ) -> None:
        k = self.num_classes
        binary_label = 0 if int(label) == 0 else 1

        if pred_probs is None or len(pred_probs) == 0:
            p_not_real = None
        elif len(pred_probs) >= 2:
            p_not_real = float(1.0 - pred_probs[0])
        else:
            p_not_real = float(pred_probs[0])

        if p_not_real is None:
            binary_pred = 0 if int(pred) == 0 else 1
        else:
            binary_pred = int(p_not_real > 0.5)

        if binary_label == 1 and binary_pred == 1:
            self.true_positives += weight
        elif binary_label == 0 and binary_pred == 0:
            self.true_negatives += weight
        elif binary_label == 0 and binary_pred == 1:
            self.false_positives += weight
        else:
            self.false_negatives += weight

        if p_not_real is not None:
            self.binary_y_true.append(binary_label)
            self.binary_weights.append(weight)
            self.binary_probs.append(p_not_real)

        t = int(label)
        if not (0 <= t < k):
            t = min(max(t, 0), k - 1)

        probs = None
        if pred_probs is not None and len(pred_probs) > 0:
            probs = np.zeros(k, dtype=float)
            src = np.asarray(pred_probs, dtype=float).ravel()
            if len(src) == 1:
                probs[0] = 1.0 - float(src[0])
                probs[1] = float(src[0])
            else:
                n = min(k, len(src))
                probs[:n] = src[:n]

        p = int(pred)
        if not (0 <= p < k):
            if probs is not None and probs.sum() > 0:
                p = int(np.argmax(probs))
            else:
                p = (t + 1) % k

        self.confusion[t, p] += weight
        self.total_weight += weight
        if t == p:
            self.correct_weight += weight

        if probs is not None:
            onehot = np.zeros(k, dtype=float)
            onehot[t] = 1.0
            self.mc_sq_error += weight * float(np.sum((probs - onehot) ** 2))
            self.mc_weight += weight

    def binary_mcc(self) -> float:
        num = (self.true_positives * self.true_negatives) - (
            self.false_positives * self.false_negatives
        )
        den = np.sqrt(
            (self.true_positives + self.false_positives)
            * (self.true_positives + self.false_negatives)
            * (self.true_negatives + self.false_positives)
            * (self.true_negatives + self.false_negatives)
        )
        return float(num / den) if den > 0 else 0.0

    def binary_brier(self) -> float:
        if not self.binary_y_true:
            return 0.25
        y = np.asarray(self.binary_y_true, dtype=float)
        p = np.clip(np.asarray(self.binary_probs, dtype=float), 1e-7, 1 - 1e-7)
        w = np.asarray(self.binary_weights, dtype=float)
        return float(np.average((p - y) ** 2, weights=w))

    def multiclass_mcc(self) -> float:
        c = self.confusion
        n = float(c.sum())
        if n <= 0:
            return 0.0
        t = c.sum(axis=1)
        p = c.sum(axis=0)
        numerator = n * float(np.trace(c)) - float(np.dot(t, p))
        denominator = np.sqrt(
            max(0.0, n * n - float(np.dot(p, p))) * max(0.0, n * n - float(np.dot(t, t)))
        )
        return float(numerator / denominator) if denominator > 0 else 0.0

    def multiclass_brier(self) -> float:
        if self.mc_weight <= 0:
            return (self.num_classes - 1) / self.num_classes
        return float(self.mc_sq_error / self.mc_weight)

    def accuracy(self) -> float:
        if self.total_weight <= 0:
            return 0.0
        return float(self.correct_weight / self.total_weight)

    def sn34_score(self, multiclass: bool = True, alpha: float = 1.2, beta: float = 1.8) -> float:
        if multiclass:
            mcc = self.multiclass_mcc()
            brier = self.multiclass_brier()
            baseline = (self.num_classes - 1) / self.num_classes
        else:
            mcc = self.binary_mcc()
            brier = self.binary_brier()
            baseline = 0.25
        mcc_norm = max(0.0, min((mcc + 1.0) / 2.0, 1.0)) ** alpha
        brier_score = max(0.0, (baseline - brier) / baseline) ** beta
        return float(max(0.0, min((max(1e-12, mcc_norm * brier_score)) ** 0.5, 1.0)))

    def report(self, multiclass: bool = True) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy(),
            "n_weight": float(self.total_weight),
            "gorodkin_mcc": self.multiclass_mcc(),
            "multiclass_brier": self.multiclass_brier(),
            "binary_mcc": self.binary_mcc(),
            "binary_brier": self.binary_brier(),
            "sn34_score": self.sn34_score(multiclass=multiclass),
            "binary_sn34_score": self.sn34_score(multiclass=False),
            "multiclass_sn34_score": self.sn34_score(multiclass=True),
        }


def blend_sn34(base: float, aug: float, aug_weight: float) -> float:
    """gasbench finalize: sn34 = (1-w)*base + w*aug."""
    w = float(aug_weight)
    return float((1.0 - w) * base + w * aug)


def provenance_weights(
    sources: list,
    shares: Dict[str, float],
) -> np.ndarray:
    """Convert target score shares into per-sample weights.

    Missing groups are dropped and the remaining shares renormalized, matching
    gasbench when a configured provenance bucket is absent.
    """
    present = {s for s in sources}
    active = {k: float(v) for k, v in shares.items() if k in present and v > 0}
    total_share = sum(active.values())
    if total_share <= 0:
        return np.ones(len(sources), dtype=np.float64)
    counts = {k: sum(1 for s in sources if s == k) for k in active}
    weights = np.ones(len(sources), dtype=np.float64)
    for i, src in enumerate(sources):
        if src not in active or counts[src] <= 0:
            weights[i] = 0.0
            continue
        weights[i] = (active[src] / total_share) / counts[src]
    # rescale so mean weight of kept samples is 1 (absolute scale does not change MCC/Brier)
    kept = weights[weights > 0]
    if len(kept):
        weights = weights / kept.mean()
    return weights
