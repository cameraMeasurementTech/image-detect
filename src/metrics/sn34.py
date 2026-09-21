"""SN34 scoring metrics: multiclass MCC (Gorodkin), Brier, and sn34_score."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def gorodkin_rk(cm: np.ndarray) -> float:
    """Gorodkin's multiclass generalization of MCC (R_K)."""
    cm = cm.astype(np.float64)
    n = cm.sum()
    if n == 0:
        return 0.0
    correct = np.trace(cm)
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    num = correct * n - float(np.dot(row, col))
    den_left = n**2 - float(np.dot(col, col))
    den_right = n**2 - float(np.dot(row, row))
    den = np.sqrt(max(den_left, 0.0) * max(den_right, 0.0))
    if den <= 0:
        return 0.0
    return float(num / den)


def multiclass_brier(probs: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    """Mean of sum_k (p_k - y_k)^2."""
    y = np.asarray(y_true, dtype=int)
    onehot = np.eye(num_classes, dtype=np.float64)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=-1)))


def binary_collapse_probs(probs: np.ndarray) -> np.ndarray:
    """p_not_real = 1 - p_real (class 0)."""
    return 1.0 - probs[:, 0]


def binary_mcc(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    # y: 0=real, 1=not-real
    tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    tn = np.sum((y_true == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true == 1) & (y_pred_bin == 0))
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        return 0.0
    return float(num / den)


def binary_brier(p_not_real: np.ndarray, y_true_bin: np.ndarray) -> float:
    return float(np.mean((p_not_real - y_true_bin) ** 2))


def normalize_mcc(m: float) -> float:
    return float(np.clip((m + 1.0) / 2.0, 0.0, 1.0) ** 1.2)


def normalize_brier(b: float, b0: float) -> float:
    return float(max(0.0, (b0 - b) / b0) ** 1.8)


def sn34_score_from_components(m: float, b: float, b0: float) -> float:
    return float(np.sqrt(normalize_mcc(m) * normalize_brier(b, b0)))


def compute_sn34_metrics(
    logits: np.ndarray,
    y_true: np.ndarray,
    num_classes: int = 3,
    mode: str = "multiclass",
) -> Dict[str, float]:
    """Compute accuracy, MCC, Brier, and sn34_score.

    mode: 'multiclass' (image/video SN34) or 'binary' (compatibility).
    """
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    probs = softmax(logits)
    y_pred = probs.argmax(axis=-1)
    acc = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0

    cm = confusion_matrix(y_true, y_pred, num_classes)
    m_multi = gorodkin_rk(cm)
    b_multi = multiclass_brier(probs, y_true, num_classes)
    b0_multi = (num_classes - 1) / num_classes

    y_bin = (y_true != 0).astype(int)
    p_nr = binary_collapse_probs(probs)
    y_pred_bin = (p_nr >= 0.5).astype(int)
    m_bin = binary_mcc(y_bin, y_pred_bin)
    b_bin = binary_brier(p_nr, y_bin.astype(np.float64))
    b0_bin = 0.25

    if mode == "binary":
        sn34 = sn34_score_from_components(m_bin, b_bin, b0_bin)
        m, b, b0 = m_bin, b_bin, b0_bin
    else:
        sn34 = sn34_score_from_components(m_multi, b_multi, b0_multi)
        m, b, b0 = m_multi, b_multi, b0_multi

    return {
        "accuracy": acc,
        "gorodkin_mcc": m_multi,
        "multiclass_brier": b_multi,
        "binary_mcc": m_bin,
        "binary_brier": b_bin,
        "m_norm": normalize_mcc(m),
        "b_norm": normalize_brier(b, b0),
        "sn34_score": sn34,
        "n_samples": float(len(y_true)),
    }


def fit_temperature(
    logits: np.ndarray, y_true: np.ndarray, max_iter: int = 100
) -> float:
    """Fit a scalar temperature T on logits for better Brier / calibration."""
    import torch
    import torch.nn.functional as F

    t = torch.nn.Parameter(torch.ones([]))
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=max_iter)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.long)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / t.clamp_min(1e-3), y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp_min(1e-3).item())
