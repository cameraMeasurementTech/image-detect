"""Unit checks for the gasbench-compatible scorer (no GPU, no downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.download import IndexedSample
from src.eval.score_model import select_samples
from src.metrics.gasbench_score import GasbenchMetrics, blend_sn34, provenance_weights


def test_perfect_multiclass():
    m = GasbenchMetrics(3)
    # balanced perfect predictions
    for label in (0, 1, 2):
        for _ in range(20):
            probs = np.zeros(3)
            probs[label] = 1.0
            m.update(label, label, probs, weight=1.0)
    assert abs(m.accuracy() - 1.0) < 1e-9
    assert abs(m.multiclass_mcc() - 1.0) < 1e-6
    assert m.multiclass_brier() < 1e-9
    assert abs(m.sn34_score() - 1.0) < 1e-6


def test_random_baseline_brier():
    m = GasbenchMetrics(3)
    uniform = np.full(3, 1.0 / 3.0)
    for label in (0, 1, 2):
        for _ in range(30):
            m.update(label, 0, uniform, weight=1.0)
    # uniform guesser Brier is (K-1)/K = 2/3, so sn34 brier term is 0
    assert abs(m.multiclass_brier() - (2.0 / 3.0)) < 1e-6
    assert m.sn34_score() < 1e-4


def test_blend_and_shares():
    assert abs(blend_sn34(0.8, 0.4, 0.25) - (0.75 * 0.8 + 0.25 * 0.4)) < 1e-9
    sources = ["public", "public", "gas_station"]
    w = provenance_weights(sources, {"public": 0.7, "gas_station": 0.3, "holdout": 0.5})
    # holdout dropped; remaining 0.7/0.3 ratio is preserved after rescaling
    assert abs((w[0] + w[1]) / w[2] - (0.7 / 0.3)) < 1e-6


def test_small_cap():
    rows = []
    for ds in ("a", "b"):
        for i in range(150):
            rows.append(
                IndexedSample(
                    path=f"/tmp/{ds}_{i}.jpg",
                    label=0,
                    dataset_name=ds,
                    generator_family="real",
                    media_type="real",
                )
            )
    small = select_samples(rows, "small", per_dataset=100, seed=1)
    full = select_samples(rows, "full")
    assert len(small) == 200
    assert len(full) == 300


def main():
    test_perfect_multiclass()
    test_random_baseline_brier()
    test_blend_and_shares()
    test_small_cap()
    print("eval scorer checks passed")


if __name__ == "__main__":
    main()
