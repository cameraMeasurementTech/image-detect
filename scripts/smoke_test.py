#!/usr/bin/env python3
"""Smoke test: metrics, split, tiny synthetic train, export bundle load."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.download import IndexedSample, write_index
from src.data.gasbench_index import LABEL_MAP, load_registry_entries, summarize_entries
from src.data.split import source_stratified_split
from src.metrics.sn34 import compute_sn34_metrics
from src.models.export_bundle import export_submission
from src.models.vit_detector import Uint8ImageClassifier, load_hf_classifier


def test_sn34_metrics():
    rng = np.random.default_rng(0)
    n, k = 200, 3
    logits = rng.normal(size=(n, k))
    # Make labels correlated with argmax for nonzero MCC
    y = logits.argmax(axis=1)
    y[::5] = (y[::5] + 1) % k
    m = compute_sn34_metrics(logits, y, num_classes=k)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["sn34_score"] <= 1.0
    assert "gorodkin_mcc" in m
    print("sn34_metrics OK", {k: round(v, 4) for k, v in m.items() if k != "n_samples"})


def test_registry_load():
    entries = load_registry_entries(cache_dir=ROOT / ".cache_yaml_test")
    assert len(entries) > 50
    summary = summarize_entries(entries)
    assert summary["by_media_type"]["real"] > 0
    assert summary["by_media_type"]["synthetic"] > 0
    print("registry_load OK", summary["n_datasets"], summary["by_media_type"])


def test_split():
    samples = []
    for fam in ["a", "b", "c", "d", "e"]:
        for i in range(10):
            samples.append(
                IndexedSample(
                    path=f"/tmp/{fam}_{i}.jpg",
                    label=i % 3,
                    dataset_name=f"ds_{fam}",
                    generator_family=fam,
                    media_type=["real", "synthetic", "semisynthetic"][i % 3],
                )
            )
    train, val = source_stratified_split(samples, val_fraction=0.2, seed=0)
    train_f = {s.generator_family for s in train}
    val_f = {s.generator_family for s in val}
    assert train_f.isdisjoint(val_f)
    print("split OK", len(train), len(val), sorted(val_f))


def test_export_and_uint8_forward():
    # Tiny random-init classifier (no download if cached); use a minimal Linear fallback if HF fails
    try:
        backbone = load_hf_classifier("google/vit-base-patch16-224-in21k", num_classes=3)
    except Exception as exc:
        print("HF load skipped:", exc)
        backbone = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 224 * 224, 3),
        )

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.config = type("C", (), {"num_labels": 3, "to_dict": lambda self: {"num_labels": 3}, "to_json_string": lambda self: json.dumps({"num_labels": 3})})()

            def forward(self, pixel_values=None, **kw):
                return type("O", (), {"logits": self.m(pixel_values)})()

            def save_pretrained(self, *a, **k):
                pass

        backbone = Wrap(backbone)

    wrapper = Uint8ImageClassifier(backbone)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # write a few fake images + index for sanity
        img_dir = td / "imgs"
        img_dir.mkdir()
        for i in range(3):
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
                img_dir / f"{i}.jpg"
            )
        out = export_submission(
            wrapper,
            submission_dir=td / "submission",
            backbone_id="google/vit-base-patch16-224-in21k",
            zip_path=td / "image_detector.zip",
        )
        assert out.exists()
        assert (td / "submission" / "model.py").exists()
        assert (td / "submission" / "model_config.yaml").exists()
        assert (td / "submission" / "model.safetensors").exists()

        x = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
        wrapper.eval()
        with torch.no_grad():
            logits = wrapper(x)
        assert logits.shape == (2, 3)
        print("export_and_uint8_forward OK", out)


def main():
    test_sn34_metrics()
    test_registry_load()
    test_split()
    test_export_and_uint8_forward()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
