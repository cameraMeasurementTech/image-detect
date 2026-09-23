#!/usr/bin/env python3
"""Check that a model / submission meets SN34 image gates before full data download.

Examples
--------
# Cached HF backbone from phase-1 config (no dataset):
python scripts/check_gates.py --config configs/train_vit_phase1.yaml

# Explicit backbone id:
python scripts/check_gates.py --backbone google/vit-base-patch16-224-in21k

# Existing submission dir or zip:
python scripts/check_gates.py --model-dir /path/to/submission
python scripts/check_gates.py --zip-path /path/to/image_detector.zip

# Optional tiny labeled slice for a Gate-5 accuracy preview (small download):
python scripts/check_gates.py --config configs/train_vit_phase1.yaml \
  --mini-exam --data-dir /mnt/data/sn34-mini --dataset-limit 2 --max-per-source 32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.check_gates import format_report, run_checks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Validate SN34 image discriminator gates (I/O + package) without full dataset"
    )
    p.add_argument("--config", default=None, help="Train YAML (reads model.backbone)")
    p.add_argument("--backbone", default=None, help="HF model id to probe-export")
    p.add_argument("--model-dir", default=None, help="Existing submission directory")
    p.add_argument("--zip-path", default=None, help="Existing image_detector.zip")
    p.add_argument(
        "--keep-probe-dir",
        default=None,
        help="If probing from backbone, write submission here instead of a temp dir",
    )
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument(
        "--mini-exam",
        action="store_true",
        help="Download a tiny public slice and report accuracy (Gate 5 proxy)",
    )
    p.add_argument("--data-dir", default=None, help="Data root for --mini-exam")
    p.add_argument("--dataset-limit", type=int, default=2)
    p.add_argument("--max-per-source", type=int, default=32)
    p.add_argument("--out", default=None, help="Write JSON report path")
    args = p.parse_args()

    if not any([args.model_dir, args.zip_path, args.backbone, args.config]):
        args.config = str(ROOT / "configs/train_vit_phase1.yaml")

    report = run_checks(
        model_dir=Path(args.model_dir) if args.model_dir else None,
        zip_path=Path(args.zip_path) if args.zip_path else None,
        backbone=args.backbone,
        config_path=Path(args.config) if args.config else None,
        num_classes=args.num_classes,
        image_size=args.image_size,
        device=args.device,
        mini_exam=args.mini_exam,
        mini_data_dir=Path(args.data_dir) if args.data_dir else None,
        dataset_limit=args.dataset_limit,
        max_per_source=args.max_per_source,
        keep_probe_dir=Path(args.keep_probe_dir) if args.keep_probe_dir else None,
    )
    text = format_report(report)
    print(text)
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"JSON report: {out}")

    # Structural failure → exit 1; mini-exam failure alone → exit 2
    if not report.structural_ok:
        return 1
    exam = [c for c in report.checks if c.gate == "5" and c.name.startswith("mini-exam")]
    if exam and not exam[0].passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
