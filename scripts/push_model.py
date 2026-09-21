#!/usr/bin/env python3
"""Push submission zip via bitmind-subnet gascli / push_model.py when ready."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBNET = Path("/home/bitmind-subnet")


def main():
    p = argparse.ArgumentParser(description="Push image detector to SN34")
    p.add_argument(
        "--zip",
        default=str(ROOT / "submission/image_detector.zip"),
    )
    p.add_argument("--wallet-name", default="default")
    p.add_argument("--wallet-hotkey", default="default")
    p.add_argument("--netuid", type=int, default=34)
    p.add_argument(
        "--require-push-ready",
        action="store_true",
        help="Abort unless runs/*/train_result.json has push_ready=true",
    )
    p.add_argument(
        "--train-result",
        default="",
        help="Path to train_result.json for margin gate",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"Missing zip: {zip_path}", file=sys.stderr)
        sys.exit(1)

    if args.require_push_ready:
        result_path = Path(args.train_result) if args.train_result else None
        if result_path is None or not result_path.exists():
            # pick latest
            candidates = sorted(ROOT.glob("runs/*/train_result.json"))
            if not candidates:
                print("No train_result.json found", file=sys.stderr)
                sys.exit(1)
            result_path = candidates[-1]
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if data.get("push_ready") is not True:
            print(
                json.dumps(
                    {
                        "error": "push_ready is not true; set king_sn34_score in config and retrain, or omit --require-push-ready",
                        "train_result": str(result_path),
                        "push_ready": data.get("push_ready"),
                        "best_metric": data.get("best_metric"),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            sys.exit(2)

    if shutil.which("gascli"):
        cmd = [
            "gascli",
            "d",
            "push",
            "--image-model",
            str(zip_path),
            "--wallet-name",
            args.wallet_name,
            "--wallet-hotkey",
            args.wallet_hotkey,
            "--netuid",
            str(args.netuid),
        ]
    elif (SUBNET / "neurons/discriminator/push_model.py").exists():
        cmd = [
            sys.executable,
            str(SUBNET / "neurons/discriminator/push_model.py"),
            "--image-model",
            str(zip_path),
            "--wallet-name",
            args.wallet_name,
            "--wallet-hotkey",
            args.wallet_hotkey,
            "--netuid",
            str(args.netuid),
        ]
    else:
        print(
            "Neither gascli nor bitmind-subnet push_model.py found. "
            "Install bitmind-subnet and activate its venv.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Command:", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
