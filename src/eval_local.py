#!/usr/bin/env python3
"""Run local gasbench gates and optional king-margin check."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_gasbench(
    model_dir: Path,
    mode: str,
    gasbench_cmd: str = "gasbench",
    timeout_s: Optional[int] = None,
) -> Dict[str, Any]:
    if shutil.which(gasbench_cmd.split()[0]) is None and gasbench_cmd == "gasbench":
        return {
            "ok": False,
            "error": (
                "gasbench CLI not found on PATH. Install from "
                "https://github.com/BitMind-AI/gasbench then re-run."
            ),
            "mode": mode,
        }

    cmd = [
        *gasbench_cmd.split(),
        "run",
        "--image-model",
        str(model_dir),
        f"--{mode}",
    ]
    print("Running:", " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout_s}s", "mode": mode}

    result: Dict[str, Any] = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "mode": mode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    # Best-effort parse of accuracy / sn34 from stdout/json dumps
    for line in proc.stdout.splitlines()[::-1]:
        lower = line.lower()
        if "sn34" in lower or "accuracy" in lower:
            result.setdefault("signal_lines", []).append(line.strip())
            if len(result["signal_lines"]) >= 20:
                break
    return result


def check_exam_gate(small_result: Dict[str, Any], min_acc: float = 0.80) -> Dict[str, Any]:
    """Interpret --small result for the 80% entrance exam gate."""
    text = (small_result.get("stdout_tail") or "") + "\n" + "\n".join(
        small_result.get("signal_lines") or []
    )
    acc = None
    for token in text.replace(",", " ").split():
        if token.startswith("0.") or token.endswith("%"):
            try:
                if token.endswith("%"):
                    acc = float(token[:-1]) / 100.0
                else:
                    val = float(token)
                    if 0.0 <= val <= 1.0:
                        acc = val
            except ValueError:
                pass
    passed = bool(small_result.get("ok")) and (acc is None or acc >= min_acc)
    return {
        "exam_passed": passed,
        "parsed_accuracy": acc,
        "min_accuracy": min_acc,
        "note": (
            "If accuracy could not be parsed, rely on gasbench exit code / printed report."
            if acc is None
            else "Parsed accuracy from gasbench output (best-effort)."
        ),
    }


def king_margin_check(
    local_sn34: Optional[float],
    king_sn34: Optional[float],
    margin: float = 0.015,
) -> Dict[str, Any]:
    if local_sn34 is None or king_sn34 is None:
        return {
            "push_ready": None,
            "reason": "Provide --local-sn34 and --king-sn34 to evaluate push margin.",
        }
    ready = float(local_sn34) >= float(king_sn34) + float(margin)
    return {
        "push_ready": ready,
        "local_sn34": float(local_sn34),
        "king_sn34": float(king_sn34),
        "margin": float(margin),
        "delta": float(local_sn34) - float(king_sn34),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Score a submission with the gasbench SN34 formula on a local index"
    )
    p.add_argument("--config", default=str(ROOT / "configs/eval_local.yaml"))
    p.add_argument("--model-dir", default=None)
    p.add_argument("--index", default=None)
    p.add_argument("--mode", choices=["small", "full"], default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--aug-weight", type=float, default=None)
    p.add_argument("--king-sn34", type=float, default=None)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--use-gasbench",
        action="store_true",
        help="Shell out to the gasbench CLI instead of the local scorer",
    )
    return p.parse_args()


def _cfg_value(cli, cfg, key, default=None):
    if cli is not None:
        return cli
    return cfg.get(key, default)


def main():
    args = parse_args()
    cfg = load_config(Path(args.config)) if Path(args.config).exists() else {}

    if args.use_gasbench:
        model_dir = Path(args.model_dir or cfg.get("model_dir") or ROOT / "submission")
        mode = args.mode or cfg.get("mode") or "small"
        result = run_gasbench(model_dir, mode)
        result["exam_gate"] = check_exam_gate(result) if mode == "small" else None
        out = Path(args.out or cfg.get("out") or ROOT / "runs/eval_local_result.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    from src.eval.score_model import score_submission, write_report

    model_dir = Path(args.model_dir or cfg.get("model_dir") or ROOT / "submission").expanduser()
    index_path = Path(args.index or cfg.get("index_path")).expanduser()
    mode = args.mode or cfg.get("mode") or "small"
    king = args.king_sn34 if args.king_sn34 is not None else cfg.get("king_sn34")
    report = score_submission(
        model_dir=model_dir,
        index_path=index_path,
        mode=mode,
        batch_size=int(_cfg_value(args.batch_size, cfg, "batch_size", 32)),
        seed=int(cfg.get("seed", 42)),
        aug_weight=float(_cfg_value(args.aug_weight, cfg, "aug_weight", 0.0)),
        shares=cfg.get("shares"),
        king_sn34=None if king is None else float(king),
        koth_margin=float(cfg.get("koth_margin", 0.01)),
    )
    summary = {k: v for k, v in report.items() if k != "per_dataset"}
    summary["n_datasets_scored"] = len(report.get("per_dataset") or {})
    out = Path(args.out or cfg.get("out") or ROOT / "runs/local_sn34_report.json")
    write_report(report, out)
    print(json.dumps(summary, indent=2))
    print(f"Full report: {out}")
    exam = report.get("exam_gate") or {}
    if exam.get("applies") and exam.get("passed") is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
