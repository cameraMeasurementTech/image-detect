"""Apply CLI path overrides onto a loaded YAML config.

Preferred layout when using ``--data-dir /path/to/store``:

    /path/to/store/
      hf/              # HuggingFace dataset downloads (created by downloaders)
      yaml/            # cached registry YAMLs
      index.jsonl      # sample index (unless --index-path set)

Other outputs stay under the project unless overridden:

    --output-dir       training checkpoints / train_result.json
    --submission-dir   model_config.yaml, model.py, weights
    --zip-path         packed image_detector.zip
    --out              eval report JSON
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def _expand(path: Optional[str]) -> Optional[str]:
    if path is None or path == "":
        return None
    return str(Path(path).expanduser().resolve())


def apply_path_overrides(
    cfg: Dict[str, Any],
    *,
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    index_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    submission_dir: Optional[str] = None,
    zip_path: Optional[str] = None,
    report_out: Optional[str] = None,
) -> Dict[str, Any]:
    """Mutate and return cfg with absolute path overrides."""
    data = _expand(data_dir)
    cache = _expand(cache_dir)
    index = _expand(index_path)

    if data is not None:
        if cache is None:
            cache = data
        if index is None:
            index = str(Path(data) / "index.jsonl")

    if cache is not None:
        cfg["cache_dir"] = cache

    if index is not None:
        cfg["index_path"] = index

    if output_dir is not None:
        cfg.setdefault("train", {})["output_dir"] = _expand(output_dir)

    if submission_dir is not None:
        cfg.setdefault("export", {})["submission_dir"] = _expand(submission_dir)

    if zip_path is not None:
        cfg.setdefault("export", {})["zip_path"] = _expand(zip_path)

    if report_out is not None:
        cfg["out"] = _expand(report_out)

    return cfg


def add_path_arguments(parser) -> None:
    """Register shared path CLI flags on an argparse parser."""
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Root directory for downloaded datasets and (by default) index.jsonl. "
            "Overrides cache_dir; sets index_path to <data-dir>/index.jsonl unless "
            "--index-path is also given."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace / YAML download cache (overrides config cache_dir).",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Path to write/read the sample index JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training run directory (checkpoints, train_result.json).",
    )
    parser.add_argument(
        "--submission-dir",
        default=None,
        help="Directory for exported model_config.yaml / model.py / weights.",
    )
    parser.add_argument(
        "--zip-path",
        default=None,
        help="Path for the packed image_detector.zip.",
    )
