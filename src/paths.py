"""Apply CLI path overrides onto a loaded YAML config.

Default external root: ``/mnt/sn34`` (override with ``SN34_DATA_ROOT`` or ``--data-dir``).

Layout:

    /mnt/sn34/
      hf/                 # dataset snapshot downloads
      yaml/               # registry YAML cache
      hf_datasets/        # datasets lib cache
      .hf_home/           # huggingface hub / transformers cache
      index.jsonl
      repo_sizes.json
      runs/...
      submission/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.storage import DEFAULT_DATA_ROOT, default_paths, ensure_data_layout


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
    use_mnt_defaults: bool = True,
) -> Dict[str, Any]:
    """Mutate and return cfg with absolute path overrides.

    When ``use_mnt_defaults`` is True and no explicit data/cache path is given,
    all train artifacts are forced onto ``/mnt/sn34`` (or ``SN34_DATA_ROOT``).
    """
    data = _expand(data_dir)
    cache = _expand(cache_dir)
    index = _expand(index_path)

    if use_mnt_defaults and data is None and cache is None:
        # Always prefer external volume unless the user passed explicit paths.
        defaults = default_paths(DEFAULT_DATA_ROOT)
        data = defaults["data_dir"]
        if index is None:
            index = defaults["index_path"]
        if output_dir is None and not _is_absolute_under_mnt(cfg.get("train", {}).get("output_dir")):
            output_dir = defaults["output_dir"]
        if submission_dir is None and not _is_absolute_under_mnt(
            (cfg.get("export") or {}).get("submission_dir")
        ):
            submission_dir = defaults["submission_dir"]
        if zip_path is None and not _is_absolute_under_mnt((cfg.get("export") or {}).get("zip_path")):
            zip_path = defaults["zip_path"]
        cfg.setdefault("chunk", {}).setdefault("catalog_path", defaults["catalog_path"])
    elif data is not None:
        ensure_data_layout(Path(data))
        if cache is None:
            cache = data
        if index is None:
            index = str(Path(data) / "index.jsonl")
        # Explicit --data-dir always co-locates runs/submission on that volume
        if output_dir is None:
            phase = Path(str(cfg.get("train", {}).get("output_dir", "runs/vit_phase1"))).name
            output_dir = str(Path(data) / "runs" / phase)
        if submission_dir is None:
            submission_dir = str(Path(data) / "submission")
        if zip_path is None:
            zip_path = str(Path(data) / "submission" / "image_detector.zip")
        cfg.setdefault("chunk", {})["catalog_path"] = str(Path(data) / "repo_sizes.json")
    elif cache is not None:
        ensure_data_layout(Path(cache))

    if data is not None and cache is None:
        cache = data

    if cache is not None:
        cfg["cache_dir"] = cache
        ensure_data_layout(Path(cache))

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
    elif use_mnt_defaults and cfg.get("out") is None:
        out_dir = Path(cfg.get("train", {}).get("output_dir", DEFAULT_DATA_ROOT / "runs"))
        cfg.setdefault("out", str(out_dir / "local_sn34_report.json"))

    return cfg


def _is_absolute_under_mnt(path: Optional[str]) -> bool:
    if not path:
        return False
    p = Path(path).expanduser()
    return p.is_absolute() and str(p).startswith("/mnt/")


def add_path_arguments(parser) -> None:
    """Register shared path CLI flags on an argparse parser."""
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            f"Root for datasets/index/runs/submission on external storage "
            f"(default: {DEFAULT_DATA_ROOT} or $SN34_DATA_ROOT)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace / YAML download cache (defaults to --data-dir).",
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
