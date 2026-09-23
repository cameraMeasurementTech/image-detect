"""External storage root for SN34 train data (default: /mnt/sn34).

All downloads, indexes, runs, and submission artifacts should live under this
tree so the project disk and home cache stay small.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Override with env SN34_DATA_ROOT=/mnt/your-volume/sn34
DEFAULT_DATA_ROOT = Path(os.environ.get("SN34_DATA_ROOT", "/mnt/sn34")).expanduser()


def data_root(override: Optional[str] = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return Path(DEFAULT_DATA_ROOT).expanduser().resolve()


def ensure_data_layout(root: Optional[Path] = None) -> Path:
    """Create the standard layout on the external volume and point HF caches there."""
    root = Path(root or DEFAULT_DATA_ROOT).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for sub in (
        "hf",
        "yaml",
        "runs",
        "runs/vit_phase1",
        "runs/vit_phase2",
        "runs/convnext_phase3",
        "submission",
        "submission_probe",
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)

    # Keep HuggingFace / datasets caches on the same volume (not ~/.cache).
    hf_home = root / ".hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf_datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    (root / "hf_datasets").mkdir(parents=True, exist_ok=True)

    try:
        usage = disk_usage(root)
        logger.info(
            "Data root %s — total=%.2f TiB used=%.2f TiB free=%.2f TiB",
            root,
            usage["total_tib"],
            usage["used_tib"],
            usage["free_tib"],
        )
        if usage["free_tib"] < 0.4:
            logger.warning(
                "Less than 0.4 TiB free under %s — chunk budget may not fit",
                root,
            )
    except OSError as exc:
        logger.warning("Could not stat %s: %s", root, exc)

    return root.resolve()


def disk_usage(path: Path) -> dict:
    import shutil

    usage = shutil.disk_usage(path)
    tib = 1024**4
    return {
        "total_tib": usage.total / tib,
        "used_tib": usage.used / tib,
        "free_tib": usage.free / tib,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def default_paths(root: Optional[Path] = None) -> dict:
    root = ensure_data_layout(root)
    return {
        "data_dir": str(root),
        "cache_dir": str(root),
        "index_path": str(root / "index.jsonl"),
        "output_dir": str(root / "runs" / "vit_phase1"),
        "submission_dir": str(root / "submission"),
        "zip_path": str(root / "submission" / "image_detector.zip"),
        "catalog_path": str(root / "repo_sizes.json"),
    }
