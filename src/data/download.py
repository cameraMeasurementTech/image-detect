"""Download HuggingFace image datasets listed in gasbench registries into a local index."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .gasbench_index import RegistryEntry, load_registry_entries

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


@dataclass
class IndexedSample:
    path: str
    label: int
    dataset_name: str
    generator_family: str
    media_type: str
    source: str = "registry"  # registry | gas_station

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "dataset_name": self.dataset_name,
            "generator_family": self.generator_family,
            "media_type": self.media_type,
            "source": self.source,
        }


def expand_cache_dir(cache_dir: Union[str, Path]) -> Path:
    return Path(cache_dir).expanduser().resolve()


def write_index(samples: Sequence[IndexedSample], index_path: Union[str, Path]) -> Path:
    path = Path(index_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")
    return path


def read_index(index_path: Union[str, Path]) -> List[IndexedSample]:
    path = Path(index_path).expanduser()
    samples: List[IndexedSample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(IndexedSample(**row))
    return samples


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _collect_local_images(root: Path, max_files: int) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if _is_image_file(p):
            files.append(p)
            if max_files > 0 and len(files) >= max_files:
                break
    return files


def download_hf_repo(
    repo_id: str,
    local_dir: Path,
    revision: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
    }
    if revision:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    try:
        snapshot_download(**kwargs)
    except Exception as exc:  # noqa: BLE001 — continue building index for other sources
        logger.warning("Failed to download %s: %s", repo_id, exc)
    return local_dir


def _allow_patterns_for_entry(entry: RegistryEntry) -> Optional[List[str]]:
    patterns: List[str] = []
    for inc in entry.include_paths:
        patterns.append(f"{inc.rstrip('/')}/**")
    for sub in entry.hf_subfolders:
        patterns.append(f"{sub.rstrip('/')}/**")
    # Prefer common image / parquet layouts without pulling everything when capped
    if entry.source_format in ("png", "jpg", "jpeg", "webp"):
        patterns.extend([f"*.{entry.source_format}", f"**/*.{entry.source_format}"])
    if not patterns:
        return None
    return patterns


def index_entry_images(
    entry: RegistryEntry,
    local_root: Path,
    max_per_source: int,
    rng: random.Random,
) -> List[IndexedSample]:
    files = _collect_local_images(local_root, max_files=max(max_per_source * 5, max_per_source))
    if not files:
        # Try parquet/datasets arrow extraction via datasets library when no loose images
        files = _extract_from_hf_dataset(entry, local_root, max_per_source)
    if max_per_source > 0 and len(files) > max_per_source:
        files = rng.sample(files, max_per_source)
    return [
        IndexedSample(
            path=str(p.resolve()),
            label=entry.label,
            dataset_name=entry.name,
            generator_family=entry.generator_family,
            media_type=entry.media_type,
            source="registry",
        )
        for p in files
    ]


def _extract_from_hf_dataset(
    entry: RegistryEntry, local_root: Path, max_per_source: int
) -> List[Path]:
    """Materialize a capped set of images from a HF dataset if only parquet/arrow is present."""
    out_dir = local_root / "_extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _collect_local_images(out_dir, max_files=max_per_source)
    if existing:
        return existing
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
    except ImportError:
        return []

    try:
        ds = load_dataset(entry.path, split="train", streaming=True)
    except Exception:
        try:
            ds = load_dataset(entry.path, split=None, streaming=True)
            # pick first split
            if isinstance(ds, dict):
                ds = next(iter(ds.values()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot stream dataset %s: %s", entry.path, exc)
            return []

    saved: List[Path] = []
    image_keys = ("image", "img", "jpg", "png", "file")
    for i, row in enumerate(ds):
        if max_per_source > 0 and i >= max_per_source:
            break
        img = None
        for key in image_keys:
            if key in row and row[key] is not None:
                img = row[key]
                break
        if img is None:
            continue
        try:
            if hasattr(img, "convert"):
                pil = img.convert("RGB")
            elif isinstance(img, (str, Path)) and Path(img).exists():
                pil = PILImage.open(img).convert("RGB")
            else:
                continue
            dest = out_dir / f"{entry.name}_{i:06d}.jpg"
            pil.save(dest, quality=95)
            saved.append(dest)
        except Exception:  # noqa: BLE001
            continue
    return saved


def build_public_index(
    cache_dir: Union[str, Path],
    index_path: Union[str, Path],
    registries: Optional[Sequence[str]] = None,
    max_per_source: int = 2000,
    seed: int = 42,
    download: bool = True,
    skip_gasstation_in_registry: bool = True,
    dataset_allowlist: Optional[Sequence[str]] = None,
    dataset_limit: Optional[int] = None,
) -> List[IndexedSample]:
    """Download (optional) and index public gasbench image registries."""
    cache = expand_cache_dir(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    yaml_cache = cache / "yaml"
    skip = ["gasstation/"] if skip_gasstation_in_registry else []
    entries = load_registry_entries(registries, cache_dir=yaml_cache, skip_paths=skip)
    if dataset_allowlist:
        allow = set(dataset_allowlist)
        entries = [e for e in entries if e.name in allow]
    if dataset_limit is not None:
        entries = entries[: dataset_limit]

    rng = random.Random(seed)
    samples: List[IndexedSample] = []
    for entry in entries:
        local_root = cache / "hf" / entry.path.replace("/", "__")
        if download:
            patterns = _allow_patterns_for_entry(entry)
            # Cap download volume for large repos by preferring image patterns
            if patterns is None and max_per_source > 0:
                patterns = [
                    "*.jpg",
                    "*.jpeg",
                    "*.png",
                    "*.webp",
                    "**/*.jpg",
                    "**/*.jpeg",
                    "**/*.png",
                    "**/*.webp",
                ]
            download_hf_repo(
                entry.path,
                local_root,
                revision=entry.hf_revision,
                allow_patterns=patterns,
            )
        entry_samples = index_entry_images(entry, local_root, max_per_source, rng)
        logger.info(
            "Indexed %s (%s): %d images label=%d",
            entry.name,
            entry.path,
            len(entry_samples),
            entry.label,
        )
        samples.extend(entry_samples)

    write_index(samples, index_path)
    logger.info("Wrote %d samples to %s", len(samples), index_path)
    return samples


def index_gas_station(
    cache_dir: Union[str, Path],
    repo_id: str = "gasstation/gs-images-v4",
    media_type: str = "synthetic",
    max_samples: int = 20000,
    weeks: Optional[Sequence[str]] = None,
    download: bool = True,
    seed: int = 42,
) -> List[IndexedSample]:
    """Index GAS-Station adversarial synthetics (Phase 2)."""
    from .gasbench_index import media_type_to_label

    cache = expand_cache_dir(cache_dir)
    local_root = cache / "hf" / repo_id.replace("/", "__")
    if download:
        patterns = None
        if weeks:
            patterns = []
            for w in weeks:
                patterns.extend([f"*{w}*", f"**/*{w}*"])
        download_hf_repo(repo_id, local_root, allow_patterns=patterns)

    rng = random.Random(seed)
    files = _collect_local_images(local_root, max_files=max(max_samples * 2, max_samples))
    if not files:
        # streaming fallback
        entry = RegistryEntry(
            name="gasstation-generated-images",
            path=repo_id,
            media_type=media_type,
            generator_family="gasstation",
        )
        files = _extract_from_hf_dataset(entry, local_root, max_samples)

    if max_samples > 0 and len(files) > max_samples:
        files = rng.sample(files, max_samples)

    label = media_type_to_label(media_type)
    return [
        IndexedSample(
            path=str(p.resolve()),
            label=label,
            dataset_name="gasstation-generated-images",
            generator_family="gasstation",
            media_type=media_type,
            source="gas_station",
        )
        for p in files
    ]


def merge_indices(
    *sample_lists: Iterable[IndexedSample],
    index_path: Union[str, Path],
) -> List[IndexedSample]:
    merged: List[IndexedSample] = []
    seen = set()
    for lst in sample_lists:
        for s in lst:
            key = (s.path, s.label)
            if key in seen:
                continue
            seen.add(key)
            merged.append(s)
    write_index(merged, index_path)
    return merged
