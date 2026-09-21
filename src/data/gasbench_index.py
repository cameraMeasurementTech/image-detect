"""Parse gasbench image registry YAMLs into dataset entries."""

from __future__ import annotations

import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import yaml

LABEL_MAP = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 2,
}

ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

DEFAULT_REGISTRY_URLS = [
    "https://raw.githubusercontent.com/BitMind-AI/gasbench/main/src/gasbench/dataset/configs/real_images.yaml",
    "https://raw.githubusercontent.com/BitMind-AI/gasbench/main/src/gasbench/dataset/configs/synthetic_images.yaml",
]


@dataclass
class RegistryEntry:
    name: str
    path: str
    media_type: str
    modality: str = "image"
    source_format: str = ""
    generator_family: str = "unknown"
    content_category: str = "unknown"
    media_per_archive: int = -1
    archives_per_dataset: int = -1
    include_paths: List[str] = field(default_factory=list)
    hf_revision: Optional[str] = None
    hf_subfolders: List[str] = field(default_factory=list)
    notes: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> int:
        return media_type_to_label(self.media_type)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label
        return d


def media_type_to_label(media_type: str) -> int:
    key = media_type.strip().lower()
    if key not in LABEL_MAP:
        raise KeyError(
            f"Invalid media_type '{media_type}'. Valid: {sorted(LABEL_MAP)}"
        )
    return LABEL_MAP[key]


def _fetch_text(url_or_path: str, cache_dir: Optional[Path] = None) -> str:
    path = Path(url_or_path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / Path(url_or_path).name
        if cached.exists():
            return cached.read_text(encoding="utf-8")
    with urllib.request.urlopen(url_or_path, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    if cache_dir is not None:
        cached.write_text(text, encoding="utf-8")
    return text


def load_yaml_datasets(
    url_or_path: str, cache_dir: Optional[Union[str, Path]] = None
) -> List[RegistryEntry]:
    cache = Path(cache_dir).expanduser() if cache_dir else None
    raw = yaml.safe_load(_fetch_text(url_or_path, cache))
    rows = raw.get("datasets", raw) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected YAML structure in {url_or_path}")

    entries: List[RegistryEntry] = []
    known = {
        "name",
        "path",
        "media_type",
        "modality",
        "source_format",
        "generator_family",
        "content_category",
        "media_per_archive",
        "archives_per_dataset",
        "include_paths",
        "hf_revision",
        "hf_subfolders",
        "notes",
    }
    for row in rows:
        if not isinstance(row, dict) or "name" not in row or "path" not in row:
            continue
        media_type = str(row.get("media_type", "")).lower()
        if media_type not in LABEL_MAP:
            continue
        extras = {k: v for k, v in row.items() if k not in known}
        entries.append(
            RegistryEntry(
                name=row["name"],
                path=row["path"],
                media_type=media_type,
                modality=str(row.get("modality", "image")),
                source_format=str(row.get("source_format", "") or ""),
                generator_family=str(row.get("generator_family", "unknown") or "unknown"),
                content_category=str(row.get("content_category", "unknown") or "unknown"),
                media_per_archive=int(row.get("media_per_archive", -1) or -1),
                archives_per_dataset=int(row.get("archives_per_dataset", -1) or -1),
                include_paths=list(row.get("include_paths") or []),
                hf_revision=row.get("hf_revision"),
                hf_subfolders=list(row.get("hf_subfolders") or []),
                notes=str(row.get("notes", "") or ""),
                extras=extras,
            )
        )
    return entries


def load_registry_entries(
    registries: Optional[Sequence[str]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    skip_paths: Optional[Iterable[str]] = None,
) -> List[RegistryEntry]:
    """Load and concatenate registry YAMLs. Optionally skip HF paths (e.g. gasstation/*)."""
    urls = list(registries or DEFAULT_REGISTRY_URLS)
    skip = set(skip_paths or [])
    out: List[RegistryEntry] = []
    seen_names = set()
    for url in urls:
        for entry in load_yaml_datasets(url, cache_dir=cache_dir):
            if entry.name in seen_names:
                continue
            if any(entry.path.startswith(prefix) for prefix in skip):
                continue
            seen_names.add(entry.name)
            out.append(entry)
    return out


def summarize_entries(entries: Sequence[RegistryEntry]) -> Dict[str, Any]:
    by_label: Dict[str, int] = {k: 0 for k in LABEL_MAP}
    families: Dict[str, int] = {}
    for e in entries:
        by_label[e.media_type] += 1
        families[e.generator_family] = families.get(e.generator_family, 0) + 1
    return {
        "n_datasets": len(entries),
        "by_media_type": by_label,
        "n_generator_families": len(families),
        "generator_families": dict(sorted(families.items(), key=lambda x: -x[1])),
    }
