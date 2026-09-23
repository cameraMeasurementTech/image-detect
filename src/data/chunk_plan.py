"""Plan download/train chunks that fit under a disk budget (e.g. 2.5 TiB on a 3 TB disk).

Packs unique HuggingFace dataset repos into ordered chunks. Repos larger than the
budget (or remaining room) are marked ``stream_cap``: only ``max_per_source`` images
are materialized via streaming, so they never consume multi-TiB snapshots.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .gasbench_index import RegistryEntry, load_registry_entries

logger = logging.getLogger(__name__)

# Defaults: leave ~0.5 TiB headroom on a 3 TB volume for checkpoints / OS / zip.
DEFAULT_BUDGET_BYTES = int(2.5 * (1024**4))  # 2.5 TiB
DEFAULT_AVG_STREAM_IMAGE_BYTES = 250 * 1024  # estimate for stream_cap disk cost


@dataclass
class RepoSpec:
    repo_id: str
    size_bytes: int
    entry_names: List[str]
    media_types: List[str]
    mode: str = "full"  # full | stream_cap

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024**3)


@dataclass
class ChunkPlan:
    chunk_id: int
    budget_bytes: int
    repos: List[RepoSpec] = field(default_factory=list)
    estimated_bytes: int = 0

    @property
    def entry_names(self) -> List[str]:
        names: List[str] = []
        for r in self.repos:
            names.extend(r.entry_names)
        return names

    @property
    def repo_ids(self) -> List[str]:
        return [r.repo_id for r in self.repos]

    @property
    def full_repo_ids(self) -> List[str]:
        return [r.repo_id for r in self.repos if r.mode == "full"]

    @property
    def stream_repo_ids(self) -> List[str]:
        return [r.repo_id for r in self.repos if r.mode == "stream_cap"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "budget_bytes": self.budget_bytes,
            "budget_tib": self.budget_bytes / (1024**4),
            "estimated_bytes": self.estimated_bytes,
            "estimated_tib": self.estimated_bytes / (1024**4),
            "n_repos": len(self.repos),
            "n_entries": len(self.entry_names),
            "repos": [asdict(r) for r in self.repos],
            "entry_names": self.entry_names,
            "full_repo_ids": self.full_repo_ids,
            "stream_repo_ids": self.stream_repo_ids,
        }


def fetch_repo_sizes(
    entries: Sequence[RegistryEntry],
    *,
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> Dict[str, int]:
    """Return ``{repo_id: used_storage_bytes}`` for unique registry paths."""
    if cache_path and cache_path.exists() and not refresh:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        sizes = {k: int(v) for k, v in raw.get("sizes", raw).items()}
        logger.info("Loaded %d repo sizes from %s", len(sizes), cache_path)
        return sizes

    from huggingface_hub import HfApi

    api = HfApi()
    repos = sorted({e.path for e in entries})
    sizes: Dict[str, int] = {}
    logger.info("Querying HuggingFace sizes for %d repos...", len(repos))
    for i, repo in enumerate(repos):
        try:
            info = api.dataset_info(repo, files_metadata=False)
            size = getattr(info, "used_storage", None) or 0
            sizes[repo] = int(size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Size lookup failed for %s: %s — treating as stream_cap", repo, exc)
            sizes[repo] = 0
        if (i + 1) % 25 == 0:
            logger.info("  sized %d/%d", i + 1, len(repos))
        time.sleep(0.03)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"sizes": sizes, "n_repos": len(sizes), "updated_unix": time.time()},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote size catalog %s", cache_path)
    return sizes


def build_repo_specs(
    entries: Sequence[RegistryEntry],
    sizes: Dict[str, int],
) -> List[RepoSpec]:
    by_repo: Dict[str, List[RegistryEntry]] = {}
    for e in entries:
        by_repo.setdefault(e.path, []).append(e)
    specs: List[RepoSpec] = []
    for repo_id, group in by_repo.items():
        specs.append(
            RepoSpec(
                repo_id=repo_id,
                size_bytes=int(sizes.get(repo_id, 0)),
                entry_names=[e.name for e in group],
                media_types=sorted({e.media_type for e in group}),
            )
        )
    return specs


def _stream_cost_bytes(spec: RepoSpec, max_per_source: int, avg_image_bytes: int) -> int:
    n = max(1, len(spec.entry_names)) * max(1, max_per_source)
    return n * avg_image_bytes


def plan_chunks(
    specs: Sequence[RepoSpec],
    *,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    max_per_source: int = 2000,
    avg_stream_image_bytes: int = DEFAULT_AVG_STREAM_IMAGE_BYTES,
) -> List[ChunkPlan]:
    """First-fit decreasing pack. Oversized repos → stream_cap (small disk)."""
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")

    full: List[RepoSpec] = []
    stream: List[RepoSpec] = []
    for s in specs:
        if s.size_bytes <= 0 or s.size_bytes > budget_bytes:
            stream.append(
                RepoSpec(
                    repo_id=s.repo_id,
                    size_bytes=_stream_cost_bytes(s, max_per_source, avg_stream_image_bytes),
                    entry_names=list(s.entry_names),
                    media_types=list(s.media_types),
                    mode="stream_cap",
                )
            )
        else:
            full.append(
                RepoSpec(
                    repo_id=s.repo_id,
                    size_bytes=s.size_bytes,
                    entry_names=list(s.entry_names),
                    media_types=list(s.media_types),
                    mode="full",
                )
            )

    # Pack full repos largest-first into bins
    full_sorted = sorted(full, key=lambda r: -r.size_bytes)
    chunks: List[ChunkPlan] = []

    def new_chunk() -> ChunkPlan:
        c = ChunkPlan(chunk_id=len(chunks), budget_bytes=budget_bytes)
        chunks.append(c)
        return c

    for spec in full_sorted:
        placed = False
        for c in chunks:
            if c.estimated_bytes + spec.size_bytes <= budget_bytes:
                c.repos.append(spec)
                c.estimated_bytes += spec.size_bytes
                placed = True
                break
        if not placed:
            c = new_chunk()
            c.repos.append(spec)
            c.estimated_bytes = spec.size_bytes

    # Attach stream_cap repos into chunks with remaining room; else new chunks
    stream_sorted = sorted(stream, key=lambda r: -r.size_bytes)
    for spec in stream_sorted:
        placed = False
        for c in chunks:
            if c.estimated_bytes + spec.size_bytes <= budget_bytes:
                c.repos.append(spec)
                c.estimated_bytes += spec.size_bytes
                placed = True
                break
        if not placed:
            c = new_chunk()
            c.repos.append(spec)
            c.estimated_bytes = spec.size_bytes

    if not chunks:
        chunks.append(ChunkPlan(chunk_id=0, budget_bytes=budget_bytes))

    # Stable order inside each chunk: full first (by size desc), then stream
    for c in chunks:
        c.repos.sort(key=lambda r: (0 if r.mode == "full" else 1, -r.size_bytes))

    return chunks


def build_chunk_plan_from_config(
    cfg: Dict[str, Any],
    *,
    catalog_path: Optional[Path] = None,
    refresh_sizes: bool = False,
) -> Tuple[List[ChunkPlan], Dict[str, int]]:
    chunk_cfg = cfg.get("chunk") or {}
    budget = chunk_cfg.get("budget_bytes")
    if budget is None:
        budget_tib = float(chunk_cfg.get("budget_tib", 2.5))
        budget = int(budget_tib * (1024**4))
    else:
        budget = int(budget)

    cache = Path(cfg["cache_dir"]).expanduser()
    yaml_cache = cache / "yaml"
    catalog = catalog_path or Path(
        chunk_cfg.get("catalog_path") or (cache / "repo_sizes.json")
    )
    entries = load_registry_entries(
        cfg.get("registries"),
        cache_dir=str(yaml_cache),
        skip_paths=["gasstation/"] if chunk_cfg.get("skip_gasstation", True) else [],
    )
    allow = cfg.get("data", {}).get("dataset_allowlist")
    if allow:
        allow_set = set(allow)
        entries = [e for e in entries if e.name in allow_set]
    limit = cfg.get("data", {}).get("dataset_limit")
    if limit is not None:
        entries = entries[: int(limit)]

    sizes = fetch_repo_sizes(entries, cache_path=catalog, refresh=refresh_sizes)
    specs = build_repo_specs(entries, sizes)
    max_per = int(cfg.get("data", {}).get("max_per_source", 2000))
    avg_img = int(chunk_cfg.get("avg_stream_image_bytes", DEFAULT_AVG_STREAM_IMAGE_BYTES))
    chunks = plan_chunks(
        specs,
        budget_bytes=budget,
        max_per_source=max_per,
        avg_stream_image_bytes=avg_img,
    )
    return chunks, sizes


def summarize_plan(chunks: Sequence[ChunkPlan]) -> Dict[str, Any]:
    return {
        "n_chunks": len(chunks),
        "budget_tib": chunks[0].budget_bytes / (1024**4) if chunks else None,
        "total_estimated_tib_if_sequential": sum(c.estimated_bytes for c in chunks)
        / (1024**4),
        "peak_estimated_tib": max((c.estimated_bytes for c in chunks), default=0)
        / (1024**4),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "estimated_tib": round(c.estimated_bytes / (1024**4), 3),
                "n_repos": len(c.repos),
                "n_full": len(c.full_repo_ids),
                "n_stream_cap": len(c.stream_repo_ids),
                "n_entries": len(c.entry_names),
                "largest_repo": (
                    max(c.repos, key=lambda r: r.size_bytes).repo_id if c.repos else None
                ),
            }
            for c in chunks
        ],
    }


def save_plan(chunks: Sequence[ChunkPlan], path: Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize_plan(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_plan(path: Path) -> List[ChunkPlan]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    chunks: List[ChunkPlan] = []
    for c in raw["chunks"]:
        repos = [RepoSpec(**r) for r in c["repos"]]
        chunks.append(
            ChunkPlan(
                chunk_id=int(c["chunk_id"]),
                budget_bytes=int(c["budget_bytes"]),
                repos=repos,
                estimated_bytes=int(c["estimated_bytes"]),
            )
        )
    return chunks
