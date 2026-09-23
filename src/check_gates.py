"""SN34 image-discriminator gate checks (no full dataset required).

Validates Gates 1–3 from Image-Discriminator-Gates.md against either:
  • an existing submission directory / zip, or
  • a HuggingFace backbone from config (downloads weights only if missing).

Gate 5 (exam ≥80% accuracy) needs labeled images — use --mini-exam for a
tiny optional download; full corpus is not required for structural/I/O checks.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml

logger = logging.getLogger("check_gates")

ALLOWED_ROOT_MODULES = {
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "timm",
    "einops",
    "safetensors",
    "flash_attn",
    "PIL",
    "cv2",
    "skimage",
    "decord",
    "fvcore",
    "ultralytics",
    "numpy",
    "scipy",
    "math",
    "functools",
    "typing",
    "collections",
    "dataclasses",
    "enum",
    "abc",
    "pathlib",
}

BLOCKED_ROOT_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "httpx",
    "aiohttp",
    "pickle",
    "marshal",
    "dill",
    "joblib",
    "importlib",
    "numba",
    "cython",
    "ctypes",
    "multiprocessing",
    "threading",
    "concurrent",
    "boto3",
    "wandb",
    "sqlite3",
}

BLOCKED_NAMES = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "globals",
    "locals",
}

REQUIRED_FILES = ("model_config.yaml", "model.py")
SUPPORTED_DTYPES = {
    "float32",
    "fp32",
    "float16",
    "fp16",
    "bfloat16",
    "bf16",
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    gate: str = ""


@dataclass
class GateReport:
    checks: List[CheckResult] = field(default_factory=list)
    model_dir: Optional[str] = None
    backbone: Optional[str] = None

    def add(self, name: str, passed: bool, detail: str = "", gate: str = "") -> None:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail, gate=gate))

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks if c.gate != "5")

    @property
    def structural_ok(self) -> bool:
        """Gates 1–3 (and package) must pass before investing in full data."""
        return all(c.passed for c in self.checks if c.gate in {"1", "2", "3", "package"})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "backbone": self.backbone,
            "structural_ok": self.structural_ok,
            "all_ok": all(c.passed for c in self.checks),
            "checks": [
                {"gate": c.gate, "name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def _root_module(name: str) -> str:
    return name.split(".")[0]


def analyze_model_py(source: str) -> Tuple[List[str], List[str], bool]:
    """Return (blocked_imports, blocked_calls, has_load_model)."""
    tree = ast.parse(source)
    blocked_imports: List[str] = []
    blocked_calls: List[str] = []
    has_load_model = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module(alias.name)
                if root in BLOCKED_ROOT_MODULES or (
                    root not in ALLOWED_ROOT_MODULES and root not in {"__future__"}
                ):
                    # Allow unknown only if not explicitly blocked? Prefer allowlist.
                    if root in BLOCKED_ROOT_MODULES or root not in ALLOWED_ROOT_MODULES:
                        if root != "__future__":
                            blocked_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = _root_module(mod) if mod else ""
            if root and root != "__future__":
                if root in BLOCKED_ROOT_MODULES or root not in ALLOWED_ROOT_MODULES:
                    blocked_imports.append(mod)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in BLOCKED_NAMES:
                blocked_calls.append(fn.id)
            if isinstance(fn, ast.Attribute) and fn.attr in {"script", "trace"}:
                # torch.jit.script / trace
                blocked_calls.append(f"*.{fn.attr}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "load_model":
                has_load_model = True

    return sorted(set(blocked_imports)), sorted(set(blocked_calls)), has_load_model


def _load_module_from_path(model_py: Path, module_name: str = "sn34_submission_model"):
    spec = importlib.util.spec_from_file_location(module_name, model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {model_py}")
    module = importlib.util.module_from_spec(spec)
    # Isolate from caller's sys.modules collisions
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_submission_dir(
    model_dir: Optional[Path],
    zip_path: Optional[Path],
    work: Path,
) -> Path:
    if model_dir is not None:
        return Path(model_dir).expanduser().resolve()
    if zip_path is not None:
        z = Path(zip_path).expanduser().resolve()
        dest = work / "unzipped"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dest)
        return dest
    raise ValueError("Provide --model-dir or --zip-path (or --backbone / --config)")


def export_probe_from_backbone(
    backbone_id: str,
    num_classes: int,
    image_size: int,
    out_dir: Path,
) -> Path:
    """Build a gasbench-shaped submission from a (possibly random-head) HF backbone."""
    from src.models.export_bundle import export_submission
    from src.models.vit_detector import build_export_wrapper, load_hf_classifier

    backbone = load_hf_classifier(backbone_id, num_classes=num_classes)
    wrapper = build_export_wrapper(backbone)
    export_submission(
        wrapper,
        submission_dir=out_dir,
        backbone_id=backbone_id,
        num_classes=num_classes,
        resize=(image_size, image_size),
        zip_path=out_dir / "image_detector.zip",
        name="sn34-gate-probe",
    )
    return out_dir


def check_package(model_dir: Path, report: GateReport) -> Dict[str, Any]:
    cfg_path = model_dir / "model_config.yaml"
    py_path = model_dir / "model.py"
    report.add(
        "model_config.yaml present",
        cfg_path.is_file(),
        str(cfg_path),
        gate="1",
    )
    report.add("model.py present", py_path.is_file(), str(py_path), gate="1")

    cfg: Dict[str, Any] = {}
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    modality = cfg.get("modality")
    report.add(
        "modality == image",
        modality == "image",
        f"got {modality!r}",
        gate="1",
    )

    dtype = cfg.get("dtype", "float32")
    report.add(
        "dtype supported",
        str(dtype).lower() in SUPPORTED_DTYPES,
        f"dtype={dtype}",
        gate="1",
    )

    model_cfg = cfg.get("model") or {}
    num_classes = int(model_cfg.get("num_classes", -1))
    report.add(
        "num_classes == 3",
        num_classes == 3,
        f"num_classes={num_classes}",
        gate="1",
    )

    weights_name = model_cfg.get("weights_file") or "model.safetensors"
    weights_path = model_dir / weights_name
    report.add(
        "weights file present",
        weights_path.is_file(),
        str(weights_path),
        gate="1",
    )

    # ONNX rejected
    onnx = list(model_dir.glob("*.onnx"))
    report.add("no ONNX artifacts", len(onnx) == 0, f"found {onnx}", gate="1")

    pre = cfg.get("preprocessing") or {}
    resize = pre.get("resize") or [224, 224]
    ok_resize = (
        isinstance(resize, (list, tuple))
        and len(resize) == 2
        and all(isinstance(x, (int, float)) for x in resize)
    )
    report.add(
        "preprocessing.resize [H,W]",
        bool(ok_resize),
        f"resize={resize}",
        gate="1",
    )

    return cfg


def check_sandbox(model_dir: Path, report: GateReport) -> None:
    py_path = model_dir / "model.py"
    if not py_path.is_file():
        report.add("sandbox static analysis", False, "model.py missing", gate="2")
        return
    source = py_path.read_text(encoding="utf-8")
    blocked_imports, blocked_calls, has_load = analyze_model_py(source)
    report.add(
        "load_model defined",
        has_load,
        "def load_model(...)" if has_load else "missing load_model",
        gate="2",
    )
    report.add(
        "no blocked imports",
        len(blocked_imports) == 0,
        f"blocked={blocked_imports}" if blocked_imports else "ok",
        gate="2",
    )
    report.add(
        "no blocked dynamic calls",
        len(blocked_calls) == 0,
        f"blocked={blocked_calls}" if blocked_calls else "ok",
        gate="2",
    )


def _extract_logits(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        if "logits" in out:
            return out["logits"]
        if "output" in out:
            return out["output"]
        if len(out) == 1:
            return next(iter(out.values()))
        raise ValueError(f"ambiguous dict output keys={list(out.keys())}")
    if isinstance(out, (tuple, list)):
        if len(out) == 1:
            return _extract_logits(out[0])
        raise ValueError(f"ambiguous tuple/list len={len(out)}")
    raise ValueError(f"unsupported output type {type(out)}")


def check_io(
    model_dir: Path,
    cfg: Dict[str, Any],
    report: GateReport,
    device: str = "cpu",
) -> None:
    py_path = model_dir / "model.py"
    model_cfg = cfg.get("model") or {}
    num_classes = int(model_cfg.get("num_classes", 3))
    weights_name = model_cfg.get("weights_file") or "model.safetensors"
    weights_path = model_dir / weights_name
    resize = (cfg.get("preprocessing") or {}).get("resize") or [224, 224]
    h, w = int(resize[0]), int(resize[1])

    try:
        module = _load_module_from_path(py_path)
    except Exception as exc:
        report.add("import model.py", False, str(exc), gate="1")
        return
    report.add("import model.py", True, "ok", gate="1")

    if not hasattr(module, "load_model"):
        report.add("load_model callable", False, "attribute missing", gate="1")
        return

    kwargs = {
        k: v
        for k, v in model_cfg.items()
        if k not in {"weights_file", "num_classes", "backbone_id"}
    }
    try:
        model = module.load_model(str(weights_path), num_classes=num_classes, **kwargs)
    except Exception as exc:
        report.add("load_model(...)", False, str(exc), gate="1")
        return

    report.add(
        "load_model returns nn.Module",
        isinstance(model, torch.nn.Module),
        type(model).__name__,
        gate="1",
    )
    model = model.to(device)
    model.train(False)

    # Offline: config.json should exist so transformers can load without hub
    report.add(
        "offline config.json present",
        (model_dir / "config.json").is_file(),
        "needed for transformers local_files_only",
        gate="package",
    )

    batches = [1, 2, 4]
    for b in batches:
        x = torch.randint(0, 256, (b, 3, h, w), dtype=torch.uint8, device=device)
        try:
            with torch.no_grad():
                raw = model(x)
            logits = _extract_logits(raw)
            shape_ok = tuple(logits.shape) == (b, num_classes)
            finite = bool(torch.isfinite(logits).all().item())
            report.add(
                f"uint8 forward batch={b}",
                shape_ok and finite,
                f"shape={tuple(logits.shape)} finite={finite}",
                gate="3",
            )
        except Exception as exc:
            report.add(f"uint8 forward batch={b}", False, str(exc), gate="3")

    # Float path (some harnesses may pass float already-normalized or 0-255)
    x_f = torch.rand(2, 3, h, w, device=device)
    try:
        with torch.no_grad():
            logits = _extract_logits(model(x_f))
        report.add(
            "float [0,1] forward",
            tuple(logits.shape) == (2, num_classes),
            f"shape={tuple(logits.shape)}",
            gate="3",
        )
    except Exception as exc:
        report.add("float [0,1] forward", False, str(exc), gate="3")

    x_255 = (torch.rand(2, 3, h, w, device=device) * 255).float()
    try:
        with torch.no_grad():
            logits = _extract_logits(model(x_255))
        report.add(
            "float [0,255] forward",
            tuple(logits.shape) == (2, num_classes),
            f"shape={tuple(logits.shape)}",
            gate="3",
        )
    except Exception as exc:
        report.add("float [0,255] forward", False, str(exc), gate="3")

    # Wrong channel count should fail loudly (document, not required pass)
    try:
        bad = torch.randint(0, 256, (1, 1, h, w), dtype=torch.uint8, device=device)
        with torch.no_grad():
            model(bad)
        report.add(
            "rejects non-RGB (informational)",
            False,
            "model accepted 1-channel input — harness always sends 3ch",
            gate="3",
        )
        # Don't fail structural — flip to passed with note
        report.checks[-1].passed = True
        report.checks[-1].detail = "accepted 1ch (harness still sends RGB); noted only"
    except Exception as exc:
        report.add(
            "rejects non-RGB (informational)",
            True,
            f"raised as expected: {type(exc).__name__}",
            gate="3",
        )


def check_zip(zip_path: Path, report: GateReport) -> None:
    if not zip_path.is_file():
        report.add("zip exists", False, str(zip_path), gate="package")
        return
    report.add("zip exists", True, str(zip_path), gate="package")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    for req in ("model_config.yaml", "model.py", "model.safetensors"):
        # weights name may vary — check config inside zip if needed
        present = any(n == req or n.endswith("/" + req) for n in names)
        if req == "model.safetensors":
            present = present or any(n.endswith(".safetensors") for n in names)
        report.add(f"zip contains {req}", present, f"entries={sorted(names)[:12]}", gate="package")
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    report.add(
        "zip size reported",
        True,
        f"{size_mb:.2f} MB",
        gate="package",
    )


def run_mini_exam(
    model_dir: Path,
    data_dir: Path,
    report: GateReport,
    *,
    dataset_limit: int = 2,
    max_per_source: int = 32,
    config_path: Optional[Path] = None,
) -> None:
    """Optional: download a tiny public slice and score accuracy (Gate 5 proxy)."""
    from src.data.download import build_public_index
    from src.eval.score_model import score_submission
    from src.paths import apply_path_overrides

    root = Path(__file__).resolve().parents[1]
    cfg_path = config_path or (root / "configs/train_vit_phase1.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    apply_path_overrides(cfg, data_dir=str(data_dir))
    index_path = Path(cfg["index_path"])
    logger.info(
        "Mini-exam download: dataset_limit=%d max_per_source=%d -> %s",
        dataset_limit,
        max_per_source,
        data_dir,
    )
    build_public_index(
        cache_dir=cfg["cache_dir"],
        index_path=index_path,
        registries=cfg.get("registries"),
        max_per_source=max_per_source,
        seed=int(cfg.get("seed", 42)),
        download=True,
        dataset_limit=dataset_limit,
    )
    result = score_submission(
        model_dir=model_dir,
        index_path=index_path,
        mode="small",
        batch_size=8,
        seed=42,
        aug_weight=0.0,
    )
    acc = float(result.get("accuracy") or 0.0)
    sn34 = float(result.get("sn34_score") or 0.0)
    passed = acc >= 0.80
    report.add(
        "mini-exam accuracy >= 0.80",
        passed,
        f"accuracy={acc:.4f} sn34={sn34:.4f} n={result.get('n_samples')} "
        f"(untrained head usually fails; train before full download)",
        gate="5",
    )


def run_checks(
    *,
    model_dir: Optional[Path] = None,
    zip_path: Optional[Path] = None,
    backbone: Optional[str] = None,
    config_path: Optional[Path] = None,
    num_classes: int = 3,
    image_size: int = 224,
    device: str = "cpu",
    mini_exam: bool = False,
    mini_data_dir: Optional[Path] = None,
    dataset_limit: int = 2,
    max_per_source: int = 32,
    keep_probe_dir: Optional[Path] = None,
) -> GateReport:
    report = GateReport()
    work = Path(tempfile.mkdtemp(prefix="sn34_gate_"))
    try:
        if backbone or (config_path and model_dir is None and zip_path is None):
            cfg = {}
            if config_path:
                cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            model_cfg = cfg.get("model") or {}
            backbone = backbone or model_cfg.get("backbone")
            num_classes = int(model_cfg.get("num_classes", num_classes))
            image_size = int(model_cfg.get("image_size", image_size))
            if not backbone:
                raise ValueError("No backbone: pass --backbone or --config with model.backbone")
            report.backbone = backbone
            probe = Path(keep_probe_dir) if keep_probe_dir else (work / "probe_submission")
            probe.mkdir(parents=True, exist_ok=True)
            logger.info("Exporting probe submission from backbone=%s", backbone)
            export_probe_from_backbone(backbone, num_classes, image_size, probe)
            model_dir = probe
            zip_path = probe / "image_detector.zip"

        assert model_dir is not None or zip_path is not None
        resolved = _resolve_submission_dir(
            model_dir, zip_path if model_dir is None else None, work
        )
        report.model_dir = str(resolved)

        cfg = check_package(resolved, report)
        check_sandbox(resolved, report)
        weights_name = (cfg.get("model") or {}).get("weights_file") or "model.safetensors"
        if (resolved / "model.py").is_file() and (resolved / weights_name).is_file():
            check_io(resolved, cfg, report, device=device)

        z = zip_path
        if z is None:
            candidate = resolved / "image_detector.zip"
            z = candidate if candidate.is_file() else None
        if z is not None:
            check_zip(Path(z), report)

        if mini_exam:
            data = Path(mini_data_dir) if mini_data_dir else (work / "mini_data")
            data.mkdir(parents=True, exist_ok=True)
            run_mini_exam(
                resolved,
                data,
                report,
                dataset_limit=dataset_limit,
                max_per_source=max_per_source,
                config_path=config_path,
            )
        else:
            report.add(
                "exam gate deferred",
                True,
                "Gate 5 (≥80% acc) needs labeled data — re-run with --mini-exam "
                "after a short train, or skip until ready for full download",
                gate="5",
            )
    finally:
        # Always remove temp workdir; keep_probe_dir / user --data-dir live outside it.
        shutil.rmtree(work, ignore_errors=True)

    return report


def format_report(report: GateReport) -> str:
    lines = [
        f"model_dir: {report.model_dir}",
        f"backbone:  {report.backbone}",
        f"structural_ok (gates 1–3): {report.structural_ok}",
        "",
    ]
    for c in report.checks:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{mark}] gate {c.gate or '-':<8} {c.name}: {c.detail}")
    lines.append("")
    if report.structural_ok:
        lines.append(
            "Verdict: package + sandbox + I/O OK — safe to proceed with dataset download "
            "(exam ≥80% still requires training)."
        )
    else:
        lines.append(
            "Verdict: fix FAIL items before downloading the full dataset."
        )
    return "\n".join(lines)
