"""Probe a base backbone under the SN34 gasbench I/O contract.

Compares:
  A) Raw HF / timm call patterns (often fail or mis-scale uint8)
  B) SN34 Uint8ImageClassifier wrapper (required contract)

Does not download datasets. Uses synthetic uint8 tensors only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.vit_detector import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_export_wrapper,
    load_hf_classifier,
)


@dataclass
class ProbeCase:
    name: str
    ok: bool
    required: bool
    detail: str
    shape: Optional[List[int]] = None


def _shape(t: Any) -> Optional[List[int]]:
    if torch.is_tensor(t):
        return list(t.shape)
    if hasattr(t, "logits") and torch.is_tensor(t.logits):
        return list(t.logits.shape)
    return None


def _as_logits(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if hasattr(out, "logits"):
        return out.logits
    raise TypeError(f"unsupported output type {type(out)}")


def probe_hf_backbone(
    backbone_id: str,
    *,
    num_classes: int = 3,
    image_size: int = 224,
    batch: int = 2,
    device: str = "cpu",
) -> Dict[str, Any]:
    cases: List[ProbeCase] = []
    h = w = int(image_size)
    x_u8 = torch.randint(0, 256, (batch, 3, h, w), dtype=torch.uint8, device=device)

    backbone = load_hf_classifier(backbone_id, num_classes=num_classes).to(device).eval()
    n_labels = int(getattr(getattr(backbone, "config", None), "num_labels", -1))
    cases.append(
        ProbeCase(
            name="head_num_labels",
            ok=n_labels == 3,
            required=True,
            detail=f"config.num_labels={n_labels} (SN34 image requires 3)",
            shape=[n_labels],
        )
    )

    # Typical HF training call — NOT what gasbench sends
    try:
        with torch.no_grad():
            out = backbone(pixel_values=x_u8.float() / 255.0)
        logits = _as_logits(out)
        cases.append(
            ProbeCase(
                name="raw_hf_float_0_1_pixel_values",
                ok=True,
                required=False,
                detail="Works for training notebooks; harness does NOT call this way",
                shape=list(logits.shape),
            )
        )
    except Exception as exc:
        cases.append(
            ProbeCase(
                name="raw_hf_float_0_1_pixel_values",
                ok=False,
                required=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    # What gasbench sends: positional uint8 NCHW
    raw_u8_logits = None
    try:
        with torch.no_grad():
            raw_u8_logits = _as_logits(backbone(x_u8))
        # Shape may look fine but scale is wrong (torch casts uint8→float without /255)
        cases.append(
            ProbeCase(
                name="raw_hf_uint8_positional",
                ok=False,  # treat as fail for SN34 even if it runs
                required=True,
                detail=(
                    f"ran with shape={tuple(raw_u8_logits.shape)} but without /255+mean/std — "
                    "logits are mis-scaled vs gasbench contract (do NOT submit raw HF)"
                ),
                shape=list(raw_u8_logits.shape),
            )
        )
    except Exception as exc:
        cases.append(
            ProbeCase(
                name="raw_hf_uint8_positional",
                ok=False,
                required=True,
                detail=f"FAIL under harness input: {type(exc).__name__}: {exc}",
            )
        )

    # Wrapped SN34 adapter
    wrap = build_export_wrapper(backbone).to(device).eval()
    try:
        with torch.no_grad():
            w_logits = wrap(x_u8)
        shape_ok = tuple(w_logits.shape) == (batch, 3)
        finite = bool(torch.isfinite(w_logits).all().item())
        bare = torch.is_tensor(w_logits)
        delta = None
        if raw_u8_logits is not None and raw_u8_logits.shape == w_logits.shape:
            delta = float((w_logits.float() - raw_u8_logits.float()).abs().max().item())
        cases.append(
            ProbeCase(
                name="wrapped_uint8_sn34_contract",
                ok=shape_ok and finite and bare,
                required=True,
                detail=(
                    f"tensor logits shape={tuple(w_logits.shape)} finite={finite}; "
                    f"max|wrap-raw_u8|={delta} (nonzero ⇒ raw path was wrong)"
                ),
                shape=list(w_logits.shape),
            )
        )
        # Correct manual preprocess must match wrapper
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        with torch.no_grad():
            manual = _as_logits(
                backbone(pixel_values=(x_u8.float() / 255.0 - mean) / std)
            )
        match = float((manual - w_logits).abs().max().item())
        cases.append(
            ProbeCase(
                name="wrapper_matches_manual_preproc",
                ok=match < 1e-5,
                required=True,
                detail=f"max|wrap - (/255-mean)/std|={match}",
                shape=list(manual.shape),
            )
        )
    except Exception as exc:
        cases.append(
            ProbeCase(
                name="wrapped_uint8_sn34_contract",
                ok=False,
                required=True,
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    required_ok = all(c.ok for c in cases if c.required and c.name != "raw_hf_uint8_positional")
    # raw_hf_uint8_positional is expected to be "not ok" — we still require the WRAP path
    wrap_ok = next(c.ok for c in cases if c.name == "wrapped_uint8_sn34_contract")
    head_ok = next(c.ok for c in cases if c.name == "head_num_labels")

    return {
        "backbone_id": backbone_id,
        "num_classes_requested": num_classes,
        "image_size": image_size,
        "meets_sn34_with_wrapper": bool(wrap_ok and head_ok),
        "raw_backbone_alone_ok": False,
        "advice": _advice(head_ok, wrap_ok, n_labels),
        "cases": [asdict(c) for c in cases],
    }


def _advice(head_ok: bool, wrap_ok: bool, n_labels: int) -> List[str]:
    tips = []
    if not head_ok:
        tips.append(
            f"Replace the classification head with num_labels/num_classes=3 "
            f"(got {n_labels}). Class order: [real, synthetic, semisynthetic]."
        )
        tips.append(
            "HF: AutoModelForImageClassification.from_pretrained(..., num_labels=3, "
            "ignore_mismatched_sizes=True). timm: create_model(..., num_classes=3) "
            "or num_classes=0 + nn.Linear(feat, 3)."
        )
    tips.append(
        "Do not submit the raw HF/timm module. Wrap with Uint8ImageClassifier "
        "(src/models/vit_detector.py) or bake the same /255 + mean/std into model.py forward()."
    )
    if wrap_ok:
        tips.append(
            "With the SN34 wrapper, I/O is compliant. Train the 3-class head on gasbench "
            "labels before expecting Gate 5 (≥80% exam)."
        )
    else:
        tips.append(
            "Wrapper forward failed — see docs/Architecture-IO-Adapter.md for custom backbones."
        )
    return tips


def main() -> int:
    p = argparse.ArgumentParser(description="Probe base model vs SN34 uint8 I/O contract")
    p.add_argument("--config", default=str(ROOT / "configs/train_vit_phase1.yaml"))
    p.add_argument("--backbone", default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    model_cfg = (cfg or {}).get("model") or {}
    backbone = args.backbone or model_cfg.get("backbone") or "google/vit-base-patch16-224-in21k"
    num_classes = args.num_classes or int(model_cfg.get("num_classes", 3))
    image_size = args.image_size or int(model_cfg.get("image_size", 224))

    report = probe_hf_backbone(
        backbone,
        num_classes=num_classes,
        image_size=image_size,
        batch=args.batch,
        device=args.device,
    )

    print(f"backbone: {report['backbone_id']}")
    print(f"meets_sn34_with_wrapper: {report['meets_sn34_with_wrapper']}")
    print(f"raw_backbone_alone_ok:   {report['raw_backbone_alone_ok']}")
    print()
    for c in report["cases"]:
        mark = "PASS" if c["ok"] else "FAIL"
        req = "required" if c["required"] else "info"
        print(f"  [{mark}] ({req}) {c['name']}: {c['detail']}")
    print()
    print("How to fix / adapt:")
    for tip in report["advice"]:
        print(f"  • {tip}")

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON: {out}")

    return 0 if report["meets_sn34_with_wrapper"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
