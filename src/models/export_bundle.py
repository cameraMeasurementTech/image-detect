"""Export a gasbench-compatible safetensors submission bundle (offline-loadable)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import yaml
from safetensors.torch import save_file

MODEL_PY = '''\
"""SN34 image discriminator — uint8 NCHW in, 3-class logits out.

Loads fully from the submission directory (no network).
"""

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForImageClassification


class Uint8Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, mean, std, temperature: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            if float(x.max()) > 1.5:
                x = x / 255.0
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x)
        logits = out.logits if hasattr(out, "logits") else out
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits


def load_model(weights_path: str, num_classes: int = 3, **kwargs) -> nn.Module:
    model_dir = Path(weights_path).resolve().parent
    temperature = float(kwargs.get("temperature", 1.0))
    mean = kwargs.get("mean", [0.485, 0.456, 0.406])
    std = kwargs.get("std", [0.229, 0.224, 0.225])

    # Architecture from local config.json only — weights come from safetensors below
    # (avoids transformers trying to load wrapper-prefixed keys as a HF checkpoint).
    cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    cfg.num_labels = int(num_classes)
    backbone = AutoModelForImageClassification.from_config(cfg)

    state = load_file(weights_path)
    cleaned = {}
    for k, v in state.items():
        if k in ("mean", "std"):
            continue
        nk = k[len("backbone."):] if k.startswith("backbone.") else k
        cleaned[nk] = v
    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    # classifier is randomly init if missing; mean/std handled on wrapper
    _ = missing, unexpected

    model = Uint8Classifier(backbone, mean=mean, std=std, temperature=temperature)
    if "mean" in state:
        with torch.no_grad():
            model.mean.copy_(state["mean"])
    if "std" in state:
        with torch.no_grad():
            model.std.copy_(state["std"])
    model.train(False)
    return model
'''


def _unwrap_backbone(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "backbone") and hasattr(model.backbone, "config"):
        return model.backbone
    return model


def export_submission(
    model: torch.nn.Module,
    submission_dir: Union[str, Path],
    backbone_id: str,
    num_classes: int = 3,
    resize: Sequence[int] = (224, 224),
    temperature: float = 1.0,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    zip_path: Optional[Union[str, Path]] = None,
    name: str = "sn34-image-detector",
) -> Path:
    """Write model_config.yaml, model.py, config.json, model.safetensors (+ zip)."""
    out = Path(submission_dir)
    out.mkdir(parents=True, exist_ok=True)

    backbone = _unwrap_backbone(model)
    # Persist HF config.json only (no weight file) for offline from_config()
    if hasattr(backbone, "config"):
        cfg = backbone.config
        if hasattr(cfg, "num_labels"):
            cfg.num_labels = int(num_classes)
        (out / "config.json").write_text(
            cfg.to_json_string() if hasattr(cfg, "to_json_string") else json.dumps(cfg.to_dict(), indent=2),
            encoding="utf-8",
        )

    # State dict under backbone.* plus mean/std buffers when wrapper used
    if hasattr(model, "backbone") and hasattr(model, "mean"):
        sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    else:
        sd = {
            f"backbone.{k}": v.detach().cpu().contiguous()
            for k, v in model.state_dict().items()
        }
        sd["mean"] = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        sd["std"] = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    save_file(sd, str(out / "model.safetensors"))

    config = {
        "name": name,
        "version": "1.0.0",
        "modality": "image",
        "dtype": "float32",
        "preprocessing": {
            "resize": list(resize),
            "normalize": {"mean": list(mean), "std": list(std)},
        },
        "model": {
            "num_classes": int(num_classes),
            "weights_file": "model.safetensors",
            "backbone_id": backbone_id,
            "temperature": float(temperature),
            "mean": list(mean),
            "std": list(std),
        },
    }
    (out / "model_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (out / "model.py").write_text(MODEL_PY, encoding="utf-8")

    if zip_path is not None:
        zpath = Path(zip_path)
        zpath.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in out.iterdir():
                if path.is_file() and path.suffix in {
                    ".yaml",
                    ".yml",
                    ".py",
                    ".json",
                    ".safetensors",
                    ".txt",
                }:
                    zf.write(path, arcname=path.name)
        return zpath
    return out
