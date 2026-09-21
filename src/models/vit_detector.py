"""Train-time and export-time image detector models."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Uint8ImageClassifier(nn.Module):
    """Wrapper that accepts uint8 NCHW [0,255] and returns logits — gasbench contract."""

    def __init__(
        self,
        backbone: nn.Module,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        temperature: float = 1.0,
    ):
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
        # x: [B,3,H,W] uint8 or float
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.dtype != torch.float32:
            x = x.float()
            if x.max() > 1.5:
                x = x / 255.0
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x) if _is_hf(self.backbone) else self.backbone(x)
        logits = out.logits if hasattr(out, "logits") else out
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits


def _is_hf(module: nn.Module) -> bool:
    return hasattr(module, "config") and hasattr(module, "forward")


def load_hf_classifier(backbone_id: str, num_classes: int = 3) -> nn.Module:
    return AutoModelForImageClassification.from_pretrained(
        backbone_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )


def load_processor(backbone_id: str):
    return AutoImageProcessor.from_pretrained(backbone_id)


class LogitAverageEnsemble(nn.Module):
    """Simple heterogeneous ensemble: average logits from multiple HF classifiers."""

    def __init__(self, branches: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        if weights is None:
            weights = [1.0 / len(branches)] * len(branches)
        total = sum(weights)
        self.register_buffer(
            "weights",
            torch.tensor([w / total for w in weights], dtype=torch.float32),
        )

    def forward(self, pixel_values: torch.Tensor = None, x: torch.Tensor = None, **kwargs):
        inp = pixel_values if pixel_values is not None else x
        logits = None
        for i, branch in enumerate(self.branches):
            out = branch(pixel_values=inp) if _expects_pixel_values(branch) else branch(inp)
            branch_logits = out.logits if hasattr(out, "logits") else out
            w = self.weights[i]
            logits = branch_logits * w if logits is None else logits + branch_logits * w
        return type("Out", (), {"logits": logits})()


def _expects_pixel_values(module: nn.Module) -> bool:
    try:
        import inspect

        return "pixel_values" in inspect.signature(module.forward).parameters
    except Exception:
        return True


def build_train_model(cfg: Dict) -> nn.Module:
    model_cfg = cfg.get("model", cfg)
    num_classes = int(model_cfg.get("num_classes", 3))
    ensemble_cfg = model_cfg.get("ensemble") or {}
    if ensemble_cfg.get("enabled"):
        branches = []
        weights = []
        for b in ensemble_cfg["branches"]:
            branches.append(load_hf_classifier(b["backbone"], num_classes))
            weights.append(float(b.get("weight", 1.0)))
        return LogitAverageEnsemble(branches, weights)
    return load_hf_classifier(model_cfg["backbone"], num_classes)


def build_export_wrapper(
    backbone: nn.Module,
    temperature: float = 1.0,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> Uint8ImageClassifier:
    return Uint8ImageClassifier(backbone, mean=mean, std=std, temperature=temperature)
