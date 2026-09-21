"""PyTorch Dataset over IndexedSample rows."""

from __future__ import annotations

import random
from typing import Callable, List, Optional

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .augment import geometric_train_aug, maybe_robustness, val_resize
from .download import IndexedSample

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageIndexDataset(Dataset):
    def __init__(
        self,
        samples: List[IndexedSample],
        image_size: int = 224,
        train: bool = True,
        robustness_aug_prob: float = 0.0,
        seed: int = 42,
    ):
        self.samples = list(samples)
        self.image_size = image_size
        self.train = train
        self.robustness_aug_prob = robustness_aug_prob
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        try:
            img = Image.open(row.path).convert("RGB")
        except Exception:
            # fallback blank — rare corrupt files
            img = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))

        rng = random.Random(self.rng.randint(0, 10**9) + idx)
        if self.train:
            img = maybe_robustness(img, self.robustness_aug_prob, rng)
            img = geometric_train_aug(img, self.image_size, rng)
        else:
            img = val_resize(img, self.image_size)

        # float CHW in [0,1] for HF models during training
        tensor = torch.from_numpy(
            __import__("numpy").asarray(img, dtype="float32").transpose(2, 0, 1) / 255.0
        )
        return {
            "pixel_values": tensor,
            "labels": int(row.label),
            "dataset_name": row.dataset_name,
        }


def collate_batch(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}
