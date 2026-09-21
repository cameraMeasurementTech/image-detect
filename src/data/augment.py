"""Training and robustness augmentations for SN34 image detectors."""

from __future__ import annotations

import io
import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def _to_numpy_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def jpeg_roundtrip(img: np.ndarray, quality: int = 55) -> np.ndarray:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def webp_roundtrip(img: np.ndarray, quality: int = 75) -> np.ndarray:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="WEBP", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def downscale_upscale(img: np.ndarray, scale: float = 0.5, min_px: int = 256) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    if scale >= 1.0:
        return img
    small_h = max(min_px, int(round(h * scale)))
    small_w = max(min_px, int(round(w * scale)))
    if small_h >= h or small_w >= w:
        return img
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_robustness_augmentations(
    image: Image.Image,
    seed: Optional[int] = None,
    jpeg_quality: int = 55,
    scale_factor: float = 0.5,
    webp_quality: int = 75,
) -> Image.Image:
    """Mimic gasbench apply_robustness_augmentations CDN chain."""
    rng = random.Random(seed)
    img = _to_numpy_rgb(image)
    img = downscale_upscale(img, scale=scale_factor)
    img = jpeg_roundtrip(img, quality=jpeg_quality)
    if webp_quality is not None:
        img = webp_roundtrip(img, quality=webp_quality)
    img = jpeg_roundtrip(img, quality=80)
    # slight jitter so train is not bitwise identical to eval chain
    if rng.random() < 0.3:
        img = jpeg_roundtrip(img, quality=rng.randint(40, 90))
    return Image.fromarray(img)


def maybe_robustness(
    image: Image.Image, prob: float, rng: random.Random
) -> Image.Image:
    if prob <= 0 or rng.random() >= prob:
        return image
    return apply_robustness_augmentations(
        image,
        seed=rng.randint(0, 10**9),
        jpeg_quality=rng.choice([45, 55, 65]),
        scale_factor=rng.choice([0.4, 0.5, 0.65]),
        webp_quality=rng.choice([60, 75, 85]),
    )


def geometric_train_aug(
    image: Image.Image,
    size: int,
    rng: random.Random,
) -> Image.Image:
    """Mild geometric augs from the original ViT notebook, full-frame friendly."""
    from torchvision.transforms import (
        CenterCrop,
        RandomHorizontalFlip,
        RandomResizedCrop,
        RandomRotation,
        Resize,
    )

    if rng.random() < 0.7:
        t = RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1))
    else:
        t = Resize((size, size))
    image = t(image)
    if rng.random() < 0.5:
        image = RandomHorizontalFlip(p=1.0)(image)
    if rng.random() < 0.2:
        image = RandomRotation(degrees=15)(image)
    # ensure exact size
    if image.size != (size, size):
        image = Resize((size, size))(image)
        image = CenterCrop(size)(image)
    return image


def val_resize(image: Image.Image, size: int) -> Image.Image:
    from torchvision.transforms import CenterCrop, Resize

    image = Resize((size, size))(image)
    return CenterCrop(size)(image)
