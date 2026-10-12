"""Histology-aware data augmentation and test-time augmentation (mini sprint 1.8).

Bu dosya ne işe yarar? Eğitim sırasında patch'leri rastgele döndürür (90°'nin katları; doku
yönsüzdür), çevirir, boya tonunu hafifçe oynatır (HED uzayında: hematoksilen-eozin-DAB kanalları
ayrı ayrı ölçeklenir), parlaklık/kontrastı küçük oranda değiştirir ve nadiren bulanıklık/gürültü
ekler. Doku anlamını bozan agresif dönüşümler (büyük renk kayması, bükme) bilerek yoktur.
Tüm rastgelelik ``seed`` ile sabitlenir. Test zamanı artırma (TTA) için 8 dihedral görünüm verir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from models.histo.stain.normalizer import HED_FROM_RGB, RGB_FROM_HED


def rgb_to_hed(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Colour deconvolution into H, E, DAB optical densities (Ruifrok & Johnston)."""
    od = -np.log(np.maximum(rgb.astype(np.float64), 1.0) / 255.0)
    return od @ HED_FROM_RGB


def hed_to_rgb(hed: NDArray[np.float64]) -> NDArray[np.uint8]:
    od = np.maximum(hed @ RGB_FROM_HED, 0.0)
    rgb = 255.0 * np.exp(-od)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class AugmentConfig:
    rot90: bool = True
    flip_horizontal: bool = True
    flip_vertical: bool = True
    hed_sigma: float = 0.05  # multiplicative (1 +- sigma) and additive (+- sigma) HED jitter
    hed_p: float = 0.8
    brightness: float = 0.08  # +- fraction
    contrast: float = 0.08  # +- fraction
    color_p: float = 0.8
    blur_p: float = 0.1
    blur_sigma_max: float = 1.0
    noise_p: float = 0.1
    noise_sigma: float = 4.0  # in 0-255 units

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HistoAugment:
    """Callable ``uint8 HWC -> uint8 HWC`` with its own seeded generator (deterministic)."""

    def __init__(self, cfg: AugmentConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or AugmentConfig()
        self.rng = np.random.default_rng(seed)

    def __call__(self, rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        cfg, rng = self.cfg, self.rng
        out = rgb
        if cfg.rot90:
            out = np.rot90(out, k=int(rng.integers(0, 4)))
        if cfg.flip_horizontal and rng.random() < 0.5:
            out = out[:, ::-1]
        if cfg.flip_vertical and rng.random() < 0.5:
            out = out[::-1, :]
        out = np.ascontiguousarray(out)
        if cfg.hed_sigma > 0 and rng.random() < cfg.hed_p:
            hed = rgb_to_hed(out)
            alpha = rng.uniform(1 - cfg.hed_sigma, 1 + cfg.hed_sigma, size=3)
            beta = rng.uniform(-cfg.hed_sigma, cfg.hed_sigma, size=3)
            out = hed_to_rgb(hed * alpha + beta)
        if (cfg.brightness > 0 or cfg.contrast > 0) and rng.random() < cfg.color_p:
            b = rng.uniform(-cfg.brightness, cfg.brightness)
            c = rng.uniform(1 - cfg.contrast, 1 + cfg.contrast)
            f = out.astype(np.float32)
            f = (f - 127.5) * c + 127.5 + 255.0 * b
            out = np.clip(np.round(f), 0, 255).astype(np.uint8)
        if cfg.blur_p > 0 and rng.random() < cfg.blur_p:
            import cv2

            sigma = float(rng.uniform(0.3, cfg.blur_sigma_max))
            out = np.asarray(cv2.GaussianBlur(out, (0, 0), sigma), dtype=np.uint8)
        if cfg.noise_p > 0 and rng.random() < cfg.noise_p:
            noise = rng.normal(0.0, cfg.noise_sigma, size=out.shape)
            out = np.clip(np.round(out.astype(np.float32) + noise), 0, 255).astype(np.uint8)
        return out


def dihedral_views(rgb: NDArray[np.uint8]) -> list[NDArray[np.uint8]]:
    """The 8 rotations/flips of a square patch (for test-time augmentation)."""
    views: list[NDArray[np.uint8]] = []
    for k in range(4):
        rotated = np.rot90(rgb, k=k)
        views.append(np.ascontiguousarray(rotated))
        views.append(np.ascontiguousarray(rotated[:, ::-1]))
    return views


def tta_mean(probabilities: NDArray[np.float64]) -> float:
    """Average class probability over TTA views (shape ``(n_views,)``)."""
    return float(np.mean(probabilities))
