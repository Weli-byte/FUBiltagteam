"""Stain normalization: Macenko and Reinhard (mini sprint 1.8).

Bu dosya ne işe yarar? Farklı hastanelerde farklı tonda boyanmış doku görüntülerini ortak bir
renk görünümüne getirir. Macenko yöntemi görüntüyü "optik yoğunluk" uzayına çevirip hematoksilen
(mor, çekirdek) ve eozin (pembe, sitoplazma) boya vektörlerini tekil değer ayrışımı ile bulur,
boya yoğunluklarını hedef görüntüye göre ölçekler ve geri çevirir. Reinhard yöntemi LAB renk
uzayında ortalama/standart sapmayı hedefe eşler. İkisi de saf numpy/OpenCV ile yazıldı;
GPU'da toplu uygulama için ``torchstain`` (MIT) alternatif olarak not edilmiştir.

Kaynaklar (doğrulanmalı): Macenko ve ark., ISBI 2009; Reinhard ve ark., IEEE CG&A 2001.
Normalizasyon her zaman iyileştirme sağlamaz; etkisi ``scripts/compare_stain_norm.py`` ile ölçülür.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Ruifrok & Johnston (2001) H&E(+DAB) stain vectors as used by scikit-image's rgb2hed.
RGB_FROM_HED: NDArray[np.float64] = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float64
)
HED_FROM_RGB: NDArray[np.float64] = np.linalg.inv(RGB_FROM_HED).astype(np.float64)


def rgb_to_od(rgb: NDArray[np.uint8], background: float = 255.0) -> NDArray[np.float64]:
    """Optical density ``-log(I / I0)``; pixels are clipped to >= 1 so the log is finite."""
    image = np.maximum(rgb.astype(np.float64), 1.0)
    return -np.log(image / background)


def od_to_rgb(od: NDArray[np.float64], background: float = 255.0) -> NDArray[np.uint8]:
    rgb = background * np.exp(-od)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


class StainNormalizer:
    """Common interface: ``fit(target)`` once, then ``transform(image)`` many times."""

    name = "base"

    def fit(self, target_rgb: NDArray[np.uint8]) -> StainNormalizer:
        raise NotImplementedError

    def transform(self, rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        """Fitted parameters (JSON-serialisable) for provenance."""
        return {}


class IdentityNormalizer(StainNormalizer):
    name = "none"

    def fit(self, target_rgb: NDArray[np.uint8]) -> StainNormalizer:
        return self

    def transform(self, rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return rgb


@dataclass
class MacenkoNormalizer(StainNormalizer):
    """Macenko et al. 2009 stain separation with concentration scaling to a target image.

    Args:
        alpha: Percentile (in %) defining the robust extremes of stain angles (paper: 1).
        beta: OD threshold below which pixels count as background (paper: 0.15).
        max_percentile: Percentile used for the per-stain "maximum concentration" (99).
        min_tissue_pixels: ``fit``/``transform`` need at least this many foreground pixels.
    """

    alpha: float = 1.0
    beta: float = 0.15
    max_percentile: float = 99.0
    min_tissue_pixels: int = 64
    name: str = "macenko"
    stain_matrix_target: NDArray[np.float64] | None = None
    max_conc_target: NDArray[np.float64] | None = None

    def _stain_matrix(self, rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
        od = rgb_to_od(rgb).reshape(-1, 3)
        od = od[(od > self.beta).all(axis=1)]
        if len(od) < self.min_tissue_pixels:
            raise ValueError(
                f"not enough tissue pixels for Macenko ({len(od)} < {self.min_tissue_pixels})"
            )
        _, eigvecs = np.linalg.eigh(np.cov(od.T))
        basis = eigvecs[:, 1:3]  # two largest eigenvalues (eigh sorts ascending)
        if basis[0, 0] < 0:
            basis[:, 0] *= -1
        if basis[0, 1] < 0:
            basis[:, 1] *= -1
        projected = od @ basis
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        low, high = np.percentile(angles, [self.alpha, 100.0 - self.alpha])
        v_min = basis @ np.array([np.cos(low), np.sin(low)])
        v_max = basis @ np.array([np.cos(high), np.sin(high)])
        # Hematoxylin has the larger red OD component; keep the (H, E) order stable.
        he = np.array([v_min, v_max]) if v_min[0] > v_max[0] else np.array([v_max, v_min])
        he /= np.linalg.norm(he, axis=1, keepdims=True)
        return he.astype(np.float64)  # (2, 3): rows are unit stain vectors

    @staticmethod
    def _concentrations(od: NDArray[np.float64], he: NDArray[np.float64]) -> NDArray[np.float64]:
        """Least-squares stain concentrations: ``od (N, 3) ~= conc (N, 2) @ he (2, 3)``."""
        conc, *_ = np.linalg.lstsq(he.T, od.T, rcond=None)
        return np.asarray(conc.T, dtype=np.float64)

    def fit(self, target_rgb: NDArray[np.uint8]) -> MacenkoNormalizer:
        he = self._stain_matrix(target_rgb)
        conc = self._concentrations(rgb_to_od(target_rgb).reshape(-1, 3), he)
        self.stain_matrix_target = he
        self.max_conc_target = np.percentile(conc, self.max_percentile, axis=0)
        return self

    def transform(self, rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.stain_matrix_target is None or self.max_conc_target is None:
            raise RuntimeError("call fit(target) first")
        h, w, _ = rgb.shape
        he = self._stain_matrix(rgb)
        conc = self._concentrations(rgb_to_od(rgb).reshape(-1, 3), he)
        max_conc = np.percentile(conc, self.max_percentile, axis=0)
        conc *= self.max_conc_target / np.maximum(max_conc, 1e-6)
        od = conc @ self.stain_matrix_target
        return od_to_rgb(od.reshape(h, w, 3))

    def state(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "stain_matrix_target": (
                self.stain_matrix_target.tolist() if self.stain_matrix_target is not None else None
            ),
            "max_conc_target": (
                self.max_conc_target.tolist() if self.max_conc_target is not None else None
            ),
        }


def rgb_to_lab(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """CIELAB with L in [0, 100], a/b roughly in [-127, 127] (OpenCV float convention)."""
    import cv2

    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return np.asarray(lab, dtype=np.float32)


def lab_to_rgb(lab: NDArray[np.float32]) -> NDArray[np.uint8]:
    import cv2

    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def lab_stats(rgb: NDArray[np.uint8], mask: NDArray[np.bool_] | None = None) -> NDArray[np.float64]:
    """``[mean_L, mean_a, mean_b, std_L, std_a, std_b]`` over all (or masked) pixels."""
    lab = rgb_to_lab(rgb).reshape(-1, 3)
    if mask is not None:
        lab = lab[mask.reshape(-1)]
    return np.concatenate([lab.mean(axis=0), lab.std(axis=0)]).astype(np.float64)


@dataclass
class ReinhardNormalizer(StainNormalizer):
    """Reinhard et al. 2001: match per-channel LAB mean and standard deviation to a target."""

    name: str = "reinhard"
    target_mean: NDArray[np.float64] | None = None
    target_std: NDArray[np.float64] | None = None

    def fit(self, target_rgb: NDArray[np.uint8]) -> ReinhardNormalizer:
        stats = lab_stats(target_rgb)
        self.target_mean, self.target_std = stats[:3], stats[3:]
        return self

    def transform(self, rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError("call fit(target) first")
        lab = rgb_to_lab(rgb)
        mean = lab.reshape(-1, 3).mean(axis=0)
        std = np.maximum(lab.reshape(-1, 3).std(axis=0), 1e-6)
        out = (lab - mean) / std * self.target_std + self.target_mean
        return lab_to_rgb(out.astype(np.float32))

    def state(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "target_mean": self.target_mean.tolist() if self.target_mean is not None else None,
            "target_std": self.target_std.tolist() if self.target_std is not None else None,
        }


def make_normalizer(method: str, **kwargs: Any) -> StainNormalizer:
    """Factory: ``"none" | "macenko" | "reinhard"``."""
    if method == "none":
        return IdentityNormalizer()
    if method == "macenko":
        return MacenkoNormalizer(**kwargs)
    if method == "reinhard":
        return ReinhardNormalizer(**kwargs)
    raise ValueError(f"unknown stain normalization method {method!r}")
