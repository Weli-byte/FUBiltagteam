"""Tissue detection on slide thumbnails (mini sprint 1.3 v0, 1.6 v1).

Bu dosya ne işe yarar? Slaytın küçük bir kopyasında (thumbnail) beyaz cam arka planı ile boyalı
dokuyu ayırır. Yöntem: renk doygunluğu (HSV'nin S kanalı; boyalı doku doygun, cam değildir) üzerinde
Otsu eşiği (iki sınıfı en iyi ayıran eşik değerini otomatik bulan yöntem). v1 bunu morfolojik
açma/kapama (küçük lekeleri silme, delikleri kapama) ve küçük nesne temizliği ile iyileştirir.
Çıktı bir "doku maskesi"dir: True = doku, False = arka plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

TISSUE_MASK_VERSION = "1.1"  # bump when the algorithm changes (stored in HDF5 attrs)


@dataclass(frozen=True)
class TissueMaskConfig:
    """Thresholds for :func:`tissue_mask` (thumbnail pixel units unless stated otherwise).

    v0 (mini sprint 1.3) = ``morphology=False, min_object_px=0, fill_holes=False``.
    v1 (mini sprint 1.6) = defaults below.
    """

    median_blur_px: int = 7
    saturation_floor: int = 15
    morphology: bool = True
    open_px: int = 5
    close_px: int = 9
    min_object_px: int = 400
    fill_holes: bool = True
    max_hole_px: int = 2000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _ellipse(k: int) -> NDArray[np.uint8]:
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))
    return np.asarray(kernel, dtype=np.uint8)


def remove_small_objects(mask: NDArray[np.bool_], min_px: int) -> NDArray[np.bool_]:
    """Drop connected components smaller than ``min_px`` pixels (8-connectivity)."""
    import cv2

    if min_px <= 0 or not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    out: NDArray[np.bool_] = keep[labels]
    return out


def fill_small_holes(mask: NDArray[np.bool_], max_px: int) -> NDArray[np.bool_]:
    """Fill background holes (components not touching the border) smaller than ``max_px``."""
    import cv2

    if max_px <= 0:
        return mask
    background = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    h, w = mask.shape
    fill = np.zeros(n, dtype=bool)
    for idx in range(1, n):
        x, y, cw, ch, area = stats[idx]
        touches_border = x == 0 or y == 0 or x + cw == w or y + ch == h
        fill[idx] = (not touches_border) and area <= max_px
    out: NDArray[np.bool_] = mask | fill[labels]
    return out


def tissue_mask(rgb: NDArray[np.uint8], cfg: TissueMaskConfig | None = None) -> NDArray[np.bool_]:
    """Boolean tissue mask of an RGB thumbnail (True = tissue).

    Steps: HSV saturation -> median blur -> Otsu threshold (never below ``saturation_floor``) ->
    optional morphology (open, close) -> optional small-object removal and hole filling.
    """
    import cv2

    cfg = cfg or TissueMaskConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an (H, W, 3) uint8 array")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    if cfg.median_blur_px > 1:
        sat = cv2.medianBlur(sat, _odd(cfg.median_blur_px))
    otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(float(otsu), float(cfg.saturation_floor))
    mask: NDArray[np.bool_] = sat > threshold
    if cfg.morphology:
        m8 = mask.astype(np.uint8)
        if cfg.open_px > 1:
            m8 = np.asarray(cv2.morphologyEx(m8, cv2.MORPH_OPEN, _ellipse(cfg.open_px)), np.uint8)
        if cfg.close_px > 1:
            m8 = np.asarray(cv2.morphologyEx(m8, cv2.MORPH_CLOSE, _ellipse(cfg.close_px)), np.uint8)
        mask = m8.astype(bool)
    if cfg.min_object_px > 0:
        mask = remove_small_objects(mask, cfg.min_object_px)
    if cfg.fill_holes:
        mask = fill_small_holes(mask, cfg.max_hole_px)
    return mask


def tissue_stats(mask: NDArray[np.bool_]) -> dict[str, float]:
    """Summary numbers for QC: tissue fraction, component count, largest component share."""
    import cv2

    total = float(mask.size)
    if not mask.any():
        return {"tissue_fraction": 0.0, "n_components": 0.0, "largest_component_fraction": 0.0}
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return {
        "tissue_fraction": float(mask.sum()) / total,
        "n_components": float(n - 1),
        "largest_component_fraction": float(areas.max()) / float(mask.sum()),
    }


def integral_image(mask: NDArray[np.bool_] | NDArray[np.uint8]) -> NDArray[np.int64]:
    """Summed-area table with a zero top row/column, so box sums are 4 lookups."""
    table = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    table[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    return table


def box_fractions(
    mask: NDArray[np.bool_] | NDArray[np.uint8],
    downsample: float,
    coords: NDArray[np.int32],
    size0: int,
) -> NDArray[np.float32]:
    """Fraction of mask pixels inside each level-0 box ``(x, y, size0, size0)``.

    ``downsample`` is level-0 pixels per mask pixel. Boxes are clipped to the mask; a box that
    falls completely outside gets 0. Works for tissue masks and tumor masks alike.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")
    if len(coords) == 0:
        return np.zeros(0, dtype=np.float32)
    table = integral_image(mask)
    h, w = mask.shape
    x0 = np.clip(np.floor(coords[:, 0] / downsample).astype(np.int64), 0, w)
    y0 = np.clip(np.floor(coords[:, 1] / downsample).astype(np.int64), 0, h)
    x1 = np.clip(np.ceil((coords[:, 0] + size0) / downsample).astype(np.int64), 0, w)
    y1 = np.clip(np.ceil((coords[:, 1] + size0) / downsample).astype(np.int64), 0, h)
    area = (x1 - x0) * (y1 - y0)
    total = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    fractions = np.where(area > 0, total / np.maximum(area, 1), 0.0)
    return fractions.astype(np.float32)
