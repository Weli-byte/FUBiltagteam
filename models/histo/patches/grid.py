"""Patch grid generation at a target resolution (mini sprint 1.3 / 1.7).

Bu dosya ne işe yarar? Slaytı, hedef büyütmede (örn. 20x = 0,5 µm/px) sabit boyutlu karelere
(patch) bölen ızgara koordinatlarını üretir ve yalnızca yeterince doku içeren kareleri bırakır.
40x taranmış bir slaytta aynı fiziksel alanı kapsamak için seviye-0'da daha büyük bir kutu
(512 px) alınır; 20x slaytta 256 px. Tüm koordinatlar seviye-0 pikselidir (sol-üst köşe).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from models.histo.wsi.reader import SlideReader
from models.histo.wsi.tissue import TissueMaskConfig, box_fractions, tissue_mask


@dataclass(frozen=True)
class PatchGridConfig:
    """Grid parameters. ``patch_px``/``stride_px`` are in pixels **at target_mpp**."""

    patch_px: int = 256
    target_mpp: float = 0.5
    stride_px: int | None = None  # None -> patch_px (no overlap)
    min_tissue_fraction: float = 0.5
    thumbnail_px: int = 2048

    @property
    def stride(self) -> int:
        return self.stride_px or self.patch_px

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def grid_coords(width: int, height: int, size0: int, stride0: int) -> NDArray[np.int32]:
    """All level-0 top-left corners ``(x, y)`` of ``size0`` boxes fully inside ``width x height``.

    Row-major order (y outer, x inner) so output is deterministic.
    """
    if size0 <= 0 or stride0 <= 0:
        raise ValueError("size0 and stride0 must be positive")
    xs = np.arange(0, width - size0 + 1, stride0, dtype=np.int64)
    ys = np.arange(0, height - size0 + 1, stride0, dtype=np.int64)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys)
    coords = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return coords.astype(np.int32)


def level0_sizes(reader: SlideReader, cfg: PatchGridConfig) -> tuple[int, int]:
    """``(size0, stride0)``: patch edge and stride in level-0 pixels for this slide."""
    size0 = reader.level0_box_size(cfg.patch_px, cfg.target_mpp)
    stride0 = reader.level0_box_size(cfg.stride, cfg.target_mpp)
    return size0, stride0


@dataclass(frozen=True)
class GridResult:
    coords: NDArray[np.int32]  # (N, 2) level-0, tissue-filtered
    tissue_fraction: NDArray[np.float32]  # (N,)
    n_candidates: int  # grid boxes before the tissue filter
    size0: int
    stride0: int
    thumbnail: NDArray[np.uint8]
    mask: NDArray[np.bool_]
    thumbnail_downsample: float


def tissue_patch_grid(
    reader: SlideReader, cfg: PatchGridConfig, mask_cfg: TissueMaskConfig | None = None
) -> GridResult:
    """Thumbnail -> tissue mask -> grid -> keep boxes with >= ``min_tissue_fraction`` tissue."""
    thumb, downsample = reader.thumbnail(cfg.thumbnail_px)
    mask = tissue_mask(thumb, mask_cfg)
    size0, stride0 = level0_sizes(reader, cfg)
    candidates = grid_coords(reader.info.width, reader.info.height, size0, stride0)
    fractions = box_fractions(mask, downsample, candidates, size0)
    keep = fractions >= cfg.min_tissue_fraction
    return GridResult(
        coords=candidates[keep],
        tissue_fraction=fractions[keep],
        n_candidates=int(len(candidates)),
        size0=size0,
        stride0=stride0,
        thumbnail=thumb,
        mask=mask,
        thumbnail_downsample=downsample,
    )


def draw_boxes(
    thumb: NDArray[np.uint8],
    downsample: float,
    coords: NDArray[np.int32],
    size0: int,
    color: tuple[int, int, int] = (0, 180, 0),
    thickness: int = 1,
) -> NDArray[np.uint8]:
    """Overlay level-0 boxes on a thumbnail (for the visual coordinate check)."""
    import cv2

    out = thumb.copy()
    for x, y in coords:
        p0 = (int(x / downsample), int(y / downsample))
        p1 = (int((x + size0) / downsample), int((y + size0) / downsample))
        cv2.rectangle(out, p0, p1, color, thickness)
    return out
