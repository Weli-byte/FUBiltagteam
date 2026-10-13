"""Slide-level quality control (mini sprint 1.6).

Bu dosya ne işe yarar? Her slayt için ölçülebilir kalite sayıları üretir: doku oranı, bulanık kare
oranı (Laplacian varyansı: keskin görüntüde yüksek, bulanıkta düşük), kalem/işaretçi izi oranı
(mavi-yeşil-siyah mürekkep için renk aralığı filtresi), aşırı koyu/katlanmış doku oranı. Sonra bu
sayıları eşiklerle karşılaştırıp "kullan / şüpheli / dışla" kararı ve gerekçelerini verir. Eşikler
konfigürasyondan gelir ve **kalibre edilmelidir** (``docs/histo/wsi_qc.md``): varsayılanlar
başlangıç değeridir, test performansına bakılarak ayarlanmaz.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from models.histo.wsi.reader import SlideReader
from models.histo.wsi.tissue import TissueMaskConfig, tissue_mask, tissue_stats

Decision = Literal["use", "suspect", "exclude"]
HsvRange = tuple[tuple[int, int, int], tuple[int, int, int]]

# OpenCV HSV ranges (H in 0..179, S/V in 0..255). Starting values; calibrate on real slides.
DEFAULT_PEN_RANGES: tuple[HsvRange, ...] = (
    ((95, 80, 40), (130, 255, 255)),  # blue marker
    ((40, 80, 40), (85, 255, 255)),  # green marker
    ((0, 0, 0), (179, 90, 45)),  # black marker: dark and unsaturated (nuclei are more saturated)
)


@dataclass(frozen=True)
class QcConfig:
    """Thresholds and analysis settings for :func:`compute_qc` and :func:`decide`."""

    thumbnail_px: int = 2048
    analysis_mpp: float = 2.0  # ~5x: fine enough for blur/pen, cheap enough for every slide
    tile_px: int = 256
    tile_tissue_min: float = 0.5  # only tiles that are mostly tissue are scored
    blur_laplacian_var_min: float = 40.0  # tile is "blurry" below this (calibrate!)
    dark_value_max: int = 60  # V channel below this = very dark (fold/overstain/ink)
    pen_ranges: tuple[HsvRange, ...] = DEFAULT_PEN_RANGES
    tissue_fraction_min: float = 0.02
    min_tissue_tiles: int = 10
    blur_fraction_max: float = 0.30
    pen_fraction_max: float = 0.05
    dark_fraction_max: float = 0.25
    suspect_ratio: float = 0.6  # metric above suspect_ratio * max -> "suspect"
    tissue_mask: TissueMaskConfig = field(default_factory=TissueMaskConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QcMetrics:
    slide_id: str
    width: int
    height: int
    mpp: float | None
    magnification: float | None
    level_count: int
    tissue_fraction: float
    n_components: int
    largest_component_fraction: float
    n_tissue_tiles: int
    blur_fraction: float
    pen_fraction: float
    dark_fraction: float
    laplacian_var_median: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QcDecision:
    decision: Decision
    reasons: tuple[str, ...]


def laplacian_variance(gray: NDArray[np.uint8]) -> float:
    """Sharpness measure: variance of the Laplacian (higher = sharper)."""
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pen_mask(
    rgb: NDArray[np.uint8], ranges: tuple[HsvRange, ...] = DEFAULT_PEN_RANGES
) -> NDArray[np.bool_]:
    """Pixels whose HSV colour falls into any configured ink range."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    out = np.zeros(rgb.shape[:2], dtype=bool)
    for lower, upper in ranges:
        out |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8)) > 0
    return out


def dark_mask(rgb: NDArray[np.uint8], value_max: int) -> NDArray[np.bool_]:
    """Very dark pixels (HSV value below ``value_max``): folds, overstaining, black ink."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    out: NDArray[np.bool_] = hsv[:, :, 2] < value_max
    return out


def _tiles(reader: SlideReader, cfg: QcConfig) -> Iterator[tuple[int, int, NDArray[np.uint8]]]:
    """Yield ``(x0, y0, rgb_tile)`` over the analysis level; (x0, y0) in level-0 pixels."""
    level, rescale = reader.level_for_mpp(cfg.analysis_mpp)
    downsample = reader.info.level_downsamples[level]
    tile_at_level = int(math.ceil(cfg.tile_px * rescale))
    width, height = reader.info.level_dimensions[level]
    for ty in range(0, height - tile_at_level + 1, tile_at_level):
        for tx in range(0, width - tile_at_level + 1, tile_at_level):
            x0, y0 = int(round(tx * downsample)), int(round(ty * downsample))
            yield x0, y0, reader.read_region(x0, y0, level, (tile_at_level, tile_at_level))


def compute_qc(reader: SlideReader, slide_id: str, cfg: QcConfig | None = None) -> QcMetrics:
    """Measure one slide. Tiles are scored only where the thumbnail tissue mask says 'tissue'."""
    import cv2

    cfg = cfg or QcConfig()
    thumb, thumb_ds = reader.thumbnail(cfg.thumbnail_px)
    mask = tissue_mask(thumb, cfg.tissue_mask)
    stats = tissue_stats(mask)
    size0 = reader.level0_box_size(cfg.tile_px, cfg.analysis_mpp)

    lap_values: list[float] = []
    pen_px = dark_px = tissue_px = 0
    for x0, y0, tile in _tiles(reader, cfg):
        mx0, my0 = int(x0 / thumb_ds), int(y0 / thumb_ds)
        mx1, my1 = int(math.ceil((x0 + size0) / thumb_ds)), int(math.ceil((y0 + size0) / thumb_ds))
        window = mask[my0:my1, mx0:mx1]
        if window.size == 0 or float(window.mean()) < cfg.tile_tissue_min:
            continue
        gray = np.asarray(cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
        lap_values.append(laplacian_variance(gray))
        pen_px += int(pen_mask(tile, cfg.pen_ranges).sum())
        dark_px += int(dark_mask(tile, cfg.dark_value_max).sum())
        tissue_px += int(gray.size)
    n_tiles = len(lap_values)
    lap = np.asarray(lap_values, dtype=np.float64)
    return QcMetrics(
        slide_id=slide_id,
        width=reader.info.width,
        height=reader.info.height,
        mpp=reader.info.mpp,
        magnification=reader.info.magnification,
        level_count=reader.info.level_count,
        tissue_fraction=stats["tissue_fraction"],
        n_components=int(stats["n_components"]),
        largest_component_fraction=stats["largest_component_fraction"],
        n_tissue_tiles=n_tiles,
        blur_fraction=float((lap < cfg.blur_laplacian_var_min).mean()) if n_tiles else 1.0,
        pen_fraction=pen_px / tissue_px if tissue_px else 0.0,
        dark_fraction=dark_px / tissue_px if tissue_px else 0.0,
        laplacian_var_median=float(np.median(lap)) if n_tiles else 0.0,
    )


def decide(metrics: QcMetrics, cfg: QcConfig | None = None) -> QcDecision:
    """Rule-based decision with explicit reasons (every reason names the metric and threshold)."""
    cfg = cfg or QcConfig()
    exclude: list[str] = []
    suspect: list[str] = []
    if metrics.tissue_fraction < cfg.tissue_fraction_min:
        exclude.append(f"tissue_fraction {metrics.tissue_fraction:.3f} < {cfg.tissue_fraction_min}")
    if metrics.n_tissue_tiles < cfg.min_tissue_tiles:
        exclude.append(f"n_tissue_tiles {metrics.n_tissue_tiles} < {cfg.min_tissue_tiles}")
    checks = (
        ("blur_fraction", metrics.blur_fraction, cfg.blur_fraction_max),
        ("pen_fraction", metrics.pen_fraction, cfg.pen_fraction_max),
        ("dark_fraction", metrics.dark_fraction, cfg.dark_fraction_max),
    )
    for name, value, limit in checks:
        if value > limit:
            exclude.append(f"{name} {value:.3f} > {limit}")
        elif value > cfg.suspect_ratio * limit:
            suspect.append(f"{name} {value:.3f} > {cfg.suspect_ratio * limit:.3f} (suspect)")
    if exclude:
        return QcDecision("exclude", tuple(exclude + suspect))
    if suspect:
        return QcDecision("suspect", tuple(suspect))
    return QcDecision("use", ())


def qc_row(metrics: QcMetrics, decision: QcDecision) -> dict[str, Any]:
    """Flat row for ``qc_results.parquet``."""
    row = metrics.to_dict()
    row["decision"] = decision.decision
    row["reasons"] = "; ".join(decision.reasons)
    return row
