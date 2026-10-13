"""Uniform slide reader on top of OpenSlide (mini sprint 1.3 / 1.7).

Bu dosya ne işe yarar? Dev boyutlu doku görüntülerini (WSI: gigapiksel dijital slayt) açar,
piramit seviyelerini (aynı görüntünün farklı çözünürlükteki kopyaları) ve mikron/piksel (mpp)
bilgisini okur. "Şu koordinattan, 20x büyütmede 256 piksellik bir kare ver" gibi istekleri hangi
seviyeden okuyup nasıl ölçekleyeceğini kendisi hesaplar; böylece 20x ve 40x taranmış slaytlar
aynı fiziksel ölçeğe getirilir.

Koordinat kuralı (tüm modül için geçerli): ``x, y`` her zaman **seviye-0 pikseli** cinsindendir.
``openslide-python`` isteğe bağlıdır; ``openslide-bin`` paketi kütüphaneyi (libopenslide) kendisi
getirir, sistem paketi gerekmez. PNG/JPEG gibi düz görüntüler ``openslide.ImageSlide`` ile tek
seviyeli slayt olarak açılır (testlerde kullanılır).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray

PROP_MPP_X = "openslide.mpp-x"
PROP_MPP_Y = "openslide.mpp-y"
PROP_OBJECTIVE = "openslide.objective-power"
PROP_VENDOR = "openslide.vendor"

# Conventional magnification <-> resolution pairs (µm/px). Scanners vary slightly (e.g. 0.252 or
# 0.23 µm/px for "40x"), so matching uses a tolerance. Values are a convention, not a measurement.
MPP_BY_MAGNIFICATION: dict[float, float] = {80.0: 0.125, 40.0: 0.25, 20.0: 0.5, 10.0: 1.0, 5.0: 2.0}


def magnification_from_mpp(mpp: float, tolerance: float = 0.3) -> float | None:
    """Nearest conventional magnification for ``mpp`` (µm/px), or ``None`` if none is close.

    ``tolerance`` is relative (0.3 = 30 %); e.g. 0.252 µm/px -> 40x, 0.5 -> 20x, 0.7 -> None.
    """
    if mpp <= 0:
        raise ValueError("mpp must be positive")
    best_mag, best_err = None, math.inf
    for mag, ref in MPP_BY_MAGNIFICATION.items():
        err = abs(mpp - ref) / ref
        if err < best_err:
            best_mag, best_err = mag, err
    return best_mag if best_err <= tolerance else None


@dataclass(frozen=True)
class SlideInfo:
    """Static facts about a slide (all sizes in pixels; ``mpp`` in micrometres per pixel)."""

    path: str
    width: int
    height: int
    level_count: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    mpp_x: float | None
    mpp_y: float | None
    objective_power: float | None
    vendor: str | None

    @property
    def mpp(self) -> float | None:
        """Mean of x/y resolution, or ``None`` if the file carries no resolution metadata."""
        if self.mpp_x is None or self.mpp_y is None:
            return None
        return (self.mpp_x + self.mpp_y) / 2.0

    @property
    def magnification(self) -> float | None:
        """Objective power from metadata, else derived from ``mpp``; ``None`` if unknown."""
        if self.objective_power is not None:
            return self.objective_power
        return magnification_from_mpp(self.mpp) if self.mpp is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "level_count": self.level_count,
            "level_downsamples": list(self.level_downsamples),
            "mpp_x": self.mpp_x,
            "mpp_y": self.mpp_y,
            "mpp": self.mpp,
            "objective_power": self.objective_power,
            "magnification": self.magnification,
            "vendor": self.vendor,
        }


def _float_or_none(properties: Any, key: str) -> float | None:
    value = properties.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SlideReader:
    """Context-managed slide handle with mpp-aware region reads.

    Args:
        path: Slide file (SVS, TIFF, NDPI, MRXS ... or a plain image for tests).
        mpp_override: Resolution to assume when the file has none (e.g. CAMELYON16 TIFFs that
            lack an mpp tag). Ignored when the file carries its own value unless ``force_mpp``.
        force_mpp: Use ``mpp_override`` even if the file has metadata (for known-bad metadata).
    """

    def __init__(
        self, path: str | Path, mpp_override: float | None = None, force_mpp: bool = False
    ) -> None:
        import openslide

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._slide: Any = openslide.open_slide(str(self.path))
        props = self._slide.properties
        mpp_x, mpp_y = _float_or_none(props, PROP_MPP_X), _float_or_none(props, PROP_MPP_Y)
        if mpp_override is not None and (force_mpp or mpp_x is None or mpp_y is None):
            mpp_x = mpp_y = float(mpp_override)
        width, height = self._slide.dimensions
        self.info = SlideInfo(
            path=str(self.path),
            width=int(width),
            height=int(height),
            level_count=int(self._slide.level_count),
            level_dimensions=tuple((int(w), int(h)) for w, h in self._slide.level_dimensions),
            level_downsamples=tuple(float(d) for d in self._slide.level_downsamples),
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            objective_power=_float_or_none(props, PROP_OBJECTIVE),
            vendor=props.get(PROP_VENDOR),
        )

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> SlideReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ geometry
    @property
    def mpp(self) -> float:
        """Resolution in µm/px; raises if unknown (pass ``mpp_override`` to the constructor)."""
        mpp = self.info.mpp
        if mpp is None:
            raise ValueError(
                f"{self.path.name}: no mpp metadata; pass mpp_override (e.g. 0.25 for 40x)"
            )
        return mpp

    def level_for_mpp(self, target_mpp: float, tolerance: float = 1e-3) -> tuple[int, float]:
        """Choose the finest-enough pyramid level for ``target_mpp`` and the residual rescale.

        Returns ``(level, rescale)`` where ``rescale = desired_downsample / level_downsample``
        (>= 1: the region read at ``level`` must still be shrunk by this factor). The level is the
        coarsest one whose downsample does not exceed the desired one, so we never upsample.
        """
        desired = target_mpp / self.mpp
        if desired < 1.0 - tolerance:
            raise ValueError(
                f"target mpp {target_mpp} is finer than the slide resolution {self.mpp:.4f}"
            )
        level = 0
        for idx, downsample in enumerate(self.info.level_downsamples):
            if downsample <= desired * (1.0 + tolerance):
                level = idx
        rescale = desired / self.info.level_downsamples[level]
        return level, max(rescale, 1.0)

    def level0_box_size(self, patch_px: int, target_mpp: float) -> int:
        """Edge length in level-0 pixels of a ``patch_px`` patch taken at ``target_mpp``."""
        return int(round(patch_px * target_mpp / self.mpp))

    # ------------------------------------------------------------------ reading
    def read_region(self, x: int, y: int, level: int, size: tuple[int, int]) -> NDArray[np.uint8]:
        """RGB array of ``size=(w, h)`` pixels at ``level``; ``(x, y)`` is the level-0 top-left."""
        region = self._slide.read_region((int(x), int(y)), int(level), (int(size[0]), int(size[1])))
        rgb = np.asarray(region.convert("RGB"), dtype=np.uint8)
        return rgb

    def read_patch(self, x: int, y: int, patch_px: int, target_mpp: float) -> NDArray[np.uint8]:
        """Square ``patch_px`` patch at ``target_mpp`` whose level-0 top-left corner is (x, y)."""
        import cv2

        level, rescale = self.level_for_mpp(target_mpp)
        size_at_level = max(1, int(math.ceil(patch_px * rescale)))
        region = self.read_region(x, y, level, (size_at_level, size_at_level))
        if size_at_level == patch_px:
            return region
        resized = cv2.resize(region, (patch_px, patch_px), interpolation=cv2.INTER_AREA)
        return np.asarray(resized, dtype=np.uint8)

    def thumbnail(self, max_px: int) -> tuple[NDArray[np.uint8], float]:
        """Whole-slide RGB thumbnail with longest edge <= ``max_px`` and its level-0 downsample."""
        image = self._slide.get_thumbnail((int(max_px), int(max_px))).convert("RGB")
        thumb = np.asarray(image, dtype=np.uint8)
        downsample = self.info.width / float(thumb.shape[1])
        return thumb, downsample

    def read_level(self, level: int) -> NDArray[np.uint8]:
        """Entire pyramid level as RGB (only for small levels; guard the size before calling)."""
        w, h = self.info.level_dimensions[level]
        return self.read_region(0, 0, level, (w, h))
