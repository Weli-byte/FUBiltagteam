"""CAMELYON16/17 annotation parsing and tumor masks (mini sprint 1.5).

Bu dosya ne işe yarar? Patologların ASAP programında çizdiği tümör çokgenlerini (XML dosyası)
okur, istenen piramit seviyesinde ikili bir "tümör haritası" (maske: 1 = tümör, 0 = değil) üretir
ve her patch için tümör oranını hesaplar. Böylece patch'ler tümör/normal diye etiketlenir.

XML biçimi (ASAP): ``ASAP_Annotations > Annotations > Annotation(Name, Type, PartOfGroup, Color)
> Coordinates > Coordinate(Order, X, Y)``; koordinatlar seviye-0 pikselidir. Grup adları veri
kopyasına göre değişir: AWS Open Data kopyasında ``Tumor`` ve ``Exclusion`` (tümör çokgeninden
çıkarılmış normal alanlar); orijinal yarışma dosyalarında ``_0``/``_1`` (tümör) ve ``_2``
(dışlama; bu yorum topluluk kodlarından çıkarımdır, resmi metin bulunamadı). Her ikisi de kabul
edilir ve konfigürasyonla değiştirilebilir.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from models.histo.wsi.tissue import box_fractions

DEFAULT_TUMOR_GROUPS: tuple[str, ...] = ("Tumor", "_0", "_1", "metastases", "Metastases")
DEFAULT_EXCLUSION_GROUPS: tuple[str, ...] = ("Exclusion", "_2")


@dataclass(frozen=True)
class Polygon:
    name: str
    group: str
    annotation_type: str
    points: NDArray[np.float64]  # (N, 2) level-0 pixels, (x, y)

    @property
    def area_level0(self) -> float:
        """Shoelace area in level-0 pixel^2 (absolute value)."""
        x, y = self.points[:, 0], self.points[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def parse_asap_xml(path: Path) -> list[Polygon]:
    """Read every ``Annotation`` with >= 3 coordinates; order of ``Coordinate`` is respected."""
    root = ET.parse(path).getroot()  # noqa: S314 - trusted local files from the official dataset
    polygons: list[Polygon] = []
    for ann in root.iter("Annotation"):
        coords = ann.find("Coordinates")
        if coords is None:
            continue
        points = sorted(
            (
                int(c.attrib.get("Order", i)),
                float(c.attrib["X"]),
                float(c.attrib["Y"]),
            )
            for i, c in enumerate(coords.findall("Coordinate"))
        )
        if len(points) < 3:
            continue
        polygons.append(
            Polygon(
                name=ann.attrib.get("Name", ""),
                group=ann.attrib.get("PartOfGroup", ""),
                annotation_type=ann.attrib.get("Type", ""),
                points=np.array([(x, y) for _, x, y in points], dtype=np.float64),
            )
        )
    return polygons


def polygons_to_mask(
    polygons: list[Polygon],
    width0: int,
    height0: int,
    downsample: float,
    tumor_groups: tuple[str, ...] = DEFAULT_TUMOR_GROUPS,
    exclusion_groups: tuple[str, ...] = DEFAULT_EXCLUSION_GROUPS,
) -> NDArray[np.uint8]:
    """Rasterise tumor polygons at ``downsample`` (level-0 px per mask px); exclusions are cut out.

    Unknown groups are ignored (listed by :func:`unknown_groups` so they are not silently lost).
    """
    import cv2

    w = int(np.ceil(width0 / downsample))
    h = int(np.ceil(height0 / downsample))
    mask = np.zeros((h, w), dtype=np.uint8)
    tumor = [p for p in polygons if p.group in tumor_groups]
    excl = [p for p in polygons if p.group in exclusion_groups]
    if tumor:
        pts = [np.round(p.points / downsample).astype(np.int32).reshape(-1, 1, 2) for p in tumor]
        cv2.fillPoly(mask, pts, 1)
    if excl:
        pts = [np.round(p.points / downsample).astype(np.int32).reshape(-1, 1, 2) for p in excl]
        cv2.fillPoly(mask, pts, 0)
    return mask


def unknown_groups(
    polygons: list[Polygon],
    tumor_groups: tuple[str, ...] = DEFAULT_TUMOR_GROUPS,
    exclusion_groups: tuple[str, ...] = DEFAULT_EXCLUSION_GROUPS,
) -> set[str]:
    return {p.group for p in polygons} - set(tumor_groups) - set(exclusion_groups)


def patch_tumor_fractions(
    mask: NDArray[np.uint8], downsample: float, coords: NDArray[np.int32], size0: int
) -> NDArray[np.float32]:
    """Tumor fraction (0-1) of each level-0 box; same geometry rule as the tissue filter."""
    return box_fractions(mask, downsample, coords, size0)


def label_patches(
    tumor_fraction: NDArray[np.float32], tumor_min: float = 0.5, normal_max: float = 0.0
) -> NDArray[np.int8]:
    """1 = tumor (fraction >= ``tumor_min``), 0 = normal (<= ``normal_max``), -1 = ambiguous."""
    labels = np.full(len(tumor_fraction), -1, dtype=np.int8)
    labels[tumor_fraction >= tumor_min] = 1
    labels[tumor_fraction <= normal_max] = 0
    return labels


def overlay_mask(
    rgb: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> NDArray[np.uint8]:
    """Blend ``color`` over mask pixels (mask is resized to the image if needed); visual check."""
    import cv2

    if mask.shape != rgb.shape[:2]:
        resized = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.asarray(resized, dtype=np.uint8)
    out = rgb.astype(np.float32)
    tint = np.array(color, dtype=np.float32)
    sel = mask > 0
    out[sel] = (1 - alpha) * out[sel] + alpha * tint
    return out.clip(0, 255).astype(np.uint8)


def slide_has_tumor(
    polygons: list[Polygon], tumor_groups: tuple[str, ...] = DEFAULT_TUMOR_GROUPS
) -> bool:
    return any(p.group in tumor_groups for p in polygons)
