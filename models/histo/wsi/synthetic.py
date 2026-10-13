"""Synthetic pyramidal slides for tests, demos and environment checks.

Bu dosya ne işe yarar? Gerçek slayt (yüzlerce MB-GB) olmadan hattı test edebilmek için küçük,
OpenSlide'ın okuyabildiği piramit TIFF dosyaları üretir: beyaz cam üzerinde pembe "doku", içinde
daha mor "tümör" bölgesi, isteğe bağlı mavi kalem izi ve bulanık bölge. Ayrıca aynı tümör bölgesi
için CAMELYON16 biçiminde (ASAP XML) anotasyon yazar. **Bu veri gerçek doku değildir**; yalnızca
geometri, koordinat ve biçim testleri içindir, bilimsel sonuç üretmek için kullanılmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
from numpy.typing import NDArray

Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 in level-0 pixels (x1/y1 exclusive)


@dataclass(frozen=True)
class SyntheticSlideSpec:
    width: int = 1536
    height: int = 1024
    mpp: float = 0.5  # µm/px of level 0 ("20x")
    levels: int = 4
    tile_px: int = 256
    tissue_center: tuple[float, float] = (0.5, 0.5)  # fraction of width/height
    tissue_radius: tuple[float, float] = (0.33, 0.35)  # fraction of width/height
    tissue_color: tuple[int, int, int] = (226, 160, 196)  # eosin pink; vary per 'centre'
    tumor_rect: Rect | None = (900, 300, 1300, 700)
    pen_rect: Rect | None = None
    blur_rect: Rect | None = None
    nuclei_per_1000px2: float = 0.6
    seed: int = 0


def _draw_nuclei(
    img: NDArray[np.uint8],
    region: NDArray[np.bool_],
    rng: np.random.Generator,
    density: float,
    color: tuple[int, int, int],
) -> None:
    import cv2

    n = int(region.sum() * density / 1000.0)
    ys, xs = np.nonzero(region)
    if n == 0 or len(ys) == 0:
        return
    pick = rng.integers(0, len(ys), size=n)
    for y, x in zip(ys[pick], xs[pick], strict=True):
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 5)), color, -1)


def render_synthetic_rgb(spec: SyntheticSlideSpec) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Return ``(rgb level-0 image, tissue mask)``; deterministic for a given ``spec.seed``."""
    import cv2

    rng = np.random.default_rng(spec.seed)
    h, w = spec.height, spec.width
    img = np.full((h, w, 3), 242, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = spec.tissue_center[0] * w, spec.tissue_center[1] * h
    rx, ry = spec.tissue_radius[0] * w, spec.tissue_radius[1] * h
    if rx <= 0 or ry <= 0:  # an empty glass slide
        tissue = np.zeros((h, w), dtype=bool)
    else:
        tissue = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0
    base = np.array(spec.tissue_color, dtype=np.int16)
    noise = rng.integers(-12, 12, size=(h, w, 3), dtype=np.int16)
    img[tissue] = np.clip(base + noise[tissue], 0, 255).astype(np.uint8)
    _draw_nuclei(img, tissue, rng, spec.nuclei_per_1000px2, (96, 48, 128))
    if spec.tumor_rect is not None:
        x0, y0, x1, y1 = spec.tumor_rect
        tumor = np.zeros_like(tissue)
        tumor[y0:y1, x0:x1] = True
        tumor &= tissue
        img[tumor] = np.clip(np.array([190, 120, 190], np.int16) + noise[tumor], 0, 255).astype(
            np.uint8
        )
        _draw_nuclei(img, tumor, rng, spec.nuclei_per_1000px2 * 3.0, (70, 30, 110))
    if spec.pen_rect is not None:
        x0, y0, x1, y1 = spec.pen_rect
        img[y0:y1, x0:x1] = (30, 60, 200)  # blue marker
    if spec.blur_rect is not None:
        x0, y0, x1, y1 = spec.blur_rect
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), 6.0)
    return img, tissue


def write_pyramid_tiff(
    path: Path, rgb: NDArray[np.uint8], mpp: float, levels: int, tile: int
) -> None:
    """Write an OpenSlide-readable 'generic-tiff' pyramid (tiled pages, reduced-image flags)."""
    import tifffile

    level_images = [rgb]
    for _ in range(levels - 1):
        level_images.append(level_images[-1][::2, ::2])
    with tifffile.TiffWriter(str(path)) as tif:
        for idx, image in enumerate(level_images):
            res = 1e4 / (mpp * 2**idx)  # pixels per centimetre
            tif.write(
                image,
                tile=(tile, tile),
                photometric="rgb",
                compression="zlib",
                resolution=(res, res),
                resolutionunit="CENTIMETER",
                subfiletype=0 if idx == 0 else 1,
            )


def write_synthetic_slide(path: Path, spec: SyntheticSlideSpec | None = None) -> SyntheticSlideSpec:
    """Render and write a synthetic slide; returns the spec used (handy for assertions)."""
    spec = spec or SyntheticSlideSpec()
    rgb, _ = render_synthetic_rgb(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pyramid_tiff(path, rgb, spec.mpp, spec.levels, spec.tile_px)
    return spec


def rect_polygon(rect: Rect) -> NDArray[np.float64]:
    x0, y0, x1, y1 = rect
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def write_asap_xml(path: Path, polygons: list[tuple[str, str, NDArray[np.float64]]]) -> None:
    """Write CAMELYON16-style ASAP annotations: ``(name, group, points (N, 2) level-0)``."""
    lines = ['<?xml version="1.0"?>', "<ASAP_Annotations>", "\t<Annotations>"]
    for name, group, points in polygons:
        lines.append(
            f'\t\t<Annotation Name={quoteattr(name)} Type="Polygon" '
            f'PartOfGroup={quoteattr(group)} Color="#F4FA58">'
        )
        lines.append("\t\t\t<Coordinates>")
        for order, (x, y) in enumerate(points):
            lines.append(f'\t\t\t\t<Coordinate Order="{order}" X="{x:.1f}" Y="{y:.1f}" />')
        lines.append("\t\t\t</Coordinates>")
        lines.append("\t\t</Annotation>")
    lines += ["\t</Annotations>", "\t<AnnotationGroups>"]
    for group in sorted({g for _, g, _ in polygons}):
        lines.append(
            f'\t\t<Group Name={quoteattr(group)} PartOfGroup="None" Color="#64FE2E">'
            "<Attributes /></Group>"
        )
    lines += ["\t</AnnotationGroups>", "</ASAP_Annotations>", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
