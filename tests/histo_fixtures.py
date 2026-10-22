"""Helpers that build small synthetic slides for the histo tests.

The slides are rendered by ``tests.histo_synthetic_fixture`` (pink 'tissue' on white glass with a
purple 'tumor' rectangle). They are geometry fixtures only, never real tissue, and exist so the
pipeline can be tested without downloading hundreds of gigabytes.
"""

from __future__ import annotations

from pathlib import Path

from tests.histo_synthetic_fixture import (
    SyntheticSlideSpec,
    rect_polygon,
    write_asap_xml,
    write_synthetic_slide,
)

DEFAULT_TUMOR = (900, 300, 1300, 700)


def make_slide(
    directory: Path,
    name: str = "slide_a",
    mpp: float = 0.5,
    width: int = 1536,
    height: int = 1024,
    tumor_rect: tuple[int, int, int, int] | None = DEFAULT_TUMOR,
    pen_rect: tuple[int, int, int, int] | None = None,
    blur_rect: tuple[int, int, int, int] | None = None,
    seed: int = 0,
    levels: int = 4,
) -> tuple[Path, SyntheticSlideSpec]:
    """Write ``<directory>/<name>.tif`` and return its path and spec."""
    spec = SyntheticSlideSpec(
        width=width,
        height=height,
        mpp=mpp,
        levels=levels,
        tumor_rect=tumor_rect,
        pen_rect=pen_rect,
        blur_rect=blur_rect,
        seed=seed,
    )
    path = directory / f"{name}.tif"
    write_synthetic_slide(path, spec)
    return path, spec


def make_xml(
    directory: Path,
    name: str,
    tumor_rect: tuple[int, int, int, int] = DEFAULT_TUMOR,
    group: str = "Tumor",
    exclusion_rect: tuple[int, int, int, int] | None = None,
) -> Path:
    """Write an ASAP XML with one tumor polygon (and optionally one exclusion polygon)."""
    polygons = [("Annotation 0", group, rect_polygon(tumor_rect))]
    if exclusion_rect is not None:
        polygons.append(("Annotation 1", "Exclusion", rect_polygon(exclusion_rect)))
    path = directory / f"{name}.xml"
    write_asap_xml(path, polygons)
    return path
