"""Tissue detection and slide QC on synthetic slides (mini sprint 1.3 / 1.6)."""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openslide")
pytest.importorskip("cv2")
pytest.importorskip("tifffile")
np = pytest.importorskip("numpy")

from models.histo.wsi.qc import QcConfig, compute_qc, decide, pen_mask, qc_row  # noqa: E402
from models.histo.wsi.reader import SlideReader  # noqa: E402
from models.histo.wsi.tissue import (  # noqa: E402
    TissueMaskConfig,
    box_fractions,
    tissue_mask,
    tissue_stats,
)
from tests.histo_fixtures import make_slide  # noqa: E402
from tests.histo_synthetic_fixture import SyntheticSlideSpec, render_synthetic_rgb  # noqa: E402

V0 = TissueMaskConfig(morphology=False, min_object_px=0, fill_holes=False)


def _iou(a: Any, b: Any) -> float:
    return float((a & b).sum()) / float((a | b).sum())


def test_mask_recovers_the_tissue_blob() -> None:
    rgb, truth = render_synthetic_rgb(SyntheticSlideSpec(width=768, height=512))
    for cfg in (V0, TissueMaskConfig()):
        mask = tissue_mask(rgb, cfg)
        assert mask.shape == truth.shape and mask.dtype == bool
        assert _iou(mask, truth) > 0.9


def test_blank_slide_has_no_tissue() -> None:
    blank = np.full((256, 384, 3), 242, dtype=np.uint8)
    assert not tissue_mask(blank).any()
    assert tissue_stats(tissue_mask(blank))["tissue_fraction"] == 0.0


def test_v1_removes_specks_and_fills_holes() -> None:
    rgb, truth = render_synthetic_rgb(SyntheticSlideSpec(width=768, height=512, seed=3))
    rgb = rgb.copy()
    rgb[20:29, 20:29] = (200, 120, 170)  # a 9x9 speck of 'tissue' on the glass (81 px)
    centre = (truth.shape[0] // 2, truth.shape[1] // 2)
    rgb[centre[0] - 8 : centre[0] + 8, centre[1] - 8 : centre[1] + 8] = 242  # small hole
    v0 = tissue_mask(rgb, V0)
    v1 = tissue_mask(rgb, TissueMaskConfig(min_object_px=100, max_hole_px=500))
    assert v0[24, 24] and not v1[24, 24]
    assert not v0[centre] and v1[centre]
    assert tissue_stats(v1)["n_components"] == 1


def test_box_fractions_match_brute_force() -> None:
    rng = np.random.default_rng(0)
    mask = rng.random((50, 70)) > 0.5
    coords = np.array([[0, 0], [40, 20], [120, 60], [500, 500]], dtype=np.int32)
    fractions = box_fractions(mask, 2.0, coords, 32)  # boxes of 16x16 mask pixels
    for (x, y), value in zip(coords, fractions, strict=True):
        window = mask[y // 2 : (y + 32) // 2, x // 2 : (x + 32) // 2]
        expected = window.mean() if window.size else 0.0
        assert value == pytest.approx(expected, abs=1e-6)
    assert fractions[-1] == 0.0  # fully outside


def test_pen_mask_finds_blue_ink() -> None:
    rgb = np.full((64, 64, 3), 230, dtype=np.uint8)
    rgb[10:30, 10:30] = (30, 60, 200)
    found = pen_mask(rgb)
    assert found[10:30, 10:30].all() and not found[40:, 40:].any()


def _qc(path: Path, cfg: QcConfig) -> tuple[Any, Any]:
    with SlideReader(path) as reader:
        metrics = compute_qc(reader, path.stem, cfg)
    return metrics, decide(metrics, cfg)


def test_clean_pen_blur_and_blank_slides_are_separated(tmp_path: Path) -> None:
    cfg = QcConfig(thumbnail_px=512, analysis_mpp=1.0, tile_px=64, min_tissue_tiles=4)
    clean, _ = make_slide(tmp_path, "clean")
    pen, _ = make_slide(tmp_path, "pen", pen_rect=(500, 250, 1000, 450))
    blur, _ = make_slide(tmp_path, "blur", blur_rect=(200, 100, 1300, 900))
    blank, _ = make_slide(tmp_path, "blank", tumor_rect=None)

    m_clean, d_clean = _qc(clean, cfg)
    m_pen, d_pen = _qc(pen, cfg)
    m_blur, d_blur = _qc(blur, cfg)
    assert d_clean.decision == "use" and d_clean.reasons == ()
    assert m_pen.pen_fraction > m_clean.pen_fraction and d_pen.decision == "exclude"
    assert any("pen_fraction" in r for r in d_pen.reasons)
    assert m_blur.blur_fraction > m_clean.blur_fraction
    assert m_blur.laplacian_var_median < m_clean.laplacian_var_median

    spec_blank = SyntheticSlideSpec(
        width=512, height=384, tissue_radius=(0.0, 0.0), tumor_rect=None
    )
    from tests.histo_synthetic_fixture import write_synthetic_slide

    write_synthetic_slide(tmp_path / "glass.tif", spec_blank)
    m_glass, d_glass = _qc(tmp_path / "glass.tif", cfg)
    assert m_glass.tissue_fraction < 0.02 and d_glass.decision == "exclude"
    assert blank.exists()


def test_qc_row_is_flat_and_serialisable(tmp_path: Path) -> None:
    cfg = QcConfig(thumbnail_px=512, analysis_mpp=1.0, tile_px=64, min_tissue_tiles=4)
    path, _ = make_slide(tmp_path, "clean")
    metrics, decision = _qc(path, cfg)
    row = qc_row(metrics, decision)
    assert row["slide_id"] == "clean" and row["decision"] == "use" and row["reasons"] == ""
    assert all(not isinstance(v, (list, dict, tuple)) for v in row.values())
