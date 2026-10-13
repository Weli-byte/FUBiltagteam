"""Slide reader: pyramid levels, mpp handling, level selection and patch reads (mini sprint 1.3)."""

from pathlib import Path

import pytest

pytest.importorskip("openslide")
pytest.importorskip("cv2")
pytest.importorskip("tifffile")
np = pytest.importorskip("numpy")

from models.histo.wsi.reader import SlideReader, magnification_from_mpp  # noqa: E402
from tests.histo_fixtures import make_slide  # noqa: E402


def test_magnification_from_mpp_matches_conventions() -> None:
    assert magnification_from_mpp(0.25) == 40.0
    assert magnification_from_mpp(0.252) == 40.0
    assert magnification_from_mpp(0.5) == 20.0
    assert magnification_from_mpp(1.0) == 10.0
    assert magnification_from_mpp(0.7) is None  # too far from every convention
    with pytest.raises(ValueError):
        magnification_from_mpp(0.0)


def test_pyramid_metadata_is_read(tmp_path: Path) -> None:
    path, spec = make_slide(tmp_path, mpp=0.5, levels=4)
    with SlideReader(path) as reader:
        info = reader.info
        assert (info.width, info.height) == (spec.width, spec.height)
        assert info.level_count == 4
        assert info.level_downsamples == (1.0, 2.0, 4.0, 8.0)
        assert info.mpp == pytest.approx(0.5, rel=1e-3)
        assert info.magnification == 20.0
        assert info.vendor == "generic-tiff"


def test_level_selection_never_upsamples(tmp_path: Path) -> None:
    path, _ = make_slide(tmp_path, mpp=0.5)
    with SlideReader(path) as reader:
        for target, level, rescale in ((0.5, 0, 1.0), (1.0, 1, 1.0), (0.75, 0, 1.5), (2.0, 2, 1.0)):
            got_level, got_rescale = reader.level_for_mpp(target)
            assert got_level == level and got_rescale == pytest.approx(rescale)
        with pytest.raises(ValueError):
            reader.level_for_mpp(0.25)  # finer than the scan


def test_patch_size_scales_with_scan_resolution(tmp_path: Path) -> None:
    path20, _ = make_slide(tmp_path, "s20", mpp=0.5)
    path40, _ = make_slide(tmp_path, "s40", mpp=0.25)
    with SlideReader(path20) as r20, SlideReader(path40) as r40:
        assert r20.level0_box_size(256, 0.5) == 256
        assert r40.level0_box_size(256, 0.5) == 512  # 40x scan: twice the level-0 pixels
        patch = r40.read_patch(100, 100, 256, 0.5)
        assert patch.shape == (256, 256, 3) and patch.dtype == np.uint8


def test_read_patch_matches_direct_region(tmp_path: Path) -> None:
    path, _ = make_slide(tmp_path, mpp=0.5)
    with SlideReader(path) as reader:
        direct = reader.read_region(1000, 400, 0, (256, 256))
        patch = reader.read_patch(1000, 400, 256, 0.5)
        assert np.array_equal(direct, patch)
        tumour_mean = patch.reshape(-1, 3).mean(axis=0)
        assert tumour_mean[0] < 215  # inside the purple tumor rectangle, not white glass


def test_thumbnail_downsample_is_consistent(tmp_path: Path) -> None:
    path, spec = make_slide(tmp_path)
    with SlideReader(path) as reader:
        thumb, downsample = reader.thumbnail(512)
        assert thumb.shape[1] == 512 and thumb.dtype == np.uint8
        assert downsample == pytest.approx(spec.width / 512, rel=1e-6)


def test_plain_image_needs_mpp_override(tmp_path: Path) -> None:
    import cv2

    png = tmp_path / "plain.png"
    cv2.imwrite(str(png), np.full((300, 400, 3), 200, dtype=np.uint8))
    with SlideReader(png) as reader:
        assert reader.info.level_count == 1 and reader.info.mpp is None
        with pytest.raises(ValueError, match="mpp"):
            _ = reader.mpp
    with SlideReader(png, mpp_override=0.5) as reader:
        assert reader.mpp == 0.5 and reader.info.magnification == 20.0


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SlideReader(tmp_path / "nope.svs")
