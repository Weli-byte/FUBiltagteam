"""Stain normalization, augmentation and TTA views (mini sprint 1.8)."""

from typing import Any

import pytest

pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from models.histo.stain.augment import (  # noqa: E402
    AugmentConfig,
    HistoAugment,
    dihedral_views,
    hed_to_rgb,
    rgb_to_hed,
    tta_mean,
)
from models.histo.stain.normalizer import (  # noqa: E402
    MacenkoNormalizer,
    ReinhardNormalizer,
    lab_stats,
    make_normalizer,
    od_to_rgb,
    rgb_to_od,
)
from tests.histo_synthetic_fixture import SyntheticSlideSpec, render_synthetic_rgb  # noqa: E402


def _he_like(seed: int, shift: tuple[int, int, int] = (0, 0, 0)) -> Any:
    """Synthetic H&E-like square patch; ``shift`` simulates a different staining protocol."""
    rgb, tissue = render_synthetic_rgb(
        SyntheticSlideSpec(width=320, height=256, tumor_rect=(120, 60, 260, 200), seed=seed)
    )
    patch = rgb[40:216, 60:236].astype(np.int16) + np.array(shift, dtype=np.int16)
    return np.clip(patch, 0, 255).astype(np.uint8)


def _he_mixture(seed: int, stain_rotation_deg: float = 0.0) -> Any:
    """A true two-stain Beer-Lambert mixture (H and E vectors from Ruifrok & Johnston).

    ``stain_rotation_deg`` rotates both stain vectors slightly, i.e. a different scanner/protocol.
    """
    from models.histo.stain.normalizer import RGB_FROM_HED

    rng = np.random.default_rng(seed)
    h, w = 160, 160
    yy, xx = np.mgrid[0:h, 0:w]
    nuclei = np.zeros((h, w))
    for _ in range(60):  # round 'nuclei' with strong hematoxylin
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(4, 9)
        nuclei[((yy - cy) ** 2 + (xx - cx) ** 2) < r * r] = 1.0
    c_h = 0.15 + 1.1 * nuclei + 0.05 * rng.random((h, w))
    c_e = 0.45 + 0.25 * np.sin(xx / 17.0) * np.cos(yy / 23.0) + 0.05 * rng.random((h, w))
    stains = RGB_FROM_HED[:2].copy()
    if stain_rotation_deg:
        theta = np.deg2rad(stain_rotation_deg)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        stains = stains @ rot
    od = c_h[..., None] * stains[0] + c_e[..., None] * stains[1]
    return od_to_rgb(od)


def test_od_roundtrip() -> None:
    rgb = _he_like(0)
    assert np.abs(od_to_rgb(rgb_to_od(rgb)).astype(int) - rgb.astype(int)).max() <= 1


def test_macenko_recovers_a_two_stain_mixture() -> None:
    target = _he_mixture(0)
    norm = MacenkoNormalizer().fit(target)
    same = norm.transform(target)
    assert same.shape == target.shape and same.dtype == np.uint8
    assert np.abs(same.astype(int) - target.astype(int)).mean() < 4.0  # near-identity
    source = _he_mixture(1, stain_rotation_deg=12.0)  # same tissue statistics, other 'scanner'
    before = np.abs(lab_stats(source)[:3] - lab_stats(target)[:3]).sum()
    after = np.abs(lab_stats(norm.transform(source))[:3] - lab_stats(target)[:3]).sum()
    assert after < 0.5 * before
    state = norm.state()
    assert state["method"] == "macenko" and len(state["stain_matrix_target"]) == 2


def test_macenko_reduces_colour_shift_on_rendered_patches() -> None:
    target = _he_like(0)
    source = _he_like(1, shift=(-25, 10, 20))
    norm = MacenkoNormalizer().fit(target)
    before = np.abs(lab_stats(source)[:3] - lab_stats(target)[:3]).sum()
    after = np.abs(lab_stats(norm.transform(source))[:3] - lab_stats(target)[:3]).sum()
    assert after < before


def test_macenko_rejects_images_without_tissue() -> None:
    with pytest.raises(ValueError, match="tissue"):
        MacenkoNormalizer().fit(np.full((64, 64, 3), 245, dtype=np.uint8))


def test_reinhard_matches_lab_statistics() -> None:
    target = _he_like(0)
    source = _he_like(1, shift=(-30, 15, 25))
    out = ReinhardNormalizer().fit(target).transform(source)
    assert out.dtype == np.uint8 and out.shape == source.shape
    assert np.abs(lab_stats(out) - lab_stats(target)).max() < 3.0
    assert make_normalizer("none").transform(source) is source
    with pytest.raises(ValueError):
        make_normalizer("vahadane")


def test_hed_roundtrip_and_augment_determinism() -> None:
    rgb = _he_like(2)
    assert np.abs(hed_to_rgb(rgb_to_hed(rgb)).astype(int) - rgb.astype(int)).max() <= 2
    cfg = AugmentConfig(blur_p=0.5, noise_p=0.5)
    a = HistoAugment(cfg, seed=7)(rgb)
    b = HistoAugment(cfg, seed=7)(rgb)
    c = HistoAugment(cfg, seed=8)(rgb)
    assert a.shape == rgb.shape and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    identity = AugmentConfig(
        rot90=False,
        flip_horizontal=False,
        flip_vertical=False,
        hed_sigma=0.0,
        brightness=0.0,
        contrast=0.0,
        blur_p=0.0,
        noise_p=0.0,
    )
    assert np.array_equal(HistoAugment(identity)(rgb), rgb)


def test_dihedral_views_are_the_eight_symmetries() -> None:
    rgb = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(4, 4, 3)
    views = dihedral_views(rgb)
    assert len(views) == 8 and all(v.shape == rgb.shape for v in views)
    assert len({v.tobytes() for v in views}) == 8
    assert np.array_equal(views[0], rgb)
    assert tta_mean(np.array([0.2, 0.4, 0.6])) == pytest.approx(0.4)
