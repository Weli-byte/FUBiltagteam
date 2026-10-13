"""Patch grid, HDF5 coordinate files, resumable extraction and datasets (mini sprint 1.3 / 1.7)."""

from pathlib import Path

import pytest

pytest.importorskip("openslide")
pytest.importorskip("cv2")
pytest.importorskip("tifffile")
pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

import h5py  # noqa: E402

from models.histo.patches.extract import (  # noqa: E402
    SlideSpec,
    extract_one,
    read_slide_list,
    run_extraction,
    summarize,
    write_results,
)
from models.histo.patches.grid import (  # noqa: E402
    PatchGridConfig,
    draw_boxes,
    grid_coords,
    tissue_patch_grid,
)
from models.histo.patches.h5 import read_coords_h5, validate_coords_h5  # noqa: E402
from models.histo.wsi.reader import SlideReader  # noqa: E402
from models.histo.wsi.tissue import TissueMaskConfig  # noqa: E402
from onkos.contracts import SCHEMA_VERSION  # noqa: E402
from onkos.contracts.patch_features import validate_patch_feature_file  # noqa: E402
from tests.histo_fixtures import make_slide  # noqa: E402

GRID = PatchGridConfig(patch_px=128, target_mpp=0.5, min_tissue_fraction=0.5, thumbnail_px=512)
MASK = TissueMaskConfig(min_object_px=20, max_hole_px=200)


def test_grid_coords_are_row_major_and_inside_bounds() -> None:
    coords = grid_coords(1000, 600, 256, 256)
    assert coords.dtype == np.int32 and coords.shape == (2 * 3, 2)
    assert coords[0].tolist() == [0, 0] and coords[1].tolist() == [256, 0]
    assert (coords[:, 0] + 256 <= 1000).all() and (coords[:, 1] + 256 <= 600).all()
    assert grid_coords(100, 100, 256, 256).shape == (0, 2)
    assert len(grid_coords(1000, 600, 256, 128)) > len(coords)  # overlap -> more boxes


def test_tissue_filter_keeps_only_tissue_boxes(tmp_path: Path) -> None:
    path, spec = make_slide(tmp_path)
    with SlideReader(path) as reader:
        grid = tissue_patch_grid(reader, GRID, MASK)
        assert 0 < len(grid.coords) < grid.n_candidates
        assert grid.coords.dtype == np.int32 and (grid.tissue_fraction >= 0.5).all()
        # every kept box centre lies inside the synthetic tissue ellipse
        cx, cy = spec.tissue_center[0] * spec.width, spec.tissue_center[1] * spec.height
        rx, ry = spec.tissue_radius[0] * spec.width, spec.tissue_radius[1] * spec.height
        centres = grid.coords + grid.size0 / 2
        inside = ((centres[:, 0] - cx) / rx) ** 2 + ((centres[:, 1] - cy) / ry) ** 2 <= 1.05
        assert inside.mean() > 0.95
        overlay = draw_boxes(grid.thumbnail, grid.thumbnail_downsample, grid.coords, grid.size0)
        assert overlay.shape == grid.thumbnail.shape


def test_same_physical_grid_for_20x_and_40x_scans(tmp_path: Path) -> None:
    p20, _ = make_slide(tmp_path, "s20", mpp=0.5, width=1536, height=1024)
    p40, _ = make_slide(tmp_path, "s40", mpp=0.25, width=3072, height=2048)
    with SlideReader(p20) as r20, SlideReader(p40) as r40:
        g20 = tissue_patch_grid(r20, GRID, MASK)
        g40 = tissue_patch_grid(r40, GRID, MASK)
    assert (g20.size0, g40.size0) == (128, 256)
    assert g20.n_candidates == g40.n_candidates
    assert abs(len(g20.coords) - len(g40.coords)) <= 0.1 * len(g20.coords)
    assert np.allclose(g40.coords[: len(g20.coords)] / 2.0, g20.coords[: len(g20.coords)], atol=2)


def test_extract_writes_validates_and_resumes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path, _ = make_slide(tmp_path, "slide_a")
    out = tmp_path / "coords"
    spec = SlideSpec("slide_a", path)
    first = extract_one(spec, out, GRID, MASK)
    assert first.status == "done" and first.n_patches > 0 and first.out_path
    attrs = validate_coords_h5(Path(first.out_path))
    assert attrs.slide_id == "slide_a" and attrs.patch_size_px == 128
    assert attrs.magnification == 20.0 and attrs.patch_level0_px == 128
    assert attrs.schema_version == SCHEMA_VERSION and len(attrs.config_hash) == 16
    coords, fractions, _ = read_coords_h5(Path(first.out_path))
    assert coords.dtype == np.int32 and fractions.dtype == np.float32

    second = extract_one(spec, out, GRID, MASK)
    assert second.status == "skipped"  # resume: nothing recomputed

    with h5py.File(first.out_path, "r+") as f:  # corrupt -> regenerated, not trusted
        del f.attrs["slide_id"]
    third = extract_one(spec, out, GRID, MASK)
    assert third.status == "done" and validate_coords_h5(Path(first.out_path)).slide_id == "slide_a"


def test_attrs_describe_extraction_scale_and_carry_no_source_path(tmp_path: Path) -> None:
    """A 40x scan extracted at 20x reports magnification 20 and 512-px level-0 boxes; the file
    never stores the slide path (TCGA file names carry the patient barcode)."""
    path, _ = make_slide(tmp_path, "TCGA-AA-0001-01Z-00-DX1.deadbeef", mpp=0.25)
    result = extract_one(SlideSpec("file-uuid-0001", path), tmp_path / "out", GRID, MASK)
    assert result.status == "done" and result.out_path and result.source_path == str(path)
    attrs = validate_coords_h5(Path(result.out_path))
    assert attrs.magnification == 20.0 and attrs.slide_mpp == pytest.approx(0.25, rel=1e-3)
    assert attrs.patch_size_px == 128 and attrs.patch_level0_px == 256
    with h5py.File(result.out_path, "r") as f:
        blob = " ".join(str(v) for v in f.attrs.values())
    assert "TCGA" not in blob and "deadbeef" not in blob


def test_broken_slide_is_reported_not_raised(tmp_path: Path) -> None:
    bad = tmp_path / "broken.svs"
    bad.write_bytes(b"not a slide")
    result = extract_one(SlideSpec("broken", bad), tmp_path / "out", GRID, MASK)
    assert result.status == "failed" and result.error and result.out_path is None
    glass = tmp_path / "glass.tif"
    from models.histo.wsi.synthetic import SyntheticSlideSpec, write_synthetic_slide

    write_synthetic_slide(glass, SyntheticSlideSpec(width=512, height=384, tissue_radius=(0, 0)))
    result = extract_one(SlideSpec("glass", glass), tmp_path / "out", GRID, MASK)
    assert result.status == "failed" and "no tissue" in (result.error or "")


def test_batch_run_with_workers_and_results_table(tmp_path: Path) -> None:
    specs = [SlideSpec(f"s{i}", make_slide(tmp_path, f"s{i}", seed=i)[0]) for i in range(2)]
    specs.append(SlideSpec("missing", tmp_path / "missing.tif"))
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text(
        "slide_id,path,mpp_override\n"
        + "\n".join(f"{s.slide_id},{s.path.name}," for s in specs)
        + "\n",
        encoding="utf-8",
    )
    listed = read_slide_list(csv_path)
    assert [s.slide_id for s in listed] == ["s0", "s1", "missing"]
    assert listed[0].path == specs[0].path and listed[0].mpp_override is None

    results = run_extraction(listed, tmp_path / "out", GRID, MASK, workers=2)
    status = {r.slide_id: r.status for r in results}
    assert status == {"s0": "done", "s1": "done", "missing": "failed"}
    summary = summarize(results)
    assert summary["done"] == 2 and summary["failed"] == 1 and summary["patches_total"] > 0
    write_results(results, tmp_path / "results.csv")
    assert (tmp_path / "results.csv").read_text(encoding="utf-8").count("\n") == 4


def test_coords_file_becomes_contract_file_once_features_are_added(tmp_path: Path) -> None:
    """Adding ``features`` to the coordinate file must satisfy the project leader's contract."""
    path, _ = make_slide(tmp_path, "slide_c")
    result = extract_one(SlideSpec("slide_c", path), tmp_path / "out", GRID, MASK)
    assert result.out_path
    with h5py.File(result.out_path, "r+") as f:
        n = f["coords"].shape[0]
        f.create_dataset("features", data=np.ones((n, 16), dtype="float16"))
        f.attrs["encoder_name"] = "test-encoder"
        f.attrs["encoder_version"] = "0"
    meta = validate_patch_feature_file(result.out_path)
    assert meta.slide_id == "slide_c" and meta.n_patches == n and meta.patch_size_px == 128


def test_patch_dataset_reads_the_same_pixels(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from models.histo.patches.dataset import PatchCoordsDataset

    path, _ = make_slide(tmp_path, "slide_d")
    result = extract_one(SlideSpec("slide_d", path), tmp_path / "out", GRID, MASK)
    assert result.out_path
    dataset = PatchCoordsDataset(path, Path(result.out_path), normalize=False)
    item = dataset[3]
    with SlideReader(path) as reader:
        expected = reader.read_patch(item["x"], item["y"], 128, 0.5)
    expected_tensor = torch.from_numpy(expected.copy()).permute(2, 0, 1).float() / 255
    assert torch.allclose(item["image"], expected_tensor)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch["image"].shape == (4, 3, 128, 128) and batch["slide_id"][0] == "slide_d"
