from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("h5py")
np = pytest.importorskip("numpy")
import h5py  # noqa: E402

from onkos.contracts.patch_features import validate_patch_feature_file  # noqa: E402
from tests.contract_fixtures import write_test_patch_features  # noqa: E402


def test_fixture_file_satisfies_contract(tmp_path: Path) -> None:
    path = write_test_patch_features(
        tmp_path / "s.h5", slide_id="slide_000009", n_patches=50, feature_dim=32
    )
    meta = validate_patch_feature_file(path)
    assert (meta.n_patches, meta.feature_dim) == (50, 32)
    assert meta.has_attention and meta.has_slide_embedding and meta.slide_id == "slide_000009"


def test_optional_datasets_may_be_absent(tmp_path: Path) -> None:
    path = write_test_patch_features(
        tmp_path / "s.h5", with_attention=False, with_slide_embedding=False
    )
    meta = validate_patch_feature_file(path)
    assert not meta.has_attention and not meta.has_slide_embedding


def _rewrite(path: Path, name: str, data: Any) -> None:
    with h5py.File(path, "r+") as f:
        del f[name]
        f.create_dataset(name, data=data)


@pytest.mark.parametrize(
    ("dataset", "bad", "message"),
    [
        ("features", np.zeros((10, 8), dtype="float32"), "float16"),
        ("coords", np.zeros((10, 2), dtype="int64"), "int32"),
        ("coords", -np.ones((64, 2), dtype="int32"), "negatif"),
        ("attention", np.ones(64, dtype="float32"), "toplamı"),
    ],
)
def test_contract_violations_are_reported(
    tmp_path: Path, dataset: str, bad: Any, message: str
) -> None:
    path = write_test_patch_features(tmp_path / "s.h5", n_patches=64, feature_dim=16)
    _rewrite(path, dataset, bad)
    with pytest.raises(ValueError, match=message):
        validate_patch_feature_file(path)


def test_missing_attribute_is_reported(tmp_path: Path) -> None:
    path = write_test_patch_features(tmp_path / "s.h5")
    with h5py.File(path, "r+") as f:
        del f.attrs["encoder_version"]
    with pytest.raises(ValueError, match="encoder_version"):
        validate_patch_feature_file(path)


def test_nan_features_rejected(tmp_path: Path) -> None:
    path = write_test_patch_features(tmp_path / "s.h5", n_patches=8, feature_dim=4)
    bad = np.full((8, 4), np.nan, dtype="float16")
    _rewrite(path, "features", bad)
    with pytest.raises(ValueError, match="NaN"):
        validate_patch_feature_file(path)
