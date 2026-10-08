from pathlib import Path

import pytest

pytest.importorskip("mlflow")

from onkos.common.config import load_config  # noqa: E402
from onkos.common.tracking import build_run_name, flatten_config  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_run_name_standard() -> None:
    assert build_run_name("omics", "baseline", 42) == "omics-baseline-42"


def test_run_name_rejects_bad_characters() -> None:
    with pytest.raises(ValueError):
        build_run_name("omics", "base line", 1)


def test_flatten_config() -> None:
    flat = flatten_config(load_config(CONFIGS))
    assert flat["train.lr"] == "0.05"
    assert flat["seed"] == "42"
