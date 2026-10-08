from pathlib import Path

import pytest

pytest.importorskip("torch")

from onkos.common.config import load_config  # noqa: E402
from onkos.common.reproducibility import set_seed  # noqa: E402
from onkos.common.smoke import run_smoke  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(tmp_path: Path, seed: int = 42):  # type: ignore[no-untyped-def]
    return load_config(CONFIGS, [f"seed={seed}", f"train.checkpoint_dir={tmp_path}/ckpt"])


def test_same_seed_gives_identical_first_100_losses(tmp_path: Path) -> None:
    a = run_smoke(_cfg(tmp_path / "a"))
    b = run_smoke(_cfg(tmp_path / "b"))
    assert len(a) == 100
    assert a == b


def test_different_seed_changes_losses(tmp_path: Path) -> None:
    assert run_smoke(_cfg(tmp_path / "a", 1)) != run_smoke(_cfg(tmp_path / "b", 2))


def test_same_seed_with_workers(tmp_path: Path) -> None:
    cfg = load_config(
        CONFIGS, ["data.num_workers=2", f"train.checkpoint_dir={tmp_path}/c1", "train.steps=20"]
    )
    cfg2 = load_config(
        CONFIGS, ["data.num_workers=2", f"train.checkpoint_dir={tmp_path}/c2", "train.steps=20"]
    )
    assert run_smoke(cfg) == run_smoke(cfg2)


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)
