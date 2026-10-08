from pathlib import Path

import pytest

from onkos.common.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_load_config_merges_groups() -> None:
    cfg = load_config(CONFIGS)
    assert cfg.seed == 42
    assert cfg.train.steps == 100
    assert cfg.model.in_features == 8


def test_overrides_apply() -> None:
    cfg = load_config(CONFIGS, ["train.lr=0.5", "seed=7"])
    assert cfg.train.lr == 0.5
    assert cfg.seed == 7


def test_missing_group_file_fails(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("defaults: [nope]\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path)
