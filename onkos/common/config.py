"""YAML configuration loading (OmegaConf) with group files and dotlist overrides."""

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(config_dir: Path | str = "configs", overrides: Sequence[str] = ()) -> DictConfig:
    """Load ``<config_dir>/config.yaml`` plus each group in its ``defaults`` list.

    Args:
        config_dir: Directory containing ``config.yaml`` and one folder per config group.
        overrides: Dotlist overrides such as ``["train.lr=0.1", "seed=7"]``.

    Returns:
        A resolved, merged configuration (``train.lr`` etc. are plain attributes).
    """
    root = Path(config_dir)
    base = OmegaConf.load(root / "config.yaml")
    if not isinstance(base, DictConfig):
        raise ValueError(f"{root / 'config.yaml'} must contain a mapping at top level")
    groups = list(base.pop("defaults", []))
    parts: list[DictConfig | ListConfig] = [base]
    for group in groups:
        parts.append(OmegaConf.create({group: OmegaConf.load(root / group / "default.yaml")}))
    parts.append(OmegaConf.from_dotlist(list(overrides)))
    merged = OmegaConf.merge(*parts)
    if not isinstance(merged, DictConfig):
        raise ValueError("merged configuration must be a mapping")
    OmegaConf.resolve(merged)
    return merged
