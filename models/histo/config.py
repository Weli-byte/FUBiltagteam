"""Configuration loading for the histo module (YAML + dotlist overrides).

Bu dosya ne işe yarar? Patch boyutu, büyütme, eşikler gibi tüm sayısal ayarları
``configs/histo/*.yaml`` dosyalarından okur; kod içinde "sihirli sayı" kalmaz. Komut satırından
``anahtar=değer`` biçiminde geçici değişiklik (override) yapılabilir.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

HISTO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "histo"


def load_histo_config(name_or_path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Load ``configs/histo/<name>.yaml`` (or an explicit path) and apply dotlist overrides.

    Args:
        name_or_path: Config name without extension (``"patch_extract"``) or a path to a YAML file.
        overrides: Dotlist overrides such as ``["grid.patch_px=512", "seed=7"]``.
    """
    path = Path(name_or_path)
    if path.suffix != ".yaml":
        path = HISTO_CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"histo config not found: {path}")
    base = OmegaConf.load(path)
    if not isinstance(base, DictConfig):
        raise ValueError(f"{path} must contain a mapping at top level")
    merged = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    if not isinstance(merged, DictConfig):
        raise ValueError("merged configuration must be a mapping")
    OmegaConf.resolve(merged)
    return merged


def to_plain(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Convert a config (OmegaConf or mapping) to a plain JSON-serialisable dict."""
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            raise ValueError("config must be a mapping")
        return {str(k): v for k, v in container.items()}
    return dict(cfg)


def section(cfg: DictConfig, name: str) -> dict[str, Any]:
    """Plain dict of one top-level config section (``cfg.grid`` -> ``{"patch_px": ...}``)."""
    if name not in cfg:
        raise KeyError(f"config section {name!r} missing")
    value = cfg[name]
    if not isinstance(value, DictConfig):
        raise ValueError(f"config section {name!r} must be a mapping")
    return to_plain(value)


def config_hash(cfg: DictConfig | Mapping[str, Any], length: int = 16) -> str:
    """Short, order-independent SHA-256 of a config; stored in output files for provenance."""
    payload = json.dumps(to_plain(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
