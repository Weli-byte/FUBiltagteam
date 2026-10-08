"""HDF5 contract for Nisa's patch-feature files (module 1 -> modules 2, 4, 5).

Layout (``features``/``coords`` required; ``attention``/``slide_embedding`` optional)::

    /features         (N, D) float16   patch feature vectors from the ViT encoder
    /coords           (N, 2) int32     patch top-left (x, y) in level-0 pixels
    /attention        (N,)   float32   ABMIL attention weights, >= 0, sum ~= 1
    /slide_embedding  (D,)   float32   slide-level embedding
    attrs: schema_version, slide_id, magnification (float), encoder_name, encoder_version,
           patch_size_px (int)

``h5py`` and ``numpy`` are imported lazily (they live in the ``train`` dependency group).
"""

import math
from pathlib import Path
from typing import Any

from pydantic import Field

from onkos.contracts import SCHEMA_VERSION
from onkos.contracts.schemas import ContractModel, SlideId

FEATURES_DTYPE = "float16"
COORDS_DTYPE = "int32"


class PatchFeatureMeta(ContractModel):
    """Summary returned by :func:`validate_patch_feature_file`."""

    slide_id: SlideId
    n_patches: int = Field(gt=0, description="Patch sayısı N.")
    feature_dim: int = Field(gt=0, description="Özellik boyutu D.")
    magnification: float = Field(gt=0, description="Büyütme (örn. 20.0 veya 40.0).")
    encoder_name: str = Field(min_length=1)
    encoder_version: str = Field(min_length=1)
    patch_size_px: int = Field(gt=0)
    has_attention: bool
    has_slide_embedding: bool


def _attr(attrs: Any, key: str) -> Any:
    if key not in attrs:
        raise ValueError(f"HDF5 attribute '{key}' eksik")
    value = attrs[key]
    return value.decode() if isinstance(value, bytes) else value


def validate_patch_feature_file(path: str | Path) -> PatchFeatureMeta:
    """Check a feature file against the contract; raise ``ValueError`` on any violation."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        for name in ("features", "coords"):
            if name not in f:
                raise ValueError(f"HDF5 dataset '{name}' eksik")
        features, coords = f["features"], f["coords"]
        if features.ndim != 2 or features.dtype != np.dtype(FEATURES_DTYPE):
            raise ValueError(f"features (N, D) {FEATURES_DTYPE} olmalı; bulundu {features.shape}")
        n_patches, dim = features.shape
        if n_patches == 0:
            raise ValueError("features boş")
        if coords.shape != (n_patches, 2) or coords.dtype != np.dtype(COORDS_DTYPE):
            raise ValueError(
                f"coords ({n_patches}, 2) {COORDS_DTYPE} olmalı; bulundu {coords.shape}"
            )
        if (coords[...] < 0).any():
            raise ValueError("coords negatif olamaz")
        if not np.isfinite(features[...].astype("float32")).all():
            raise ValueError("features NaN/Inf içeriyor")
        has_attention = "attention" in f
        if has_attention:
            att = f["attention"][...]
            if (
                att.shape != (n_patches,)
                or (att < 0).any()
                or not math.isclose(float(att.sum()), 1.0, abs_tol=1e-3)
            ):
                raise ValueError("attention (N,) olmalı, >= 0 ve toplamı ~1 olmalı")
        has_embedding = "slide_embedding" in f
        if has_embedding and f["slide_embedding"].shape != (dim,):
            raise ValueError(f"slide_embedding ({dim},) olmalı")
        version = str(_attr(f.attrs, "schema_version"))
        if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"uyumsuz schema_version {version}")
        return PatchFeatureMeta(
            slide_id=str(_attr(f.attrs, "slide_id")),
            n_patches=int(n_patches),
            feature_dim=int(dim),
            magnification=float(_attr(f.attrs, "magnification")),
            encoder_name=str(_attr(f.attrs, "encoder_name")),
            encoder_version=str(_attr(f.attrs, "encoder_version")),
            patch_size_px=int(_attr(f.attrs, "patch_size_px")),
            has_attention=has_attention,
            has_slide_embedding=has_embedding,
        )


def write_mock_patch_features(
    path: str | Path,
    slide_id: str = "slide_000001",
    n_patches: int = 64,
    feature_dim: int = 128,
    seed: int = 0,
    with_attention: bool = True,
    with_slide_embedding: bool = True,
) -> Path:
    """Write a random but contract-valid feature file (for developing modules 2/4/5 early)."""
    import h5py
    import numpy as np

    rng = np.random.default_rng(seed)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset(
            "features", data=rng.standard_normal((n_patches, feature_dim)).astype(FEATURES_DTYPE)
        )
        grid = int(math.ceil(math.sqrt(n_patches)))
        xs = (np.arange(n_patches) % grid) * 256
        ys = (np.arange(n_patches) // grid) * 256
        f.create_dataset("coords", data=np.stack([xs, ys], axis=1).astype(COORDS_DTYPE))
        if with_attention:
            att = rng.random(n_patches).astype("float32")
            f.create_dataset("attention", data=att / att.sum())
        if with_slide_embedding:
            f.create_dataset(
                "slide_embedding", data=rng.standard_normal(feature_dim).astype("float32")
            )
        f.attrs.update(
            schema_version=SCHEMA_VERSION,
            slide_id=slide_id,
            magnification=20.0,
            encoder_name="mock-vit",
            encoder_version="0.0.0-mock",
            patch_size_px=256,
        )
    return out
