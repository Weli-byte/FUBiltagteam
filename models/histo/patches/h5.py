"""HDF5 coordinate files: the precursor of the patch-feature contract (mini sprint 1.7).

Bu dosya ne işe yarar? Her slayt için seçilen patch koordinatlarını tek bir HDF5 dosyasına yazar
ve okur. Patch'lerin kendisi diske yazılmaz (milyonlarca küçük PNG yerine), ihtiyaç anında
OpenSlide'dan okunur. Dosya, proje liderinin sözleşmesiyle (``onkos.contracts.patch_features``)
aynı öznitelikleri (``slide_id``, ``magnification``, ``patch_size_px`` ...) taşır; Sprint 2'de
``features`` veri kümesi eklendiğinde dosya sözleşmeye tam uyumlu hale gelir.

Düzen::

    /coords           (N, 2) int32    patch sol-üst (x, y), seviye-0 pikseli
    /tissue_fraction  (N,)   float32  doku oranı (0-1)
    attrs: schema_version, slide_id, magnification, target_mpp, slide_mpp, patch_size_px,
           patch_level0_px, stride_level0_px, slide_width, slide_height, level_count,
           min_tissue_fraction, tissue_mask_version, config_hash, pipeline_version, created_at

``magnification`` is the **extraction** magnification (derived from ``target_mpp``: 0.5 -> 20x),
not the scanner's native one (that is implied by ``slide_mpp``). The source file path is kept
out of the file on purpose: TCGA file names carry the patient barcode, and these files are
shared with other modules (contract rule: no patient identifier in shared artefacts).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from onkos.contracts import SCHEMA_VERSION
from onkos.contracts.patch_features import COORDS_DTYPE

PIPELINE_VERSION = "patch-extract-1.0"


@dataclass(frozen=True)
class CoordsAttrs:
    slide_id: str
    magnification: float
    target_mpp: float
    slide_mpp: float
    patch_size_px: int
    patch_level0_px: int
    stride_level0_px: int
    slide_width: int
    slide_height: int
    level_count: int
    min_tissue_fraction: float
    tissue_mask_version: str
    config_hash: str
    created_at: str
    pipeline_version: str = PIPELINE_VERSION
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_attrs(cls, attrs: Any) -> CoordsAttrs:
        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in attrs:
                raise ValueError(f"HDF5 attribute '{f.name}' eksik")
            raw = attrs[f.name]
            value = raw.decode() if isinstance(raw, bytes) else raw
            values[f.name] = f.type_cast(value) if hasattr(f, "type_cast") else value
        return cls(
            slide_id=str(values["slide_id"]),
            magnification=float(values["magnification"]),
            target_mpp=float(values["target_mpp"]),
            slide_mpp=float(values["slide_mpp"]),
            patch_size_px=int(values["patch_size_px"]),
            patch_level0_px=int(values["patch_level0_px"]),
            stride_level0_px=int(values["stride_level0_px"]),
            slide_width=int(values["slide_width"]),
            slide_height=int(values["slide_height"]),
            level_count=int(values["level_count"]),
            min_tissue_fraction=float(values["min_tissue_fraction"]),
            tissue_mask_version=str(values["tissue_mask_version"]),
            config_hash=str(values["config_hash"]),
            created_at=str(values["created_at"]),
            pipeline_version=str(values["pipeline_version"]),
            schema_version=str(values["schema_version"]),
        )


def write_coords_h5(
    path: Path,
    coords: NDArray[np.int32],
    tissue_fraction: NDArray[np.float32],
    attrs: CoordsAttrs,
) -> Path:
    """Write atomically (``.tmp`` then rename) so an interrupted run never leaves a half file."""
    import h5py

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")
    if len(tissue_fraction) != len(coords):
        raise ValueError("tissue_fraction must have one value per patch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("coords", data=coords.astype(COORDS_DTYPE), compression="gzip")
        f.create_dataset(
            "tissue_fraction", data=tissue_fraction.astype("float32"), compression="gzip"
        )
        for key, value in attrs.to_dict().items():
            f.attrs[key] = value
    os.replace(tmp, path)
    return path


def read_coords_h5(path: Path) -> tuple[NDArray[np.int32], NDArray[np.float32], CoordsAttrs]:
    import h5py

    with h5py.File(path, "r") as f:
        coords = np.asarray(f["coords"][...], dtype=np.int32)
        fractions = np.asarray(f["tissue_fraction"][...], dtype=np.float32)
        attrs = CoordsAttrs.from_attrs(f.attrs)
    return coords, fractions, attrs


def validate_coords_h5(path: Path) -> CoordsAttrs:
    """Check dtype/shape/bounds; raise ``ValueError`` on any violation (used for resume checks)."""
    coords, fractions, attrs = read_coords_h5(path)
    if coords.dtype != np.dtype(COORDS_DTYPE):
        raise ValueError(f"coords {COORDS_DTYPE} olmalı")
    if len(coords) == 0:
        raise ValueError("coords boş")
    if (coords < 0).any():
        raise ValueError("coords negatif olamaz")
    if (coords[:, 0] + attrs.patch_level0_px > attrs.slide_width).any() or (
        coords[:, 1] + attrs.patch_level0_px > attrs.slide_height
    ).any():
        raise ValueError("patch slayt sınırının dışına taşıyor")
    if ((fractions < 0) | (fractions > 1)).any():
        raise ValueError("tissue_fraction 0-1 aralığında olmalı")
    if attrs.schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(f"uyumsuz schema_version {attrs.schema_version}")
    return attrs
