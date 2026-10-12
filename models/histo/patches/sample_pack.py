"""patch_v0: the small, real-data sample pack the whole team waits for (mini sprint 1.3).

Bu dosya ne işe yarar? Az sayıda slayttan (30-50) OpenSlide ile okunan, doku filtresinden geçmiş
256x256 patch'leri PNG olarak kaydeder; her patch için slayt kimliği, seviye-0 koordinatı,
etiket (tümör/normal), bölme (train/test) ve kaynak bilgisini ``patches.csv``'ye yazar. CAMELYON
slaytlarında etiket tümör maskesinden (patch'in >= %50'si tümör ise tümör, hiç tümör yoksa
normal, arası belirsiz ve atlanır), TCGA slaytlarında yalnızca slayt düzeyinde ("birincil tümör
slaydı") verilir. Doğrulama için her slaytın küçük resmine patch kutuları çizilir.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from models.histo.camelyon.annotations import (
    label_patches,
    overlay_mask,
    parse_asap_xml,
    patch_tumor_fractions,
    polygons_to_mask,
)
from models.histo.patches.grid import PatchGridConfig, draw_boxes, tissue_patch_grid
from models.histo.wsi.reader import SlideReader
from models.histo.wsi.tissue import TissueMaskConfig

log = logging.getLogger(__name__)

CSV_COLUMNS = (
    "image",
    "slide_id",
    "source",
    "split",
    "label",
    "label_name",
    "label_method",
    "x",
    "y",
    "patch_level0_px",
    "patch_px",
    "target_mpp",
    "slide_mpp",
    "tissue_fraction",
    "tumor_fraction",
)
LABEL_NAMES = {0: "normal", 1: "tumor"}


@dataclass(frozen=True)
class SamplePackConfig:
    patches_per_slide: int = 200
    seed: int = 42
    tumor_fraction_min: float = 0.5  # patch is "tumor" at/above this
    normal_fraction_max: float = 0.0  # patch is "normal" at/below this (between = skipped)
    overlay: bool = True
    overlay_max_boxes: int = 300

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PackSlide:
    """One slide to sample. ``slide_label``: 'tumor'/'normal' when only slide-level truth exists."""

    slide_id: str
    path: Path
    source: str  # camelyon16 | tcga
    split: str  # train | test | unassigned
    slide_label: str | None = None
    annotation_xml: Path | None = None
    mpp_override: float | None = None


def _balanced_subsample(
    labels: np.ndarray[Any, Any], n_total: int, rng: np.random.Generator
) -> np.ndarray[Any, Any]:
    """Indices with up to n_total/2 tumor and n_total/2 normal patches (ambiguous = -1 dropped)."""
    tumor = np.flatnonzero(labels == 1)
    normal = np.flatnonzero(labels == 0)
    half = n_total // 2
    take_t = min(len(tumor), half)
    take_n = min(len(normal), n_total - take_t)
    take_t = min(len(tumor), n_total - take_n)  # refill if one class is short
    chosen = np.concatenate(
        [
            rng.choice(tumor, size=take_t, replace=False) if take_t else np.zeros(0, int),
            rng.choice(normal, size=take_n, replace=False) if take_n else np.zeros(0, int),
        ]
    )
    return np.sort(chosen.astype(int))


def sample_slide(
    slide: PackSlide,
    out_dir: Path,
    cfg: SamplePackConfig,
    grid_cfg: PatchGridConfig,
    mask_cfg: TissueMaskConfig,
) -> list[dict[str, Any]]:
    """Extract patches of one slide to ``out_dir/images/<slide_id>/`` and return CSV rows."""
    import cv2

    rng = np.random.default_rng(cfg.seed + (hash(slide.slide_id) % 100_000))
    rows: list[dict[str, Any]] = []
    with SlideReader(slide.path, mpp_override=slide.mpp_override) as reader:
        grid = tissue_patch_grid(reader, grid_cfg, mask_cfg)
        if len(grid.coords) == 0:
            log.warning("%s: no tissue patches", slide.slide_id)
            return rows
        tumor_fraction = np.zeros(len(grid.coords), dtype=np.float32)
        if slide.annotation_xml is not None:
            polygons = parse_asap_xml(slide.annotation_xml)
            mask = polygons_to_mask(
                polygons, reader.info.width, reader.info.height, grid.thumbnail_downsample
            )
            tumor_fraction = patch_tumor_fractions(
                mask, grid.thumbnail_downsample, grid.coords, grid.size0
            )
            labels = label_patches(tumor_fraction, cfg.tumor_fraction_min, cfg.normal_fraction_max)
            label_method = "camelyon_xml_mask"
        elif slide.slide_label is not None:
            labels = np.full(len(grid.coords), 1 if slide.slide_label == "tumor" else 0, np.int8)
            label_method = "slide_level"
        else:
            labels = np.full(len(grid.coords), -1, np.int8)
            label_method = "none"
        pick = (
            _balanced_subsample(labels, cfg.patches_per_slide, rng)
            if label_method != "none"
            else np.sort(
                rng.choice(
                    len(grid.coords),
                    size=min(cfg.patches_per_slide, len(grid.coords)),
                    replace=False,
                )
            )
        )
        image_dir = out_dir / "images" / slide.slide_id
        image_dir.mkdir(parents=True, exist_ok=True)
        for idx in pick:
            x, y = (int(v) for v in grid.coords[idx])
            patch = reader.read_patch(x, y, grid_cfg.patch_px, grid_cfg.target_mpp)
            rel = Path("images") / slide.slide_id / f"{x}_{y}.png"
            cv2.imwrite(str(out_dir / rel), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            label = int(labels[idx])
            rows.append(
                {
                    "image": rel.as_posix(),
                    "slide_id": slide.slide_id,
                    "source": slide.source,
                    "split": slide.split,
                    "label": label,
                    "label_name": LABEL_NAMES.get(label, "unknown"),
                    "label_method": label_method,
                    "x": x,
                    "y": y,
                    "patch_level0_px": grid.size0,
                    "patch_px": grid_cfg.patch_px,
                    "target_mpp": grid_cfg.target_mpp,
                    "slide_mpp": reader.mpp,
                    "tissue_fraction": float(grid.tissue_fraction[idx]),
                    "tumor_fraction": float(tumor_fraction[idx]),
                }
            )
        if cfg.overlay:
            qc_dir = out_dir / "overlays"
            qc_dir.mkdir(parents=True, exist_ok=True)
            thumb = grid.thumbnail
            if slide.annotation_xml is not None:
                thumb = overlay_mask(thumb, mask)
            boxes = grid.coords[pick[: cfg.overlay_max_boxes]]
            overlay = draw_boxes(thumb, grid.thumbnail_downsample, boxes, grid.size0)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(qc_dir / f"{slide.slide_id}.png"), overlay_bgr)
    return rows


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_sample_pack(
    slides: list[PackSlide],
    out_dir: Path,
    cfg: SamplePackConfig,
    grid_cfg: PatchGridConfig,
    mask_cfg: TissueMaskConfig,
) -> list[dict[str, Any]]:
    """Run :func:`sample_slide` for every slide, write ``patches.csv`` and ``pack_meta.json``."""
    if len({s.slide_id for s in slides}) != len(slides):
        raise ValueError("slide_id values must be unique")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for slide in slides:
        try:
            rows += sample_slide(slide, out_dir, cfg, grid_cfg, mask_cfg)
        except Exception as exc:  # keep going; report at the end
            log.exception("%s failed", slide.slide_id)
            failed.append({"slide_id": slide.slide_id, "error": repr(exc)})
    write_rows(rows, out_dir / "patches.csv")
    counts = {
        "patches": len(rows),
        "slides": len({r["slide_id"] for r in rows}),
        "tumor": sum(r["label"] == 1 for r in rows),
        "normal": sum(r["label"] == 0 for r in rows),
        "unlabelled": sum(r["label"] == -1 for r in rows),
        "by_split": {
            s: sum(r["split"] == s for r in rows) for s in sorted({r["split"] for r in rows})
        },
        "by_source": {
            s: sum(r["source"] == s for r in rows) for s in sorted({r["source"] for r in rows})
        },
    }
    meta = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "grid": grid_cfg.to_dict(),
        "tissue_mask": mask_cfg.to_dict(),
        "counts": counts,
        "failed": failed,
        "coordinate_system": "x, y = level-0 pixels of the patch top-left corner",
    }
    (out_dir / "pack_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return rows
