"""Production patch-coordinate extraction: parallel, resumable, fault tolerant (mini sprint 1.7).

Bu dosya ne işe yarar? Bir slayt listesini alır, her slayt için doku maskesi + ızgara üretip
koordinatları HDF5'e yazar (CLAM tarzı: patch görüntüleri değil, yalnızca koordinatlar).
Birden çok slaytı aynı anda işler (her işlem slaytı kendisi açar; OpenSlide nesnesi işlemler
arasında paylaşılmaz), tamamlanmış slaytları atlar (kesintiden devam), bozuk slaytta çökmek yerine
hatayı kaydedip devam eder ve sonunda özet tablo yazar.
"""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from models.histo.config import config_hash
from models.histo.patches.grid import PatchGridConfig, tissue_patch_grid
from models.histo.patches.h5 import CoordsAttrs, validate_coords_h5, write_coords_h5
from models.histo.wsi.reader import SlideReader, magnification_from_mpp
from models.histo.wsi.tissue import TISSUE_MASK_VERSION, TissueMaskConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlideSpec:
    """One slide to process. ``slide_id`` must be anonymous (never a TCGA barcode)."""

    slide_id: str
    path: Path
    mpp_override: float | None = None


@dataclass(frozen=True)
class ExtractResult:
    slide_id: str
    status: str  # done | skipped | failed
    n_patches: int
    n_candidates: int
    seconds: float
    out_path: str | None
    error: str | None = None
    source_path: str | None = None  # local provenance only; never written into the HDF5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_slide_list(path: Path) -> list[SlideSpec]:
    """CSV with columns ``slide_id,path[,mpp_override]``; relative paths resolve to the CSV dir."""
    specs: list[SlideSpec] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("mpp_override") or ""
            slide_path = Path(row["path"])
            if not slide_path.is_absolute():
                slide_path = path.parent / slide_path
            specs.append(
                SlideSpec(
                    slide_id=row["slide_id"].strip(),
                    path=slide_path,
                    mpp_override=float(raw) if raw.strip() else None,
                )
            )
    if len({s.slide_id for s in specs}) != len(specs):
        raise ValueError("slide_id values must be unique")
    return specs


def _output_path(out_dir: Path, slide_id: str) -> Path:
    return out_dir / f"{slide_id}.h5"


def _already_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        validate_coords_h5(path)
    except (OSError, ValueError, KeyError):
        log.warning("existing file %s is invalid; it will be regenerated", path)
        return False
    return True


def extract_one(
    spec: SlideSpec,
    out_dir: Path,
    grid_cfg: PatchGridConfig,
    mask_cfg: TissueMaskConfig,
    overwrite: bool = False,
) -> ExtractResult:
    """Process one slide to ``<out_dir>/<slide_id>.h5``. Never raises: failures are reported."""
    out_path = _output_path(out_dir, spec.slide_id)
    if not overwrite and _already_done(out_path):
        return ExtractResult(
            spec.slide_id, "skipped", 0, 0, 0.0, str(out_path), source_path=str(spec.path)
        )
    started = time.perf_counter()
    try:
        with SlideReader(spec.path, mpp_override=spec.mpp_override) as reader:
            grid = tissue_patch_grid(reader, grid_cfg, mask_cfg)
            if len(grid.coords) == 0:
                raise ValueError("no tissue patches found (empty slide or mask failure)")
            # Extraction magnification (what the patches are), not the scanner's native one.
            magnification = magnification_from_mpp(grid_cfg.target_mpp) or (
                0.25 / grid_cfg.target_mpp * 40.0
            )
            attrs = CoordsAttrs(
                slide_id=spec.slide_id,
                magnification=float(magnification),
                target_mpp=grid_cfg.target_mpp,
                slide_mpp=reader.mpp,
                patch_size_px=grid_cfg.patch_px,
                patch_level0_px=grid.size0,
                stride_level0_px=grid.stride0,
                slide_width=reader.info.width,
                slide_height=reader.info.height,
                level_count=reader.info.level_count,
                min_tissue_fraction=grid_cfg.min_tissue_fraction,
                tissue_mask_version=TISSUE_MASK_VERSION,
                config_hash=config_hash({"grid": grid_cfg.to_dict(), "mask": mask_cfg.to_dict()}),
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            write_coords_h5(out_path, grid.coords, grid.tissue_fraction, attrs)
        return ExtractResult(
            spec.slide_id,
            "done",
            int(len(grid.coords)),
            grid.n_candidates,
            time.perf_counter() - started,
            str(out_path),
            source_path=str(spec.path),
        )
    except Exception as exc:  # a broken slide must not stop the batch
        log.exception("slide %s failed", spec.slide_id)
        return ExtractResult(
            spec.slide_id,
            "failed",
            0,
            0,
            time.perf_counter() - started,
            None,
            repr(exc),
            source_path=str(spec.path),
        )


def run_extraction(
    specs: Sequence[SlideSpec],
    out_dir: Path,
    grid_cfg: PatchGridConfig,
    mask_cfg: TissueMaskConfig,
    workers: int = 1,
    overwrite: bool = False,
) -> list[ExtractResult]:
    """Process all slides; with ``workers > 1`` uses a process pool (spawn-safe)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExtractResult] = []
    if workers <= 1:
        for spec in specs:
            results.append(extract_one(spec, out_dir, grid_cfg, mask_cfg, overwrite))
            last = results[-1]
            log.info("%s: %s (%d patches)", spec.slide_id, last.status, last.n_patches)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_one, spec, out_dir, grid_cfg, mask_cfg, overwrite): spec
            for spec in specs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            log.info("%s: %s (%d patches)", result.slide_id, result.status, result.n_patches)
    return sorted(results, key=lambda r: r.slide_id)


def write_results(results: Iterable[ExtractResult], path: Path) -> None:
    rows = [r.to_dict() for r in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ExtractResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: Sequence[ExtractResult]) -> dict[str, Any]:
    done = [r for r in results if r.status == "done"]
    seconds = sum(r.seconds for r in done)
    return {
        "n_slides": len(results),
        "done": len(done),
        "skipped": sum(r.status == "skipped" for r in results),
        "failed": sum(r.status == "failed" for r in results),
        "patches_total": sum(r.n_patches for r in done),
        "seconds_per_slide": seconds / len(done) if done else 0.0,
        "slides_per_minute": 60.0 * len(done) / seconds if seconds > 0 else 0.0,
    }
