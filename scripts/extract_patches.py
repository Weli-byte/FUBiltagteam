"""Production patch-coordinate extraction over a slide list (mini sprint 1.7).

Bu dosya ne işe yarar? Tüm slaytlar için doku maskesi + ızgara üretir, koordinatları slayt
başına bir HDF5 dosyasına yazar; paralel çalışır, kesintiden devam eder (tamamlanmış ve geçerli
dosyalar atlanır), bozuk slaytı loglayıp geçer ve sonunda özet tablo + performans sayıları basar.

Usage:
    uv run --group histo python scripts/extract_patches.py     # configs/histo/patch_extract.yaml
    uv run --group histo python scripts/extract_patches.py run.workers=8 grid.patch_px=512
    uv run --group histo python scripts/extract_patches.py --slide-list my.csv --out out_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from models.histo.config import load_histo_config, section
from models.histo.patches.extract import read_slide_list, run_extraction, summarize, write_results
from models.histo.patches.grid import PatchGridConfig
from models.histo.wsi.tissue import TissueMaskConfig

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="patch_extract")
    parser.add_argument("--slide-list", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_histo_config(args.config, args.overrides)
    grid_cfg = PatchGridConfig(**section(cfg, "grid"))
    mask_cfg = TissueMaskConfig(**section(cfg, "tissue_mask"))
    slide_list = args.slide_list or ROOT / str(cfg.run.slide_list)
    out_dir = args.out or ROOT / str(cfg.run.out_dir)
    specs = read_slide_list(slide_list)
    print(f"{len(specs)} slides from {slide_list} -> {out_dir} (workers={int(cfg.run.workers)})")

    results = run_extraction(
        specs,
        out_dir,
        grid_cfg,
        mask_cfg,
        workers=int(cfg.run.workers),
        overwrite=bool(cfg.run.overwrite),
    )
    results_csv = (
        Path(str(cfg.run.results_csv)) if args.out is None else out_dir.with_suffix(".results.csv")
    )
    write_results(results, ROOT / results_csv if not results_csv.is_absolute() else results_csv)
    summary = summarize(results)
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    failed = [r for r in results if r.status == "failed"]
    for r in failed:
        print(f"FAILED {r.slide_id}: {r.error}")
    print("Koordinat sistemi: x, y = seviye-0 pikseli (docs/histo/koordinat_sistemi.md).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
