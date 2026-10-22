"""Check the patch data against the project leader's patient-level splits (mini sprint 1.10).

Bu dosya ne işe yarar? ``slide_splits.parquet`` (proje lideri, 1.6) ile patch/slayt tablolarını
(örnek paket ``patches.csv`` ya da koordinat klasörü) karşılaştırır: aynı hasta iki kümede olamaz,
bölmesi olmayan slayt eğitime giremez, patch'in kümesi slaytınkiyle aynı olmalı. Sorun bulursa
sıfırdan farklı çıkış kodu verir ve sorunları listeler; bölme dosyasını asla değiştirmez.

Usage:
    uv run --group histo python scripts/check_split_compat.py \
        --slide-splits data/processed/splits/splits-v1/slide_splits.parquet \
        --patches data/processed/patch_v0/patches.csv
    uv run --group histo python scripts/check_split_compat.py --slide-splits ... \
        --coords-dir data/processed/patches/coords_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from models.histo.patches.splits import check_patch_split_leakage, slide_split_summary


def coords_table(coords_dir: Path) -> pd.DataFrame:
    """One row per coordinate file (slide) in a directory."""
    from models.histo.patches.h5 import read_coords_h5

    rows = []
    for path in sorted(coords_dir.glob("*.h5")):
        coords, _, attrs = read_coords_h5(path)
        rows.append({"slide_id": attrs.slide_id, "n_patches": len(coords)})
    return pd.DataFrame(rows, columns=["slide_id", "n_patches"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--slide-splits", type=Path, required=True)
    parser.add_argument("--patches", type=Path, default=None, help="patches.csv of a sample pack")
    parser.add_argument(
        "--coords-dir", type=Path, default=None, help="directory of coordinate HDF5 files"
    )
    args = parser.parse_args(argv)
    if (args.patches is None) == (args.coords_dir is None):
        parser.error("give exactly one of --patches or --coords-dir")

    slide_splits = pd.read_parquet(args.slide_splits)
    patches = (
        pd.read_csv(args.patches) if args.patches is not None else coords_table(args.coords_dir)
    )
    patches["slide_id"] = patches["slide_id"].astype(str)
    problems = check_patch_split_leakage(slide_splits, patches)
    print(slide_split_summary(slide_splits, patches).to_string(index=False))
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"- {p}")
        print("\nBölme dosyasını değiştirmeyin; sorunu proje lideriyle birlikte çözün.")
        return 1
    print("\nOK: hasta düzeyi kesişim sıfır; her slayt tek kümede.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
