"""Print labeled patients for manual checking against cBioPortal (read-only).

show_label_sample.py                 5 random labeled patients (seed 1)
show_label_sample.py --positives     2 EGFR+ LUAD, 2 KRAS+ COAD, 1 TP53+ BRCA patients (seed 1),
                                     with the counted variants (protein change) to look up
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = {"patient_barcode", "project_id", "label_status", "label_sample_barcode"}
# (project, gene, how many)
POSITIVE_PICKS = (("TCGA-LUAD", "EGFR", 2), ("TCGA-COAD", "KRAS", 2), ("TCGA-BRCA", "TP53", 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--positives", action="store_true")
    args = parser.parse_args(argv)

    folder = ROOT / "data" / "processed" / "labels"
    labels = pd.read_parquet(folder / "labels.parquet")
    genes = [c for c in labels.columns if c not in META]
    labeled = labels[labels["label_status"] == "labeled"]
    columns = ["patient_barcode", "project_id", "label_sample_barcode", *genes]

    if not args.positives:
        print(labeled.sample(min(5, len(labeled)), random_state=1)[columns].to_string(index=False))
        return 0

    picked = []
    for project, gene, count in POSITIVE_PICKS:
        pool = labeled[(labeled["project_id"] == project) & (labeled[gene] == 1)]
        if len(pool) < count:
            print(f"UYARI: {project} icinde yalnizca {len(pool)} {gene}-pozitif hasta var")
        picked.append(pool.sample(min(count, len(pool)), random_state=1).assign(check_gene=gene))
    chosen = pd.concat(picked)
    print(chosen[[*columns, "check_gene"]].to_string(index=False))

    detail_path = folder / "labels_variant_detail.parquet"
    if detail_path.exists():
        detail = pd.read_parquet(detail_path)
        wanted = chosen[["patient_barcode", "check_gene"]]
        merged = detail.merge(
            wanted,
            left_on=["patient_barcode", "Hugo_Symbol"],
            right_on=["patient_barcode", "check_gene"],
        )
        merged = merged[merged["counted"].astype(bool)]
        cols = [
            c
            for c in (
                "patient_barcode",
                "Hugo_Symbol",
                "Variant_Classification",
                "HGVSp_Short",
                "Exon_Number",
                "annotations",
            )
            if c in merged.columns
        ]
        print("\nSayilan varyantlar (cBioPortal'da bu protein degisikligini ara):")
        print(merged[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
