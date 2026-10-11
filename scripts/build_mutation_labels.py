"""Build patient-level mutation labels from the downloaded GDC MAF files (mini sprint 1.5).

    uv run python scripts/build_mutation_labels.py
    uv run python scripts/build_mutation_labels.py genes=[TP53,KRAS,EGFR,BRCA1,PIK3CA]

Inputs : data/interim/patient_map.parquet, data/raw/gdc/<project>/maf/...
Outputs: data/processed/labels/labels.parquet, labels_variant_detail.parquet (not for git),
         docs/mutation_labels/dengesizlik_raporu.md + figure (aggregate counts, safe for git)
Rules  : configs/omics/labels.yaml (see docs/etiket_tanimi.md)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from models.omics.labels import (
    balance_markdown,
    build_labels,
    class_balance,
    collect_maf,
    recommend,
    rules_from_config,
    save_balance_plot,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "omics" / "labels.yaml")
    parser.add_argument(
        "--patient-map", type=Path, default=ROOT / "data" / "interim" / "patient_map.parquet"
    )
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "gdc")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "labels")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs" / "mutation_labels")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. min_positives=20")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    rules = rules_from_config(cfg)  # type: ignore[arg-type]
    patient_map = pd.read_parquet(args.patient_map)
    print(f"patients in map: {len(patient_map)}; genes: {', '.join(rules.genes)}")

    variants, samples = collect_maf(patient_map, args.raw, rules.genes)
    result = build_labels(variants, samples, patient_map, rules)
    labels = result.labels
    if len(labels) != len(patient_map):
        raise RuntimeError("label table lost patients; this is a bug")

    balance = class_balance(labels, rules)
    rec = recommend(balance, rules)
    args.out.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.out / "labels.parquet", index=False)
    result.detail.to_parquet(args.out / "labels_variant_detail.parquet", index=False)
    (args.report_dir / "dengesizlik_raporu.md").write_text(
        balance_markdown(balance, rec, rules, result.n_excluded_by_class), encoding="utf-8"
    )
    save_balance_plot(balance, rules, args.report_dir / "sinif_dengesi.png")

    print(labels["label_status"].value_counts().to_string())
    print(balance[balance["project_id"] == "ALL"].to_string(index=False))
    print(f"\nWrote {args.out} and {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
