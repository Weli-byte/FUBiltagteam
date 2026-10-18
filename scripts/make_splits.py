"""Create (once) and verify patient-level splits (mini sprint 1.6).

  make:    uv run python scripts/make_splits.py --version splits-v1
  verify:  uv run python scripts/make_splits.py --verify splits-v1

Inputs : data/interim/patient_map.parquet, data/processed/labels/labels.parquet (1.3, 1.5)
Outputs: data/processed/splits/<version>/  (patient-level; never committed)
           splits.parquet, slide_splits.parquet, train_ids.txt, val_ids.txt, test_ids.txt,
           train_ids_omics.txt (train patients with RNA-seq: `fit-transform --train-ids` input)
         docs/splits/<version>_manifest.json + <version>_dagilim_raporu.md  (aggregates and hashes)
An existing version is NEVER overwritten: create a new version name instead.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from models.omics.splits import (
    SplitConfig,
    assert_no_leakage,
    build_manifest,
    build_split_table,
    check_leakage,
    distribution_markdown,
    get_ids,
    make_splits,
    slide_table,
    verify_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
LABEL_META = {"patient_barcode", "project_id", "label_status", "label_sample_barcode"}


def _config(path: Path, overrides: list[str]) -> SplitConfig:
    cfg = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides))
    return SplitConfig(
        seed=int(cfg.seed),
        ratios=tuple(float(r) for r in cfg.ratios),  # type: ignore[arg-type]
        n_folds=int(cfg.n_folds),
        site_holdout_fraction=float(cfg.site_holdout_fraction),
        min_positives_per_split=int(cfg.min_positives_per_split),
        max_ratio_deviation=float(cfg.max_ratio_deviation),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--version", default="splits-v1")
    parser.add_argument(
        "--verify", metavar="VERSION", default=None, help="verify an existing version"
    )
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 if the balance warnings exist"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "omics" / "splits.yaml")
    parser.add_argument(
        "--patient-map", type=Path, default=ROOT / "data" / "interim" / "patient_map.parquet"
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "data" / "processed" / "labels" / "labels.parquet"
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs" / "splits")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. seed=7")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.verify:
        out = args.out_root / args.verify
        splits = pd.read_parquet(out / "splits.parquet")
        slides = pd.read_parquet(out / "slide_splits.parquet")
        manifest = json.loads(
            (args.report_dir / f"{args.verify}_manifest.json").read_text(encoding="utf-8")
        )
        problems = verify_manifest(splits, manifest) + check_leakage(splits, slides)
        if problems:
            print("VERIFY FAILED:\n- " + "\n- ".join(problems))
            return 1
        print(
            f"OK: {args.verify} matches its frozen manifest and has no leakage "
            f"({len(splits)} patients, {len(slides)} slides)"
        )
        return 0

    cfg = _config(args.config, args.overrides)
    out = args.out_root / args.version
    manifest_path = args.report_dir / f"{args.version}_manifest.json"
    if out.exists() or manifest_path.exists():
        print(
            f"DUR: {args.version} already exists; not overwriting. "
            "Use --version splits-v2 for a new split."
        )
        return 1

    patient_map = pd.read_parquet(args.patient_map)
    labels = pd.read_parquet(args.labels)
    genes = [c for c in labels.columns if c not in LABEL_META]
    table = build_split_table(patient_map, labels, genes)
    result = make_splits(table, genes, cfg)
    slides = slide_table(patient_map, result.splits)
    assert_no_leakage(result.splits, slides)

    out.mkdir(parents=True)
    result.splits.to_parquet(out / "splits.parquet", index=False)
    slides.to_parquet(out / "slide_splits.parquet", index=False)
    for name in ("train", "val", "test"):
        ids = get_ids(result.splits, name, final_evaluation=True)
        (out / f"{name}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    omics_train = get_ids(result.splits, "train", modality="has_rnaseq")
    (out / "train_ids_omics.txt").write_text("\n".join(omics_train) + "\n", encoding="utf-8")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        build_manifest(result.splits, args.version, cfg, genes, result.warnings), manifest_path
    )
    (args.report_dir / f"{args.version}_dagilim_raporu.md").write_text(
        distribution_markdown(result.splits, genes, args.version, result.warnings, cfg),
        encoding="utf-8",
    )
    print(f"{args.version}: {len(result.splits)} patients -> {out}")
    print(result.splits["split"].value_counts().to_string())
    print(f"warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"  - {warning}")
    if args.strict and result.warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
