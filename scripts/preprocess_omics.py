"""Omics preprocessing CLI (mini sprint 1.4).

  inspect  <file>     print the layout of one STAR-Counts file (verify the expected columns)
  build               assemble the TPM matrix + sample QC + batch diagnostics (no learning)
  fit-transform       fit gene filter / selection / z-score on TRAIN samples, transform all

Examples (run `uv sync --group data --group omics` first):
  uv run python scripts/preprocess_omics.py inspect data/raw/gdc/TCGA-LUAD/rnaseq/<id>/<file>.tsv
  uv run python scripts/preprocess_omics.py build
  uv run python scripts/preprocess_omics.py fit-transform --train-ids data/interim/train_ids.txt
  uv run python scripts/preprocess_omics.py fit-transform --eda-all-samples   # exploration only!
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from models.omics.expression import parse_star_counts
from models.omics.pipeline import build_cohort, fit_transform, load_ids

ROOT = Path(__file__).resolve().parents[1]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "omics" / "preprocess.yaml"
    )
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. mad_threshold=3.5")


def _cfg(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("file", type=Path)

    p = sub.add_parser("build")
    _common(p)
    p.add_argument(
        "--patient-map", type=Path, default=ROOT / "data" / "interim" / "patient_map.parquet"
    )
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "gdc")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "omics")
    p.add_argument("--report-dir", type=Path, default=ROOT / "docs" / "omics_qc")
    p.add_argument("--projects", nargs="+", default=None)
    p.add_argument("--meta", type=Path, default=ROOT / "data" / "manifests" / "query_meta.json")

    p = sub.add_parser("fit-transform")
    _common(p)
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "omics",
        help="cohort directory made by `build`",
    )
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "omics")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train-ids", type=Path, help="file with one train sample/patient barcode per line"
    )
    group.add_argument(
        "--eda-all-samples", action="store_true", help="fit on ALL samples (exploration only)"
    )
    p.add_argument("--drop-flagged", action="store_true", help="exclude QC-flagged samples")
    p.add_argument(
        "--gene-sets", type=Path, default=None, help="optional .gmt file for pathway scores"
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "inspect":
        head = args.file.read_text(encoding="utf-8").splitlines()[:8]
        print("\n".join(head))
        parsed = parse_star_counts(args.file)
        print(
            f"\nOK: {len(parsed.gene_id)} genes; types: "
            f"{pd.Series(parsed.gene_type).value_counts().head(5).to_dict()}"
        )
        return 0

    cfg = _cfg(args)
    if args.command == "build":
        release = None
        if args.meta.exists():
            release = json.loads(args.meta.read_text(encoding="utf-8")).get("data_release")
        result = build_cohort(
            pd.read_parquet(args.patient_map),
            args.raw,
            args.out,
            args.report_dir,
            cfg,
            projects=args.projects,
            data_release=release,
        )
        print(json.dumps(result, indent=2, default=str))
        print(f"Report: {args.report_dir / 'qc_report.md'}")
        return 0

    written = fit_transform(
        args.input,
        args.out,
        cfg,
        load_ids(args.train_ids),
        eda_all_samples=args.eda_all_samples,
        drop_flagged=args.drop_flagged,
        gene_sets_path=args.gene_sets,
    )
    print("\n".join(str(p) for p in written))
    if args.eda_all_samples:
        print("UYARI: --eda-all-samples çıktıları model eğitimi için KULLANILMAZ (veri sızıntısı).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
