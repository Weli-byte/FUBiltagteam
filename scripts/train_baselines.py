"""Train and evaluate the baseline omics models with nested CV (mini sprint 1.8).

    uv run python scripts/train_baselines.py --run baseline-v1 --splits-version splits-v1
    uv run python scripts/train_baselines.py --run quick --splits-version splits-v1 --quick

Reads (never writes) the omics cohort of mini sprint 1.4, the labels of 1.5 and the splits of 1.6.
Only the development pool (train+val) is used; the test split is not touched here.
Writes data/processed/baselines/<run>/ (results.csv, oof_predictions.parquet; not for git) and
docs/baselines/<run>_rapor.md + figures (aggregate numbers). Existing runs are never overwritten.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from models.omics.baselines import config_from, load_dev_data, run_baselines
from models.omics.eval_report import build_report

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run", required=True, help="run name, e.g. baseline-v1")
    parser.add_argument("--splits-version", default="splits-v1")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "omics" / "baselines.yaml"
    )
    parser.add_argument("--cohort", type=Path, default=ROOT / "data" / "processed" / "omics")
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "data" / "processed" / "labels" / "labels.parquet"
    )
    parser.add_argument("--splits-root", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "processed" / "baselines")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs" / "baselines")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one value per grid, fewer models: a fast smoke run (results are NOT final)",
    )
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. n_jobs=2 genes=[TP53]")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = args.out_root / args.run
    report = args.report_dir / f"{args.run}_rapor.md"
    if out.exists() or report.exists():
        print(f"DUR: run '{args.run}' already exists; not overwriting. Choose a new --run name.")
        return 1

    cfg_raw = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    if args.quick:
        raw_grids: Any = OmegaConf.to_container(cfg_raw.grids)
        quick_grids = {m: {k: v[:1] for k, v in g.items()} for m, g in raw_grids.items()}
        cfg_raw = OmegaConf.merge(
            cfg_raw, OmegaConf.create({"models": ["logreg_l2", "gbm"], "grids": quick_grids})
        )
    cfg = config_from(cfg_raw)  # type: ignore[arg-type]

    dev = load_dev_data(
        args.cohort,
        args.splits_root / args.splits_version / "splits.parquet",
        args.labels,
        cfg.genes,
        cfg.drop_flagged,
    )
    print(
        f"development pool: {len(dev.patients)} labeled train/val patients "
        f"(test patients are never loaded); projects: "
        f"{dev.patients['project_id'].value_counts().to_dict()}"
    )
    results, oof = run_baselines(dev, cfg, log=lambda m: print(m, flush=True))

    out.mkdir(parents=True)
    results.to_csv(out / "results.csv", index=False)
    oof.to_parquet(out / "oof_predictions.parquet", index=False)
    written = build_report(results, oof, cfg, args.run, args.report_dir)
    print("\n".join(f"wrote {p}" for p in written))
    if args.quick:
        print("UYARI: --quick sonuclari nihai degildir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
