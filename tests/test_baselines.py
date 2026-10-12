"""Nested-CV baseline tests, focused on leakage, signal recovery and honest reporting.

The cohort is synthetic and built so the truth is known: TP53 depends on one gene's expression
(real signal), KRAS is pure noise, EGFR depends only on cancer type, BRCA1 is too rare to train.
"""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
pytest.importorskip("anndata")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from models.omics import baselines as B  # noqa: E402
from models.omics import eval_report as R  # noqa: E402
from models.omics import metrics as metrics_module  # noqa: E402
from models.omics.baselines import (  # noqa: E402
    POOLED,
    TRIVIAL,
    BaselineConfig,
    config_from,
    load_dev_data,
    make_estimator,
    run_baselines,
)
from models.omics.pipeline import cohort_paths  # noqa: E402
from models.omics.preprocess import ExpressionPreprocessor  # noqa: E402
from models.omics.splits import SplitConfig, build_split_table, make_splits  # noqa: E402

GENES = ["TP53", "KRAS", "EGFR", "BRCA1"]
N_GENES = 90
PROJECTS = ["TCGA-AAA", "TCGA-BBB", "TCGA-CCC"]


def build_world(
    root: Path, n: int = 600, seed: int = 0, shuffle_labels: bool = False
) -> dict[str, Any]:
    """Write cohort/labels/splits files for a synthetic world; return paths and tables."""
    rng = np.random.default_rng(seed)
    project = rng.choice(PROJECTS, size=n, p=[0.45, 0.35, 0.20])
    ids = [f"TCGA-{p[-1]}{i % 7}-{i:04d}" for i, p in enumerate(project)]
    offset = {
        p: rng.normal(0, 0.6, N_GENES) for p in PROJECTS
    }  # cancer-type expression differences
    log_expr = np.stack([offset[p] for p in project]) + rng.normal(0, 1, (n, N_GENES))
    tpm = (2 ** (log_expr + 3)).astype(np.float32)  # all genes expressed well above TPM 1

    signal = log_expr[:, 0]
    tp53 = (signal + rng.normal(0, 0.5, n) > np.quantile(signal, 0.65)).astype(int)
    kras = (rng.random(n) < 0.20).astype(int)
    prev = {"TCGA-AAA": 0.05, "TCGA-BBB": 0.22, "TCGA-CCC": 0.5}
    egfr = np.array([rng.random() < prev[p] for p in project]).astype(int)
    brca1 = np.zeros(n, dtype=int)
    brca1[rng.choice(n, 10, replace=False)] = 1
    labels = pd.DataFrame(
        {
            "patient_barcode": ids,
            "project_id": project,
            "label_status": "labeled",
            "label_sample_barcode": [i + "-01A" for i in ids],
        }
    )
    for gene, values in zip(GENES, (tp53, kras, egfr, brca1), strict=True):
        v = rng.permutation(values) if shuffle_labels else values
        labels[gene] = pd.array(v, dtype="Int8")

    patient_map = pd.DataFrame(
        {
            "patient_barcode": ids,
            "project_id": project,
            "rnaseq_file_id": "f",
            "n_maf": 1,
            "n_slides": 1,
            "slide_file_ids": [[f"s{i}"] for i in range(n)],
        }
    )
    table = build_split_table(patient_map, labels, GENES)
    splits = make_splits(table, GENES, SplitConfig(min_positives_per_split=1)).splits

    cohort = root / "cohort"
    cohort.mkdir(parents=True)
    paths = cohort_paths(cohort)
    obs = pd.DataFrame(
        {"patient_barcode": ids, "project_id": project, "tss": "AA", "flag_any": False},
        index=pd.Index([i + "-01A" for i in ids], name="sample_barcode"),
    )
    var = pd.DataFrame(
        {
            "gene_id": [f"G{i}" for i in range(N_GENES)],
            "gene_name": [f"N{i}" for i in range(N_GENES)],
            "gene_type": "protein_coding",
        }
    ).set_index("gene_id", drop=False)
    np.save(paths["tpm"], tpm)
    obs.to_parquet(paths["obs"])
    var.to_parquet(paths["var"])
    paths["meta"].write_text(json.dumps({"n_samples": n, "n_genes": N_GENES}))
    splits.to_parquet(root / "splits.parquet", index=False)
    labels.to_parquet(root / "labels.parquet", index=False)
    return {"root": root, "tpm": tpm, "labels": labels, "splits": splits, "cohort": cohort}


def fast_cfg(**kw: Any) -> BaselineConfig:
    base: dict[str, Any] = dict(
        genes=tuple(GENES),
        models=("logreg_l2",),
        seed=1,
        top_genes=40,
        n_boot=60,
        min_positives=30,
        inner_folds=3,
        per_cancer=False,
        grids={"logreg_l2": {"C": [0.1]}},
    )
    base.update(kw)
    return BaselineConfig(**base)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("world")
    w = build_world(root)
    w["dev"] = load_dev_data(w["cohort"], root / "splits.parquet", root / "labels.parquet", GENES)
    w["results"], w["oof"] = run_baselines(w["dev"], fast_cfg(), log=lambda m: None)
    return w


def row(
    results: pd.DataFrame, gene: str, model: str = "logreg_l2", scope: str = POOLED
) -> pd.Series:
    sel = results[
        (results["gene"] == gene) & (results["model"] == model) & (results["scope"] == scope)
    ]
    assert len(sel) == 1, (gene, model, scope, len(sel))
    return sel.iloc[0]


# ----------------------------------------------------------------- no test patients
def test_dev_pool_contains_no_test_patients(world: dict[str, Any]) -> None:
    dev = world["dev"]
    test_ids = set(world["splits"].loc[world["splits"]["split"] == "test", "patient_barcode"])
    assert test_ids and not (set(dev.patients["patient_barcode"]) & test_ids)
    assert (dev.patients["cv_fold"] >= 0).all()
    assert len(dev.patients) == int((world["splits"]["split"] != "test").sum())


def test_test_patients_expression_is_never_read(tmp_path: Path) -> None:
    """Fill all test patients' expression with NaN: any read would raise; results stay identical."""
    w = build_world(tmp_path / "a")
    base_dev = load_dev_data(
        w["cohort"], w["root"] / "splits.parquet", w["root"] / "labels.parquet", GENES
    )
    base_res, base_oof = run_baselines(
        base_dev, fast_cfg(genes=("TP53", "KRAS")), log=lambda m: None
    )

    path = cohort_paths(w["cohort"])["tpm"]
    tpm = np.load(path)
    obs = pd.read_parquet(cohort_paths(w["cohort"])["obs"])
    test_ids = set(w["splits"].loc[w["splits"]["split"] == "test", "patient_barcode"])
    test_rows = np.flatnonzero(obs["patient_barcode"].isin(test_ids).to_numpy())
    assert len(test_rows) == len(test_ids) > 50  # these rows really are the test patients
    tpm[test_rows] = np.nan
    np.save(path, tpm)
    dev = load_dev_data(
        w["cohort"], w["root"] / "splits.parquet", w["root"] / "labels.parquet", GENES
    )
    res, oof = run_baselines(dev, fast_cfg(genes=("TP53", "KRAS")), log=lambda m: None)
    pd.testing.assert_frame_equal(base_oof, oof)
    pd.testing.assert_frame_equal(base_res, res)


def test_preprocessing_is_fitted_per_fold_on_training_rows_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = build_world(tmp_path / "w")
    dev = load_dev_data(
        w["cohort"], w["root"] / "splits.parquet", w["root"] / "labels.parquet", GENES
    )
    seen: list[np.ndarray[Any, Any]] = []
    original = ExpressionPreprocessor.fit

    def spy(self: Any, X: Any, y: Any = None, sample_idx: Any = None) -> Any:  # noqa: N803
        seen.append(np.asarray(sample_idx))
        return original(self, X, y, sample_idx)

    monkeypatch.setattr(ExpressionPreprocessor, "fit", spy)
    positions = np.arange(len(dev.patients))
    folds = B.prepare_scope(dev, positions, fast_cfg())
    assert len(seen) == 5
    for fold, used in zip(folds, seen, strict=True):
        held_out_rows = set(dev.rows[fold.test])
        assert not (set(used) & held_out_rows), "fold test rows were used to fit the preprocessor"
        assert set(used) == set(dev.rows[fold.train])


def test_f1_threshold_uses_training_side_predictions_only(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    dev = world["dev"]
    folds = B.prepare_scope(dev, np.arange(len(dev.patients)), fast_cfg())
    recorded: list[int] = []
    original = metrics_module.best_f1_threshold

    def spy(y: Any, p: Any, *args: Any, **kwargs: Any) -> float:
        recorded.append(len(y))
        return original(y, p, *args, **kwargs)

    monkeypatch.setattr(metrics_module, "best_f1_threshold", spy)
    B.run_target(dev, folds, "TP53", "logreg_l2", fast_cfg(), POOLED)
    assert recorded == [
        len(f.train) for f in folds
    ]  # exactly the training rows, never the held-out fold


# ----------------------------------------------------------------------- signal vs noise
def test_real_signal_is_recovered_and_noise_stays_near_chance(world: dict[str, Any]) -> None:
    tp53, kras = row(world["results"], "TP53"), row(world["results"], "KRAS")
    assert tp53["auroc"] > 0.75 and tp53["auroc_lo"] > 0.65
    assert tp53["auprc"] > tp53["prevalence"] + 0.15
    assert abs(kras["auroc"] - 0.5) < 0.12
    assert kras["auroc_lo"] <= 0.5 <= kras["auroc_hi"]  # the interval contains chance level


def test_shuffled_labels_give_chance_level_performance(tmp_path: Path) -> None:
    """If this fails, labels or information leak into the features (the canary test)."""
    w = build_world(tmp_path / "s", shuffle_labels=True)
    dev = load_dev_data(
        w["cohort"], w["root"] / "splits.parquet", w["root"] / "labels.parquet", GENES
    )
    res, _ = run_baselines(dev, fast_cfg(genes=("TP53", "KRAS", "EGFR")), log=lambda m: None)
    ok = res[(res["status"] == "ok") & (res["model"] == "logreg_l2") & (res["scope"] == POOLED)]
    assert len(ok) == 3 and ((ok["auroc"] - 0.5).abs() < 0.13).all(), ok[["gene", "auroc"]]


def test_trivial_baseline_captures_cancer_type_and_comparison_is_honest(
    world: dict[str, Any],
) -> None:
    results, oof = world["results"], world["oof"]
    triv = row(results, "EGFR", TRIVIAL)
    assert triv["auroc"] > 0.65  # EGFR prevalence differs by cancer type
    comparison = R.trivial_comparison(results, oof, fast_cfg()).set_index("gene")
    assert (
        comparison.loc["TP53", "d_auroc_lo"] > 0
    )  # expression adds information beyond cancer type
    assert "fazla bilgi" in R.verdict(comparison.loc["TP53"])
    egfr = comparison.loc["EGFR"]
    assert (
        egfr["d_auroc_lo"] <= 0 <= egfr["d_auroc_hi"] or egfr["d_auroc"] < 0.1
    )  # no real gain claimed
    assert "fazla bilgi veriyor" not in R.verdict(egfr)


# ------------------------------------------------------------------ bookkeeping and honesty
def test_rare_target_is_reported_as_skipped_not_trained(world: dict[str, Any]) -> None:
    brca1 = world["results"][world["results"]["gene"] == "BRCA1"]
    assert (brca1["status"].str.startswith("skipped")).all() and brca1["auroc"].isna().all()
    assert "BRCA1" not in set(world["oof"]["gene"])


def test_oof_covers_every_patient_exactly_once_per_target(world: dict[str, Any]) -> None:
    oof, dev = world["oof"], world["dev"]
    for (_, _, _), part in oof.groupby(["gene", "scope", "model"]):
        assert part["patient_barcode"].is_unique and len(part) == len(dev.patients)
        assert part["proba"].between(0, 1).all()
        assert set(part["outer_fold"]) == {0, 1, 2, 3, 4}


def test_results_are_reproducible(world: dict[str, Any]) -> None:
    again, oof = run_baselines(world["dev"], fast_cfg(), log=lambda m: None)
    pd.testing.assert_frame_equal(world["results"], again)
    pd.testing.assert_frame_equal(world["oof"], oof)


def test_per_cancer_scopes_and_pooled_subgroups(world: dict[str, Any]) -> None:
    res, _ = run_baselines(
        world["dev"],
        fast_cfg(genes=("TP53",), per_cancer=True, min_positives=20),
        log=lambda m: None,
    )
    scopes = set(res["scope"])
    assert POOLED in scopes and {f"{POOLED}@{p}" for p in PROJECTS} <= scopes
    assert {"TCGA-AAA", "TCGA-BBB"} <= scopes
    sub = res[res["scope"] == f"{POOLED}@TCGA-AAA"].iloc[0]
    assert sub["n"] < row(res, "TP53")["n"]


@pytest.mark.parametrize("name", ["logreg_l1", "logreg_l2", "random_forest", "gbm", "mlp"])
def test_every_model_trains_and_predicts_probabilities(name: str, world: dict[str, Any]) -> None:
    dev = world["dev"]
    folds = B.prepare_scope(dev, np.arange(len(dev.patients)), fast_cfg())[:1]
    cfg = fast_cfg(models=(name,), grids={})
    oof, reason = B.run_target(dev, folds, "TP53", name, cfg, POOLED)
    assert reason is None and oof["proba"].between(0, 1).all() and oof["y"].nunique() == 2
    est = make_estimator(name, 0).fit(
        folds[0].x_train, dev.patients["TP53"].to_numpy()[folds[0].train]
    )
    assert est.predict_proba(folds[0].x_test).shape[1] == 2


def test_config_validation_enforces_plan_rules() -> None:
    from omegaconf import OmegaConf

    raw = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "configs" / "omics" / "baselines.yaml"
    )
    assert config_from(raw).n_boot >= 1000  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_boot"):
        config_from(OmegaConf.merge(raw, {"n_boot": 100}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_version"):
        config_from(OmegaConf.merge(raw, {"feature_version": "other"}))  # type: ignore[arg-type]


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        make_estimator("transformer", 0)


# ---------------------------------------------------------------------------- reporting
def test_suspiciously_high_auroc_is_flagged() -> None:
    results = pd.DataFrame(
        [{"status": "ok", "scope": POOLED, "gene": "TP53", "model": "gbm", "auroc": 0.99}]
    )
    assert any("sızıntı" in w for w in R.suspicious(results))
    assert not R.suspicious(results.assign(auroc=0.8))


def test_report_and_figures_are_written_and_contain_no_patient_ids(
    world: dict[str, Any], tmp_path: Path
) -> None:
    cfg = fast_cfg()
    written = R.build_report(world["results"], world["oof"], cfg, "unit", tmp_path)
    names = {p.name for p in written}
    assert {"unit_rapor.md", "unit_roc.png", "unit_pr.png", "unit_calibration.png"} <= names
    text = (tmp_path / "unit_rapor.md").read_text(encoding="utf-8")
    for needle in ("test kümesine dokunulmadı", "Atlanan kapsamlar", "BRCA1", "ΔAUPRC", "Birleşik"):
        assert needle in text
    import re

    assert not re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", text)  # no patient barcodes
    with pytest.raises(FileExistsError):
        R.build_report(world["results"], world["oof"], cfg, "unit", tmp_path)
    assert all(p.stat().st_size < 500_000 for p in written)


def test_cli_quick_run_end_to_end_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The >= 1000 bootstrap rule is enforced by config_from (tested separately); the bootstrap
    # itself is unit-tested in test_metrics. Use few resamples here so the end-to-end run is fast.
    real_boot, real_delta = metrics_module.bootstrap_ci, metrics_module.paired_bootstrap_delta
    real_hgb = HistGradientBoostingClassifier

    def few_boot(y: Any, p: Any, metrics: Any, n_boot: int = 1000, seed: int = 0, **k: Any) -> Any:
        return real_boot(y, p, metrics, n_boot=40, seed=seed, **k)

    def few_delta(
        y: Any, a: Any, b: Any, metric: Any, n_boot: int = 1000, seed: int = 0, **k: Any
    ) -> Any:
        return real_delta(y, a, b, metric, n_boot=40, seed=seed, **k)

    monkeypatch.setattr(metrics_module, "bootstrap_ci", few_boot)
    monkeypatch.setattr(metrics_module, "paired_bootstrap_delta", few_delta)
    monkeypatch.setattr(
        B, "HistGradientBoostingClassifier", lambda **kw: real_hgb(**{**kw, "max_iter": 15})
    )
    from scripts.train_baselines import main

    w = build_world(tmp_path / "w")
    (tmp_path / "w" / "splits-v1").mkdir()
    (tmp_path / "w" / "splits.parquet").rename(tmp_path / "w" / "splits-v1" / "splits.parquet")
    common = [
        "--splits-version",
        "splits-v1",
        "--cohort",
        str(w["cohort"]),
        "--labels",
        str(tmp_path / "w" / "labels.parquet"),
        "--splits-root",
        str(tmp_path / "w"),
        "--out-root",
        str(tmp_path / "out"),
        "--report-dir",
        str(tmp_path / "docs"),
    ]
    args = [
        "--run",
        "r1",
        "--quick",
        *common,
        "genes=[TP53]",
        "top_genes=30",
        "per_cancer=false",
    ]
    assert main(args) == 0
    assert (tmp_path / "out" / "r1" / "results.csv").stat().st_size > 0
    assert (tmp_path / "out" / "r1" / "oof_predictions.parquet").exists()
    assert "TP53" in (tmp_path / "docs" / "r1_rapor.md").read_text(encoding="utf-8")
    before = (tmp_path / "out" / "r1" / "results.csv").read_bytes()
    assert main(args) == 1  # same run name: refuses to overwrite
    assert (tmp_path / "out" / "r1" / "results.csv").read_bytes() == before
