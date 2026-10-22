"""Baseline omics models with leakage-safe nested cross-validation (sprint 1.8).

Protocol
--------
* Only the development pool (patients with ``split`` in train/val and a ``cv_fold`` >= 0) is ever
  read. Test patients are excluded before any array is touched; this module has no test evaluation.
* Outer loop: the predefined ``cv_fold`` values. For every outer fold the preprocessor (expression
  filter, high-variance selection, z-score) is fitted on that fold's TRAINING rows only.
* Inner loop: grid search on the training rows (stratified K-fold, AUPRC) selects hyper-parameters,
  and the F1 threshold is chosen from inner cross-validated predictions of the training rows.
* The outer fold is predicted once with the selected model: these out-of-fold (OOF) predictions are
  what all metrics and confidence intervals are computed from.
* Class imbalance: class weights (inside every fold). No resampling: nothing synthetic is created.
* A trivial baseline (cancer type only) shows how much the omics features add.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier

from models.omics import metrics as M
from models.omics.pipeline import load_cohort
from models.omics.preprocess import ExpressionPreprocessor

LOG = logging.getLogger(__name__)
TRIVIAL = "project_only"
POOLED = "pooled"
ROW_CHUNK = 256


@dataclass(frozen=True)
class BaselineConfig:
    genes: tuple[str, ...]
    models: tuple[str, ...]
    seed: int = 42
    feature_version: str = "protein_coding"
    top_genes: int | None = 2000
    min_tpm: float = 1.0
    min_fraction: float = 0.2
    inner_folds: int = 3
    n_boot: int = 1000
    min_positives: int = 30
    per_cancer: bool = True
    drop_flagged: bool = False
    n_jobs: int = 1
    grids: dict[str, dict[str, list[Any]]] = field(default_factory=dict)


def _grids(cfg: DictConfig) -> dict[str, dict[str, list[Any]]]:
    raw: Any = OmegaConf.to_container(cfg.grids)
    return {m: {k: list(v) for k, v in g.items()} for m, g in raw.items()}


def config_from(cfg: DictConfig) -> BaselineConfig:
    """Build and validate the run configuration (reported results need >= 1000 bootstraps)."""
    if int(cfg.n_boot) < 1000:
        raise ValueError("n_boot must be >= 1000 for reported results")
    if str(cfg.feature_version) not in ("protein_coding", "all"):
        raise ValueError("feature_version must be 'protein_coding' or 'all'")
    top = cfg.top_genes
    return BaselineConfig(
        genes=tuple(str(g) for g in cfg.genes),
        models=tuple(str(m) for m in cfg.models),
        seed=int(cfg.seed),
        feature_version=str(cfg.feature_version),
        top_genes=None if top is None else int(top),
        min_tpm=float(cfg.min_tpm),
        min_fraction=float(cfg.min_fraction),
        inner_folds=int(cfg.inner_folds),
        n_boot=int(cfg.n_boot),
        min_positives=int(cfg.min_positives),
        per_cancer=bool(cfg.per_cancer),
        drop_flagged=bool(cfg.drop_flagged),
        n_jobs=int(cfg.n_jobs),
        grids=_grids(cfg),
    )


# ------------------------------------------------------------------------------- data
@dataclass
class DevData:
    """Development pool: matrix rows, patient table (labels as 0/1) and the protein-coding mask."""

    tpm: np.ndarray[Any, Any]
    rows: np.ndarray[Any, Any]
    patients: pd.DataFrame
    protein_coding: np.ndarray[Any, Any]


def load_dev_data(
    cohort_dir: Path,
    splits_path: Path,
    labels_path: Path,
    genes: Sequence[str],
    drop_flagged: bool = False,
) -> DevData:
    """Join cohort, splits and labels; keep only labeled train/val patients with RNA-seq."""
    cohort = load_cohort(cohort_dir)
    obs = cohort.obs.reset_index().assign(row=np.arange(len(cohort.obs)))
    if drop_flagged:
        obs = obs[~obs["flag_any"].to_numpy()]
    splits = pd.read_parquet(splits_path)[["patient_barcode", "split", "cv_fold"]]
    labels = pd.read_parquet(labels_path)
    missing = [g for g in genes if g not in labels.columns]
    if missing:
        raise ValueError(f"labels lack gene columns {missing}")
    table = (
        obs[["patient_barcode", "project_id", "row"]]
        .merge(splits, on="patient_barcode", how="inner", validate="one_to_one")
        .merge(
            labels[["patient_barcode", "label_status", *genes]],
            on="patient_barcode",
            how="inner",
            validate="one_to_one",
        )
    )
    table = table[(table["split"] != "test") & (table["label_status"] == "labeled")]
    table = table.sort_values("patient_barcode").reset_index(drop=True)
    if (table["cv_fold"] < 0).any() or (table["split"] == "test").any():
        raise AssertionError("test patients reached the development pool")  # defensive
    for gene in genes:
        table[gene] = table[gene].astype(int)
    protein_coding = (cohort.var["gene_type"] == "protein_coding").to_numpy()
    return DevData(
        tpm=cohort.tpm,
        rows=table["row"].to_numpy(),
        patients=table.drop(columns=["row", "split", "label_status"]),
        protein_coding=protein_coding,
    )


# ----------------------------------------------------------------------------- models
class WeightedMLP(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Two-hidden-layer MLP with balanced class weights applied through ``sample_weight``."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        alpha: float = 1e-4,
        max_iter: int = 300,
        random_state: int = 0,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "WeightedMLP":  # noqa: N803
        y_arr = np.asarray(y)
        counts = np.bincount(y_arr, minlength=2).astype(float)
        weights = (len(y_arr) / (2.0 * np.maximum(counts, 1.0)))[y_arr]
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
            early_stopping=True,
            random_state=self.random_state,
        ).fit(X, y_arr, sample_weight=weights)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray[Any, Any]:  # noqa: N803
        return np.asarray(self.model_.predict_proba(X))

    def predict(self, X: Any) -> np.ndarray[Any, Any]:  # noqa: N803
        return np.asarray(self.classes_[np.argmax(self.predict_proba(X), axis=1)])


def gbm_backend() -> str:
    """``xgboost`` if installed, else scikit-learn's histogram gradient boosting."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return "hist_gradient_boosting"
    return "xgboost"


def make_estimator(name: str, seed: int, n_jobs: int = 1) -> BaseEstimator:
    if name == "logreg_l1":
        return LogisticRegression(
            l1_ratio=1.0, solver="saga", class_weight="balanced", max_iter=2000, random_state=seed
        )
    if name == "logreg_l2":
        return LogisticRegression(
            l1_ratio=0.0, solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=seed
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed
        )
    if name == "gbm":
        if gbm_backend() == "xgboost":  # pragma: no cover - optional dependency
            from xgboost import XGBClassifier

            class _XGB(XGBClassifier):  # type: ignore[misc]
                def fit(self, X: Any, y: Any, **kw: Any) -> Any:  # noqa: N803
                    pos = max(float(np.sum(y)), 1.0)
                    self.set_params(scale_pos_weight=(len(y) - pos) / pos)
                    return super().fit(X, y, **kw)

            return _XGB(n_estimators=200, tree_method="hist", n_jobs=n_jobs, random_state=seed)
        return HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=200, early_stopping=False, random_state=seed
        )
    if name == "mlp":
        return WeightedMLP(random_state=seed)
    raise ValueError(f"unknown model {name!r}")


# ------------------------------------------------------------------ nested CV machinery
def transform_rows(
    prep: ExpressionPreprocessor, tpm: np.ndarray[Any, Any], rows: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Apply a fitted preprocessor to ``tpm[rows]`` in row chunks."""
    parts = [prep.transform(tpm[rows[i : i + ROW_CHUNK]]) for i in range(0, len(rows), ROW_CHUNK)]
    return np.concatenate(parts, axis=0)


@dataclass
class FoldData:
    fold: int
    train: np.ndarray[Any, Any]  # positions in dev.patients
    test: np.ndarray[Any, Any]
    x_train: np.ndarray[Any, Any]
    x_test: np.ndarray[Any, Any]


def prepare_scope(
    dev: DevData, positions: np.ndarray[Any, Any], cfg: BaselineConfig
) -> list[FoldData]:
    """Per outer fold: fit the preprocessor on training rows only and build train/test matrices."""
    folds = dev.patients["cv_fold"].to_numpy()
    allowed = dev.protein_coding if cfg.feature_version == "protein_coding" else None
    out: list[FoldData] = []
    for fold in sorted(set(folds[positions])):
        train = positions[folds[positions] != fold]
        test = positions[folds[positions] == fold]
        prep = ExpressionPreprocessor(
            min_tpm=cfg.min_tpm,
            min_fraction=cfg.min_fraction,
            top_k=cfg.top_genes,
            allowed_genes=allowed,
            zscore=True,
        ).fit(dev.tpm, sample_idx=dev.rows[train])
        out.append(
            FoldData(
                fold=int(fold),
                train=train,
                test=test,
                x_train=transform_rows(prep, dev.tpm, dev.rows[train]),
                x_test=transform_rows(prep, dev.tpm, dev.rows[test]),
            )
        )
    return out


def project_one_hot(dev: DevData, projects: Sequence[str]) -> np.ndarray[Any, Any]:
    return np.stack(
        [(dev.patients["project_id"] == p).to_numpy() for p in projects], axis=1
    ).astype(float)


def run_target(
    dev: DevData,
    folds: list[FoldData],
    gene: str,
    model: str,
    cfg: BaselineConfig,
    scope: str,
    trivial_features: np.ndarray[Any, Any] | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Nested CV for one (gene, model, scope). Returns (OOF predictions, skip reason or None)."""
    y_all = dev.patients[gene].to_numpy()
    positions = np.concatenate([f.test for f in folds])
    n_pos = int(y_all[positions].sum())
    if n_pos < cfg.min_positives:
        return pd.DataFrame(), f"skipped: {n_pos} positives (< {cfg.min_positives})"
    for f in folds:
        if int(y_all[f.train].sum()) < cfg.inner_folds:
            return (
                pd.DataFrame(),
                f"skipped: fewer than {cfg.inner_folds} training positives in fold {f.fold}",
            )
        if len(np.unique(y_all[f.train])) < 2:
            return pd.DataFrame(), f"skipped: one class only in training fold {f.fold}"

    inner = StratifiedKFold(cfg.inner_folds, shuffle=True, random_state=cfg.seed)
    rows: list[pd.DataFrame] = []
    for f in folds:
        y_train, y_test = y_all[f.train], y_all[f.test]
        if model == TRIVIAL:
            assert trivial_features is not None
            x_train, x_test = trivial_features[f.train], trivial_features[f.test]
            estimator: Any = LogisticRegression(class_weight="balanced", max_iter=1000)
            params: dict[str, Any] = {}
        else:
            x_train, x_test = f.x_train, f.x_test
            base = make_estimator(model, cfg.seed, cfg.n_jobs)
            grid = cfg.grids.get(model, {})
            if grid:
                search = GridSearchCV(
                    base, grid, cv=inner, scoring="average_precision", n_jobs=cfg.n_jobs, refit=True
                )
                search.fit(x_train, y_train)
                estimator, params = search.best_estimator_, dict(search.best_params_)
            else:
                estimator, params = base.fit(x_train, y_train), {}
        inner_pred = cross_val_predict(
            clone(estimator), x_train, y_train, cv=inner, method="predict_proba"
        )[:, 1]
        threshold = M.best_f1_threshold(y_train, inner_pred)
        if model == TRIVIAL:
            estimator.fit(x_train, y_train)
        proba = estimator.predict_proba(x_test)[:, 1]
        rows.append(
            pd.DataFrame(
                {
                    "patient_barcode": dev.patients["patient_barcode"].to_numpy()[f.test],
                    "project_id": dev.patients["project_id"].to_numpy()[f.test],
                    "gene": gene,
                    "scope": scope,
                    "model": model,
                    "outer_fold": f.fold,
                    "y": y_test,
                    "proba": proba,
                    "threshold": threshold,
                    "params": json.dumps(params, default=str),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), None


def summarize_oof(oof: pd.DataFrame, cfg: BaselineConfig) -> dict[str, Any]:
    """Metrics with bootstrap CIs for one (gene, scope, model) block of OOF predictions."""
    threshold = float(np.median(oof.groupby("outer_fold")["threshold"].first()))
    summary = M.summarize(
        oof["y"].to_numpy(), oof["proba"].to_numpy(), threshold, n_boot=cfg.n_boot, seed=cfg.seed
    )
    summary["positives"] = int(oof["y"].sum())
    summary["threshold"] = threshold
    return summary


def run_baselines(
    dev: DevData, cfg: BaselineConfig, log: Any = LOG.info
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every (scope, gene, model); return ``(results, oof_predictions)``."""
    projects = sorted(dev.patients["project_id"].unique())
    scopes: dict[str, np.ndarray[Any, Any]] = {POOLED: np.arange(len(dev.patients))}
    if cfg.per_cancer:
        for project in projects:
            scopes[project] = np.flatnonzero((dev.patients["project_id"] == project).to_numpy())
    one_hot = project_one_hot(dev, projects)
    backend = gbm_backend()

    results: list[dict[str, Any]] = []
    all_oof: list[pd.DataFrame] = []
    for scope, positions in scopes.items():
        log(f"scope {scope}: {len(positions)} patients; preprocessing folds")
        folds = prepare_scope(dev, positions, cfg)
        models = list(cfg.models) + ([TRIVIAL] if scope == POOLED else [])
        for gene in cfg.genes:
            for model in models:
                oof, reason = run_target(dev, folds, gene, model, cfg, scope, one_hot)
                base = {
                    "gene": gene,
                    "scope": scope,
                    "model": model,
                    "backend": backend if model == "gbm" else "",
                }
                if reason:
                    log(f"  {scope}/{gene}/{model}: {reason}")
                    results.append({**base, "status": reason})
                    continue
                row = {**base, "status": "ok", **summarize_oof(oof, cfg)}
                results.append(row)
                all_oof.append(oof)
                log(
                    f"  {scope}/{gene}/{model}: AUROC {row['auroc']:.3f}  AUPRC {row['auprc']:.3f} "
                    f"(prevalence {row['prevalence']:.3f})"
                )
                if scope == POOLED:  # subgroup metrics of the pooled model per cancer type
                    for project in projects:
                        sub = oof[oof["project_id"] == project]
                        if int(sub["y"].sum()) >= 2 and sub["y"].nunique() == 2:
                            results.append(
                                {
                                    **base,
                                    "scope": f"{POOLED}@{project}",
                                    "status": "ok",
                                    **summarize_oof(sub, cfg),
                                }
                            )
    oof_all = pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()
    return pd.DataFrame(results), oof_all
