"""Cohort assembly + QC (``build``) and leakage-safe fitting (``fit_transform``), sprint 1.4.

Storage of the raw TPM cohort (a directory, default ``data/processed/omics``):
``expression_tpm.npy`` (float32 samples x genes, memory-mapped), ``expression_tpm_obs.parquet``
(sample table incl. QC flags, index = sample barcode), ``expression_tpm_var.parquet`` (gene
dictionary, row order == matrix columns) and ``expression_tpm_meta.json``. The matrix never has to
fit in RAM: every step reads it in gene blocks or row chunks.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from models.omics import qc as qcm
from models.omics.expression import (
    ExpressionData,
    assemble_expression,
    read_id_list,
    select_samples,
)
from models.omics.preprocess import ExpressionPreprocessor, aggregate_gene_sets, parse_gmt

LOG = logging.getLogger(__name__)
EDA_ONLY = "ALL_SAMPLES_EDA_ONLY_NOT_FOR_MODEL_TRAINING"
TPM_NAME = "expression_tpm"
ROW_CHUNK = 256  # samples per chunk when transforming; 256 x 60k genes x 4 B = ~60 MB


@dataclass(frozen=True)
class Cohort:
    """A stored TPM cohort: memory-mapped matrix + sample and gene tables."""

    tpm: np.ndarray[Any, Any]
    obs: pd.DataFrame
    var: pd.DataFrame
    meta: dict[str, Any]


def cohort_paths(directory: Path) -> dict[str, Path]:
    return {
        "tpm": directory / f"{TPM_NAME}.npy",
        "obs": directory / f"{TPM_NAME}_obs.parquet",
        "var": directory / f"{TPM_NAME}_var.parquet",
        "meta": directory / f"{TPM_NAME}_meta.json",
    }


def load_cohort(directory: Path | str) -> Cohort:
    paths = cohort_paths(Path(directory))
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"cohort files missing (run `build` first): {missing}")
    return Cohort(
        tpm=np.load(paths["tpm"], mmap_mode="r"),
        obs=pd.read_parquet(paths["obs"]),
        var=pd.read_parquet(paths["var"]),
        meta=json.loads(paths["meta"].read_text(encoding="utf-8")),
    )


def build_cohort(
    patient_map: pd.DataFrame,
    raw_root: Path,
    out_dir: Path,
    report_dir: Path,
    cfg: DictConfig,
    projects: Sequence[str] | None = None,
    data_release: str | None = None,
) -> dict[str, Any]:
    """Assemble the TPM matrix on disk, run sample QC and batch diagnostics, write the report.

    Nothing learned from data is applied here (no scaling, no gene filtering by prevalence).
    Existing output files are never overwritten.
    """
    if projects:
        patient_map = patient_map[patient_map["project_id"].isin(list(projects))]
    paths = cohort_paths(out_dir)
    for key in ("obs", "var", "meta"):
        if paths[key].exists():
            raise FileExistsError(f"{paths[key]} already exists; not overwriting")
    qc_path = out_dir / "sample_qc.parquet"
    if qc_path.exists():
        raise FileExistsError(f"{qc_path} already exists; not overwriting")

    data: ExpressionData = assemble_expression(patient_map, raw_root, tpm_path=paths["tpm"])
    qc = qcm.sample_qc(
        data.tpm,
        data.obs["library_size"].to_numpy(),
        data.obs["n_detected_genes"].to_numpy(),
        mad_threshold=float(cfg.mad_threshold),
    )
    qc.index = data.obs.index
    logged_top = qcm.top_variance_log(data.tpm, int(cfg.pca_top_genes))
    pca = qcm.pca_embedding(logged_top, int(cfg.pca_components), seed=int(cfg.seed))
    del logged_top
    umap_xy = qcm.umap_embedding(pca, seed=int(cfg.seed))
    project = data.obs["project_id"].to_numpy()
    tss = data.obs["tss"].to_numpy()
    batch = qcm.batch_effect_summary(pca, project, tss)
    corr = qcm.sample_correlation(data.tpm, project, int(cfg.max_corr_samples), seed=int(cfg.seed))
    figures = qcm.save_qc_plots(
        report_dir, qc.reset_index(drop=True), corr, project, tss, pca, umap_xy
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "qc_report.md").write_text(
        qcm.qc_report_markdown(data.obs, qc, batch, float(cfg.mad_threshold), figures),
        encoding="utf-8",
    )

    obs = data.obs.join(qc.drop(columns=["library_size", "n_detected_genes"]))
    obs.to_parquet(paths["obs"])
    data.genes.to_parquet(paths["var"])
    qc.to_parquet(qc_path)
    meta = {
        "source": "GDC STAR-Counts, column tpm_unstranded",
        "data_release": data_release or "unknown",
        "n_samples": int(data.tpm.shape[0]),
        "n_genes": int(data.tpm.shape[1]),
        "dtype": "float32",
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {
        "n_samples": meta["n_samples"],
        "n_genes": meta["n_genes"],
        "n_flagged": int(qc["flag_any"].sum()),
        "batch": batch,
    }


def _transform_rows(
    prep: ExpressionPreprocessor, tpm: np.ndarray[Any, Any], rows: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Apply a fitted preprocessor to ``tpm[rows]`` in row chunks (never the whole matrix)."""
    parts = [prep.transform(tpm[rows[i : i + ROW_CHUNK]]) for i in range(0, len(rows), ROW_CHUNK)]
    return np.concatenate(parts, axis=0)


def fit_transform(
    cohort_dir: Path,
    out_dir: Path,
    cfg: DictConfig,
    train_ids: Sequence[str] | None,
    eda_all_samples: bool = False,
    drop_flagged: bool = False,
    top_genes: Sequence[int | None] | None = None,
    versions: Sequence[str] | None = None,
    gene_sets_path: Path | None = None,
) -> list[Path]:
    """Fit the preprocessor on TRAINING samples only and transform every kept sample.

    Either ``train_ids`` (sample or patient barcodes, from the patient-level split of mini sprint
    1.6) or ``eda_all_samples=True`` is required. The latter is for exploration and is stamped into
    every output as ``fit_on = ALL_SAMPLES_EDA_ONLY_NOT_FOR_MODEL_TRAINING``.
    Existing output files are never overwritten.
    """
    if (train_ids is None) == (not eda_all_samples):
        raise ValueError("give exactly one of train_ids or eda_all_samples=True")
    cohort = load_cohort(cohort_dir)
    rows = np.arange(len(cohort.obs))
    if drop_flagged:
        rows = np.flatnonzero(~cohort.obs["flag_any"].to_numpy())
    obs = cohort.obs.iloc[rows]
    fit_pos = np.arange(len(rows)) if train_ids is None else select_samples(obs, train_ids)
    if len(fit_pos) < 2:
        raise ValueError(f"only {len(fit_pos)} training samples after QC filtering; need >= 2")
    fit_rows = rows[fit_pos]
    is_fit = np.zeros(len(rows), dtype=bool)
    is_fit[fit_pos] = True
    fit_on = EDA_ONLY if train_ids is None else f"train_ids:{len(fit_pos)}_samples"

    gene_sets = parse_gmt(gene_sets_path) if gene_sets_path else None
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [
        (v, k)
        for v in (versions or list(cfg.versions))
        for k in (top_genes if top_genes is not None else list(cfg.top_genes))
    ]
    for version, _ in plan:
        if version not in ("all", "protein_coding"):
            raise ValueError(f"unknown version {version!r}")
    existing = [
        out_dir / f"{stem}_{version}_top{k if k is not None else 'all'}.{ext}"
        for version, k in plan
        for stem, ext in (("expression", "h5ad"), ("preprocessor", "joblib"))
    ]
    clash = [str(p) for p in existing if p.exists()]
    if clash:
        raise FileExistsError(f"output files already exist, not overwriting: {clash[:3]}")

    written: list[Path] = []
    for version, k in plan:
        allowed = (
            (cohort.var["gene_type"] == "protein_coding").to_numpy()
            if version == "protein_coding"
            else None
        )
        prep = ExpressionPreprocessor(
            min_tpm=float(cfg.min_tpm),
            min_fraction=float(cfg.min_fraction),
            top_k=None if k is None else int(k),
            allowed_genes=allowed,
            zscore=bool(cfg.zscore),
            clip=None if cfg.clip is None else float(cfg.clip),
        ).fit(cohort.tpm, sample_idx=fit_rows)
        z = _transform_rows(prep, cohort.tpm, rows)
        tag = f"{version}_top{k if k is not None else 'all'}"
        out = ad.AnnData(
            X=z,
            obs=obs.assign(is_fit_sample=is_fit),
            var=cohort.var.iloc[prep.selected_idx_].copy(),
        )
        out.uns["fit_on"] = fit_on
        out.uns["preprocessing"] = json.dumps(
            {
                "version": version,
                "top_k": k,
                "min_tpm": float(cfg.min_tpm),
                "min_fraction": float(cfg.min_fraction),
                "zscore": bool(cfg.zscore),
                "n_fit_samples": int(len(fit_pos)),
                "n_genes_out": int(z.shape[1]),
            }
        )
        path = out_dir / f"expression_{tag}.h5ad"
        out.write_h5ad(path, compression="gzip")
        joblib.dump(prep, out_dir / f"preprocessor_{tag}.joblib")
        written += [path, out_dir / f"preprocessor_{tag}.joblib"]
        LOG.info("%s: %d samples x %d genes (fit on %d)", tag, *z.shape, len(fit_pos))
        if gene_sets is not None:
            scores, names, _ = aggregate_gene_sets(
                z, out.var["gene_name"].tolist(), gene_sets, int(cfg.min_gene_set_overlap)
            )
            pw = ad.AnnData(X=scores, obs=out.obs.copy(), var=pd.DataFrame(index=names))
            pw.uns["fit_on"] = fit_on
            pw_path = out_dir / f"pathways_{tag}.h5ad"
            if pw_path.exists():
                raise FileExistsError(f"{pw_path} already exists; not overwriting")
            pw.write_h5ad(pw_path, compression="gzip")
            written.append(pw_path)
        del z, out
    return written


def load_ids(path: Path | None) -> list[str] | None:
    return None if path is None else read_id_list(path)
