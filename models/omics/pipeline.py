"""Cohort assembly + QC (``build``) and leakage-safe fitting (``fit_transform``), sprint 1.4."""

import json
import logging
from collections.abc import Sequence
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
TPM_FILE = "expression_tpm.h5ad"


def build_cohort(
    patient_map: pd.DataFrame,
    raw_root: Path,
    out_dir: Path,
    report_dir: Path,
    cfg: DictConfig,
    projects: Sequence[str] | None = None,
    data_release: str | None = None,
) -> dict[str, Any]:
    """Assemble the TPM matrix, run sample QC and batch diagnostics, write h5ad + report.

    Nothing learned from data is applied here (no scaling, no gene filtering by prevalence).
    """
    if projects:
        patient_map = patient_map[patient_map["project_id"].isin(list(projects))]
    data: ExpressionData = assemble_expression(patient_map, raw_root)
    logged = qcm.log2_tpm(data.tpm)
    qc = qcm.sample_qc(
        data.tpm,
        data.obs["library_size"].to_numpy(),
        data.obs["n_detected_genes"].to_numpy(),
        mad_threshold=float(cfg.mad_threshold),
        logged=logged,
    )
    qc.index = data.obs.index
    pca = qcm.pca_embedding(
        logged, int(cfg.pca_components), int(cfg.pca_top_genes), seed=int(cfg.seed)
    )
    umap_xy = qcm.umap_embedding(pca, seed=int(cfg.seed))
    project = data.obs["project_id"].to_numpy()
    tss = data.obs["tss"].to_numpy()
    batch = qcm.batch_effect_summary(pca, project, tss)
    figures = qcm.save_qc_plots(
        report_dir,
        qc.reset_index(drop=True),
        logged,
        project,
        tss,
        pca,
        umap_xy,
        max_corr_samples=int(cfg.max_corr_samples),
        seed=int(cfg.seed),
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "qc_report.md").write_text(
        qcm.qc_report_markdown(data.obs, qc, batch, float(cfg.mad_threshold), figures),
        encoding="utf-8",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    obs = data.obs.join(qc.drop(columns=["library_size", "n_detected_genes"]))
    adata = ad.AnnData(X=data.tpm, obs=obs, var=data.genes)
    adata.uns["source"] = "GDC STAR-Counts, column tpm_unstranded"
    adata.uns["data_release"] = data_release or "unknown"
    adata.write_h5ad(out_dir / TPM_FILE, compression="gzip")
    qc.to_parquet(out_dir / "sample_qc.parquet")
    return {
        "n_samples": int(data.tpm.shape[0]),
        "n_genes": int(data.tpm.shape[1]),
        "n_flagged": int(qc["flag_any"].sum()),
        "batch": batch,
    }


def fit_transform(
    tpm_h5ad: Path,
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
    """
    if (train_ids is None) == (not eda_all_samples):
        raise ValueError("give exactly one of train_ids or eda_all_samples=True")
    adata = ad.read_h5ad(tpm_h5ad)
    if drop_flagged:
        adata = adata[~adata.obs["flag_any"].to_numpy()].copy()
    tpm = np.asarray(adata.X, dtype=np.float32)
    fit_idx = np.arange(adata.n_obs) if train_ids is None else select_samples(adata.obs, train_ids)
    if len(fit_idx) < 2:
        raise ValueError(f"only {len(fit_idx)} training samples after QC filtering; need >= 2")
    is_fit = np.zeros(adata.n_obs, dtype=bool)
    is_fit[fit_idx] = True
    fit_on = EDA_ONLY if train_ids is None else f"train_ids:{len(fit_idx)}_samples"

    gene_sets = parse_gmt(gene_sets_path) if gene_sets_path else None
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for version in versions or list(cfg.versions):
        if version not in ("all", "protein_coding"):
            raise ValueError(f"unknown version {version!r}")
        allowed = (
            (adata.var["gene_type"] == "protein_coding").to_numpy()
            if version == "protein_coding"
            else None
        )
        for k in top_genes if top_genes is not None else list(cfg.top_genes):
            prep = ExpressionPreprocessor(
                min_tpm=float(cfg.min_tpm),
                min_fraction=float(cfg.min_fraction),
                top_k=None if k is None else int(k),
                allowed_genes=allowed,
                zscore=bool(cfg.zscore),
                clip=None if cfg.clip is None else float(cfg.clip),
            ).fit(tpm[fit_idx])
            z = prep.transform(tpm)
            tag = f"{version}_top{k if k is not None else 'all'}"
            var = adata.var.iloc[prep.selected_idx_].copy()
            out = ad.AnnData(X=z, obs=adata.obs.assign(is_fit_sample=is_fit), var=var)
            out.uns["fit_on"] = fit_on
            out.uns["preprocessing"] = json.dumps(
                {
                    "version": version,
                    "top_k": k,
                    "min_tpm": float(cfg.min_tpm),
                    "min_fraction": float(cfg.min_fraction),
                    "zscore": bool(cfg.zscore),
                    "n_fit_samples": int(len(fit_idx)),
                    "n_genes_out": int(z.shape[1]),
                }
            )
            path = out_dir / f"expression_{tag}.h5ad"
            out.write_h5ad(path, compression="gzip")
            joblib.dump(prep, out_dir / f"preprocessor_{tag}.joblib")
            written += [path, out_dir / f"preprocessor_{tag}.joblib"]
            LOG.info("%s: %d samples x %d genes (fit on %d)", tag, *z.shape, len(fit_idx))
            if gene_sets is not None:
                scores, names, _ = aggregate_gene_sets(
                    z, var["gene_name"].tolist(), gene_sets, int(cfg.min_gene_set_overlap)
                )
                pw = ad.AnnData(X=scores, obs=out.obs.copy(), var=pd.DataFrame(index=names))
                pw.uns["fit_on"] = fit_on
                pw_path = out_dir / f"pathways_{tag}.h5ad"
                pw.write_h5ad(pw_path, compression="gzip")
                written.append(pw_path)
    return written


def load_ids(path: Path | None) -> list[str] | None:
    return None if path is None else read_id_list(path)
