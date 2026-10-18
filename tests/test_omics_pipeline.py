"""End-to-end test of ``build`` and ``fit-transform`` on tiny synthetic files (logic only)."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pytest.importorskip("sklearn")
anndata = pytest.importorskip("anndata")
pytest.importorskip("matplotlib")
joblib = pytest.importorskip("joblib")
pytest.importorskip("pyarrow")

from omegaconf import OmegaConf  # noqa: E402

from models.omics.pipeline import (  # noqa: E402
    EDA_ONLY,
    build_cohort,
    cohort_paths,
    fit_transform,
    load_cohort,
)
from tests.test_omics_preprocess import make_genes, write_star  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_GENES = 120


@pytest.fixture(scope="module")
def cfg() -> Any:
    base = OmegaConf.load(ROOT / "configs" / "omics" / "preprocess.yaml")
    overrides = OmegaConf.from_dotlist(
        ["pca_top_genes=60", "pca_components=5", "top_genes=[20,40]", "max_corr_samples=20"]
    )
    return OmegaConf.merge(base, overrides)


@pytest.fixture(scope="module")
def cohort(tmp_path_factory: pytest.TempPathFactory, cfg: Any) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("omics")
    rng = np.random.default_rng(0)
    genes = make_genes(N_GENES)
    profiles = {p: rng.gamma(2.0, 4.0, N_GENES) for p in ("TCGA-AAA", "TCGA-BBB")}
    rows = []
    for project, profile in profiles.items():
        for i in range(18):
            tpm = profile * rng.lognormal(0, 0.15, N_GENES)
            fid = f"{project}-{i}"
            write_star(
                root / "raw" / project / "rnaseq" / fid / "f.tsv",
                genes,
                np.round(tpm * 100),
                np.round(tpm, 3),
            )
            patient = f"TCGA-{project[-1]}{i:02d}-0001"
            rows.append(
                {
                    "patient_barcode": patient,
                    "project_id": project,
                    "rnaseq_file_id": fid,
                    "rnaseq_file_name": "f.tsv",
                    "rnaseq_sample_barcode": patient + "-01A",
                }
            )
    patient_map = pd.DataFrame(rows)
    out = root / "processed"
    summary = build_cohort(
        patient_map, root / "raw", out, root / "report", cfg, data_release="test"
    )
    return {"root": root, "out": out, "summary": summary, "patient_map": patient_map}


def test_build_writes_disk_backed_matrix_qc_and_report(cohort: dict[str, Any]) -> None:
    assert cohort["summary"]["n_samples"] == 36 and cohort["summary"]["n_genes"] == N_GENES
    loaded = load_cohort(cohort["out"])
    assert loaded.tpm.shape == (36, N_GENES) and loaded.tpm.dtype == np.float32
    assert isinstance(loaded.tpm, np.memmap)  # read from disk, not held in RAM
    assert loaded.meta["data_release"] == "test" and loaded.meta["n_genes"] == N_GENES
    assert {"project_id", "tss", "flag_any", "corr_to_median"} <= set(loaded.obs.columns)
    assert list(loaded.var.index[:2]) == [g[0] for g in make_genes(N_GENES)[:2]]
    assert not list(cohort["out"].glob("*.part"))
    report = (cohort["root"] / "report" / "qc_report.md").read_text(encoding="utf-8")
    assert "Batch etkisi" in report and "TCGA-AAA" in report
    for name in ("library_size_hist.png", "pca_project_vs_tss.png", "sample_correlation.png"):
        assert (cohort["root"] / "report" / name).stat().st_size > 1000


def test_matrix_on_disk_equals_matrix_in_memory(cohort: dict[str, Any], tmp_path: Path) -> None:
    from models.omics.expression import assemble_expression

    in_memory = assemble_expression(cohort["patient_map"], cohort["root"] / "raw")
    np.testing.assert_array_equal(load_cohort(cohort["out"]).tpm, in_memory.tpm)


def test_build_never_overwrites_existing_outputs(cohort: dict[str, Any], cfg: Any) -> None:
    with pytest.raises(FileExistsError, match="already exists"):
        build_cohort(
            cohort["patient_map"],
            cohort["root"] / "raw",
            cohort["out"],
            cohort["root"] / "report2",
            cfg,
        )
    # the original matrix is untouched
    assert cohort_paths(cohort["out"])["tpm"].exists()


def test_fit_transform_requires_explicit_training_set(cohort: dict[str, Any], cfg: Any) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        fit_transform(cohort["out"], cohort["root"] / "o1", cfg, None)
    with pytest.raises(ValueError, match="not in the cohort"):
        fit_transform(cohort["out"], cohort["root"] / "o2", cfg, ["TCGA-NOPE-0000"])


def test_fit_transform_learns_from_train_ids_only(cohort: dict[str, Any], cfg: Any) -> None:
    pm = cohort["patient_map"]
    train_ids = pm["patient_barcode"].iloc[[*range(0, 14), *range(18, 32)]].tolist()
    out = cohort["root"] / "ft"
    written = fit_transform(
        cohort["out"], out, cfg, train_ids, versions=["all", "protein_coding"], top_genes=[20, None]
    )
    assert (out / "expression_all_top20.h5ad").exists() and len(written) == 8
    adata = anndata.read_h5ad(out / "expression_protein_coding_top20.h5ad")
    assert adata.n_obs == 36 and int(adata.obs["is_fit_sample"].sum()) == len(train_ids)
    assert adata.uns["fit_on"].startswith("train_ids") and adata.n_vars == 20
    assert (adata.var["gene_type"] == "protein_coding").all()

    # Recompute the parameters from the training rows only and compare with what was saved.
    loaded = load_cohort(cohort["out"])
    mask = loaded.obs["patient_barcode"].isin(train_ids).to_numpy()
    prep = joblib.load(out / "preprocessor_all_top20.joblib")
    train_log = np.log2(np.asarray(loaded.tpm)[mask][:, prep.selected_idx_].astype(np.float64) + 1)
    np.testing.assert_allclose(prep.mean_, train_log.mean(axis=0), rtol=1e-6)
    # Held-out rows are standardised with TRAIN statistics, so their mean is not exactly zero.
    transformed = np.asarray(anndata.read_h5ad(out / "expression_all_top20.h5ad").X)
    np.testing.assert_allclose(transformed[mask].mean(axis=0), 0, atol=1e-4)
    assert np.abs(transformed[~mask].mean(axis=0)).max() > 1e-4


def test_fit_transform_never_overwrites(cohort: dict[str, Any], cfg: Any) -> None:
    out = cohort["root"] / "ow"
    fit_transform(
        cohort["out"], out, cfg, None, eda_all_samples=True, versions=["all"], top_genes=[20]
    )
    before = (out / "expression_all_top20.h5ad").stat().st_mtime_ns
    with pytest.raises(FileExistsError, match="not overwriting"):
        fit_transform(
            cohort["out"], out, cfg, None, eda_all_samples=True, versions=["all"], top_genes=[20]
        )
    assert (out / "expression_all_top20.h5ad").stat().st_mtime_ns == before


def test_row_chunked_transform_matches_one_shot(cohort: dict[str, Any], cfg: Any) -> None:
    from models.omics import pipeline
    from models.omics.preprocess import ExpressionPreprocessor

    loaded = load_cohort(cohort["out"])
    prep = ExpressionPreprocessor(min_fraction=0.2, top_k=30).fit(loaded.tpm)
    rows = np.arange(len(loaded.obs))
    old = pipeline.ROW_CHUNK
    pipeline.ROW_CHUNK = 7  # force several chunks
    try:
        chunked = pipeline._transform_rows(prep, loaded.tpm, rows)
    finally:
        pipeline.ROW_CHUNK = old
    np.testing.assert_allclose(chunked, prep.transform(np.asarray(loaded.tpm)))


def test_eda_mode_is_stamped_and_gene_sets_work(cohort: dict[str, Any], cfg: Any) -> None:
    gmt = cohort["root"] / "sets.gmt"
    names = [g[1] for g in make_genes(N_GENES)]
    gmt.write_text(
        f"SET1\td\t{chr(9).join(names[:30])}\nSET2\td\t{chr(9).join(names[30:60])}\n",
        encoding="utf-8",
    )
    out = cohort["root"] / "eda"
    written = fit_transform(
        cohort["out"],
        out,
        cfg,
        None,
        eda_all_samples=True,
        versions=["all"],
        top_genes=[None],
        gene_sets_path=gmt,
    )
    adata = anndata.read_h5ad(out / "expression_all_topall.h5ad")
    assert adata.uns["fit_on"] == EDA_ONLY and adata.obs["is_fit_sample"].all()
    pathways = anndata.read_h5ad(out / "pathways_all_topall.h5ad")
    assert pathways.n_vars == 2 and pathways.uns["fit_on"] == EDA_ONLY
    assert out / "pathways_all_topall.h5ad" in written
    meta = json.loads(adata.uns["preprocessing"])
    assert meta["n_fit_samples"] == 36


def test_drop_flagged_removes_flagged_samples(cohort: dict[str, Any], cfg: Any) -> None:
    copy = cohort["root"] / "flagged_copy"
    shutil.copytree(cohort["out"], copy)
    paths = cohort_paths(copy)
    obs = pd.read_parquet(paths["obs"])
    obs.loc[obs.index[:3], "flag_any"] = True
    obs.to_parquet(paths["obs"])
    out = cohort["root"] / "dropped"
    fit_transform(
        copy,
        out,
        cfg,
        None,
        eda_all_samples=True,
        drop_flagged=True,
        versions=["all"],
        top_genes=[20],
    )
    assert anndata.read_h5ad(out / "expression_all_top20.h5ad").n_obs == 33
