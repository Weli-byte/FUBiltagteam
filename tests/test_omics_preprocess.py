"""Tests for omics parsing, the fitted preprocessor and QC.

All inputs are tiny synthetic arrays/files built in the tests to exercise the logic. They are not
data and not used outside the tests. Real-data behaviour is verified with ``inspect``/``build``.
"""

import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pytest.importorskip("sklearn")
pytest.importorskip("anndata")
pytest.importorskip("matplotlib")

from sklearn.base import clone  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from models.omics import qc as qcm  # noqa: E402
from models.omics.expression import (  # noqa: E402
    assemble_expression,
    gene_dictionary,
    parse_star_counts,
    read_id_list,
    select_samples,
    tss_of,
)
from models.omics.preprocess import (  # noqa: E402
    ExpressionPreprocessor,
    aggregate_gene_sets,
    parse_gmt,
)

HEADER = "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\ttpm_unstranded\n"


def write_star(path: Path, genes: list[tuple[str, str, str]], counts: Any, tpm: Any) -> Path:
    lines = ["# gene-model: test", HEADER.rstrip("\n")]
    lines += [f"N_{n}\t\t\t5\t5\t0" for n in ("unmapped", "multimapping")]
    for (gid, name, gtype), c, t in zip(genes, counts, tpm, strict=True):
        lines.append(f"{gid}\t{name}\t{gtype}\t{int(c)}\t{int(c)}\t{float(t)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_genes(n: int) -> list[tuple[str, str, str]]:
    return [
        (f"ENSG{i:011d}.1", f"GENE{i}", "protein_coding" if i % 2 == 0 else "lncRNA")
        for i in range(n)
    ]


# ------------------------------------------------------------------------------ parsing
def test_parse_star_counts_skips_comments_and_summary_rows(tmp_path: Path) -> None:
    genes = make_genes(4)
    path = write_star(tmp_path / "a.tsv", genes, [10, 20, 30, 40], [1.5, 2.5, 3.5, 4.5])
    parsed = parse_star_counts(path)
    assert list(parsed.gene_id) == [g[0] for g in genes]
    assert parsed.counts.sum() == 100 and parsed.tpm.dtype == np.float32
    assert not any(g.startswith("N_") for g in parsed.gene_id)


def test_parse_reports_missing_columns_with_file_name(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tsv"
    bad.write_text("gene_id\tgene_name\nA\tB\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.tsv.*missing columns"):
        parse_star_counts(bad)


def test_gene_dictionary_flags_duplicates_without_dropping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from models.omics.expression import StarCounts

    parsed = StarCounts(
        gene_id=np.array(["ENSG1.5", "ENSG1.5_PAR_Y", "ENSG2.1", "ENSG3.1"]),
        gene_name=np.array(["X", "X", "Y", ""]),
        gene_type=np.array(["protein_coding"] * 4),
        counts=np.zeros(4),
        tpm=np.zeros(4, dtype=np.float32),
    )
    with caplog.at_level(logging.WARNING):
        table = gene_dictionary(parsed)
    assert len(table) == 4  # nothing dropped
    assert table["symbol_is_duplicate"].tolist() == [True, True, False, False]
    assert table["ensembl_is_duplicate"].tolist() == [True, True, False, False]
    assert "duplicated" in caplog.text


def test_tss_of_barcode() -> None:
    assert tss_of("TCGA-AB-1234-01A") == "AB" and tss_of("weird") == "unknown"


# ---------------------------------------------------------------------------- assembly
def _cohort_files(tmp_path: Path, n_per_project: int = 3, n_genes: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    genes = make_genes(n_genes)
    rows: list[dict[str, Any]] = []
    for project in ("TCGA-AAA", "TCGA-BBB"):
        for i in range(n_per_project):
            patient = f"TCGA-{project[-1]}{i}-0001"
            fid = f"{project}-{i}"
            tpm = rng.gamma(2.0, 3.0, n_genes)
            write_star(
                tmp_path / project / "rnaseq" / fid / "f.tsv",
                genes,
                np.round(tpm * 100),
                np.round(tpm, 3),
            )
            rows.append(
                {
                    "patient_barcode": patient,
                    "project_id": project,
                    "rnaseq_file_id": fid,
                    "rnaseq_file_name": "f.tsv",
                    "rnaseq_sample_barcode": patient + "-01A",
                }
            )
    rows.append(
        {
            "patient_barcode": "TCGA-NO-0000",
            "project_id": "TCGA-AAA",
            "rnaseq_file_id": None,
            "rnaseq_file_name": None,
            "rnaseq_sample_barcode": None,
        }
    )
    return pd.DataFrame(rows)


def test_assemble_expression_builds_matrix_and_obs(tmp_path: Path) -> None:
    data = assemble_expression(_cohort_files(tmp_path), tmp_path)
    assert data.tpm.shape == (6, 12)  # patient without RNA-seq is skipped
    assert data.obs["project_id"].nunique() == 2
    assert (data.obs["library_size"] > 0).all() and data.obs.index.is_unique
    assert list(data.genes.index) == [g[0] for g in make_genes(12)]


def test_assemble_to_disk_never_overwrites_and_leaves_no_part_file(tmp_path: Path) -> None:
    pm = _cohort_files(tmp_path)
    target = tmp_path / "out" / "m.npy"
    data = assemble_expression(pm, tmp_path, tpm_path=target)
    assert isinstance(data.tpm, np.memmap) and target.exists()
    assert not list(target.parent.glob("*.part"))
    in_memory = assemble_expression(pm, tmp_path)
    np.testing.assert_array_equal(data.tpm, in_memory.tpm)
    before = target.stat().st_mtime_ns
    with pytest.raises(FileExistsError, match="not overwriting"):
        assemble_expression(pm, tmp_path, tpm_path=target)
    assert target.stat().st_mtime_ns == before


def test_assemble_rejects_changed_gene_order_and_missing_files(tmp_path: Path) -> None:
    pm = _cohort_files(tmp_path)
    shuffled = list(reversed(make_genes(12)))
    write_star(
        tmp_path / "TCGA-BBB" / "rnaseq" / "TCGA-BBB-1" / "f.tsv",
        shuffled,
        np.ones(12),
        np.ones(12),
    )
    with pytest.raises(ValueError, match="gene order"):
        assemble_expression(pm, tmp_path)
    (tmp_path / "TCGA-AAA" / "rnaseq" / "TCGA-AAA-0" / "f.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="not found"):
        assemble_expression(pm, tmp_path)


def test_select_samples_accepts_patient_or_sample_barcodes(tmp_path: Path) -> None:
    data = assemble_expression(_cohort_files(tmp_path), tmp_path)
    by_patient = select_samples(data.obs, ["TCGA-A0-0001"])
    by_sample = select_samples(data.obs, ["TCGA-A0-0001-01A"])
    assert list(by_patient) == list(by_sample) and len(by_patient) == 1
    with pytest.raises(ValueError, match="not in the cohort"):
        select_samples(data.obs, ["TCGA-ZZ-9999"])


def test_read_id_list_ignores_blanks_and_comments(tmp_path: Path) -> None:
    f = tmp_path / "ids.txt"
    f.write_text("# train\nA\n\n  B  \n", encoding="utf-8")
    assert read_id_list(f) == ["A", "B"]


# ------------------------------------------------------------------------ preprocessor
def _tpm(n: int = 40, g: int = 30, seed: int = 1) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 5.0, size=(1, g)) * rng.uniform(0.2, 3.0, size=(n, g))
    return base.astype(np.float32)


def test_parameters_come_from_training_samples_only() -> None:
    tpm = _tpm()
    train, test = tpm[:25], tpm[25:].copy()
    prep = ExpressionPreprocessor(min_fraction=0.2).fit(train)
    logged = np.log2(train[:, prep.selected_idx_].astype(np.float64) + 1)
    np.testing.assert_allclose(prep.mean_, logged.mean(axis=0))
    np.testing.assert_allclose(prep.scale_, logged.std(axis=0))
    # Wild test values must not move the learned parameters.
    before = (prep.selected_idx_.copy(), prep.mean_.copy(), prep.scale_.copy())
    test[:] = 1e6
    prep.transform(test)
    assert np.array_equal(before[0], prep.selected_idx_)
    np.testing.assert_array_equal(before[1], prep.mean_)
    np.testing.assert_array_equal(before[2], prep.scale_)
    # Fitting on different test data gives the same model: test rows never enter fit().
    again = ExpressionPreprocessor(min_fraction=0.2).fit(train)
    np.testing.assert_array_equal(again.mean_, prep.mean_)


def test_constant_gene_with_float_noise_is_not_zscored() -> None:
    tpm = _tpm(g=4)
    tpm[:, 1] = np.float32(0.1) * 3  # constant column; log2 mean may differ by ~1e-16
    tpm[:, 1] += 5.0
    prep = ExpressionPreprocessor(min_fraction=0.0).fit(tpm)
    assert 1 not in set(prep.selected_idx_)
    assert np.isfinite(prep.transform(tpm)).all()


def test_train_set_is_standardised() -> None:
    tpm = _tpm()
    prep = ExpressionPreprocessor(min_fraction=0.0)
    z = prep.fit_transform(tpm)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-4)
    assert z.dtype == np.float32


def test_expression_filter_and_zero_variance_genes_removed() -> None:
    tpm = _tpm(g=6)
    tpm[:, 0] = 0.0  # never expressed
    tpm[:, 1] = 5.0  # constant -> zero variance
    tpm[:, 2] = 0.0
    tpm[:3, 2] = 9.0  # expressed in 3/40 = 7.5% of samples (< 20%)
    prep = ExpressionPreprocessor(min_tpm=1.0, min_fraction=0.2).fit(tpm)
    assert set(prep.selected_idx_) == {3, 4, 5}


def test_top_k_keeps_highest_variance_in_original_order() -> None:
    rng = np.random.default_rng(3)
    tpm = rng.uniform(5, 6, size=(50, 8)).astype(np.float32)
    tpm[:, 5] *= rng.uniform(0.1, 20, size=50)
    tpm[:, 2] *= rng.uniform(0.1, 10, size=50)
    prep = ExpressionPreprocessor(min_fraction=0.0, top_k=2).fit(tpm)
    assert list(prep.selected_idx_) == [2, 5]


def test_allowed_gene_mask_is_respected() -> None:
    tpm = _tpm(g=10)
    allowed = np.arange(10) % 2 == 0
    prep = ExpressionPreprocessor(min_fraction=0.0, allowed_genes=allowed).fit(tpm)
    assert all(allowed[i] for i in prep.selected_idx_)


def test_clip_limits_extreme_values() -> None:
    tpm = _tpm(g=5)
    prep = ExpressionPreprocessor(min_fraction=0.0, clip=1.5).fit(tpm[:30])
    spiked = tpm[30:].copy()
    spiked[0] = 1e7
    assert np.abs(prep.transform(spiked)).max() <= 1.5 + 1e-6


@pytest.mark.parametrize("bad", ["nan", "negative", "wrong_shape"])
def test_invalid_input_is_rejected(bad: str) -> None:
    tpm = _tpm()
    prep = ExpressionPreprocessor(min_fraction=0.0).fit(tpm)
    probe = tpm[:2].copy()
    if bad == "nan":
        probe[0, 0] = np.nan
    elif bad == "negative":
        probe[0, 0] = -1.0
    else:
        probe = probe[:, :-1]
    with pytest.raises(ValueError):
        prep.transform(probe)


def test_edge_cases_all_zero_matrix_and_too_few_samples() -> None:
    with pytest.raises(ValueError, match="no gene passed"):
        ExpressionPreprocessor().fit(np.zeros((10, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="at least 2"):
        ExpressionPreprocessor().fit(_tpm(n=1))
    with pytest.raises(NotFittedError):
        ExpressionPreprocessor().transform(_tpm())


def test_sklearn_compatibility_clone_and_pipeline() -> None:
    prep = ExpressionPreprocessor(min_fraction=0.1, top_k=5)
    assert clone(prep).get_params()["top_k"] == 5
    out = Pipeline([("expr", prep)]).fit_transform(_tpm())
    assert out.shape[1] == 5


def test_fit_is_deterministic() -> None:
    a = ExpressionPreprocessor(top_k=7).fit(_tpm())
    b = ExpressionPreprocessor(top_k=7).fit(_tpm())
    assert np.array_equal(a.selected_idx_, b.selected_idx_)


# ---------------------------------------------------------------------------- gene sets
def test_gmt_parsing_and_aggregation(tmp_path: Path) -> None:
    gmt = tmp_path / "s.gmt"
    gmt.write_text("SET_A\tdesc\tg1\tg2\tG3\nSET_B\tdesc\tg1\n", encoding="utf-8")
    sets = parse_gmt(gmt)
    assert sets["SET_A"] == ["g1", "g2", "G3"]
    z = np.arange(12, dtype=np.float32).reshape(2, 6)
    scores, names, overlap = aggregate_gene_sets(
        z, ["G1", "G2", "g3", "x", "y", "z"], sets, min_overlap=2
    )
    assert names == ["SET_A"] and overlap == {"SET_A": 3, "SET_B": 1}  # SET_B skipped, reported
    np.testing.assert_allclose(scores[:, 0], z[:, :3].mean(axis=1))
    gmt.write_text("S\td\ta\nS\td\tb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        parse_gmt(gmt)
    with pytest.raises(ValueError, match="no gene set"):
        aggregate_gene_sets(z, ["a"] * 6, {"S": ["a"]}, min_overlap=100)


# -------------------------------------------------------------------------------- QC
def test_robust_z_handles_constant_input() -> None:
    assert (qcm.robust_z(np.ones(5)) == 0).all()
    z = qcm.robust_z(np.array([1.0, 2, 3, 4, 100]))
    assert z[-1] > 10 and abs(z[2]) < 1e-9


def test_sample_qc_flags_planted_outliers() -> None:
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 4.0, size=(1, 200))
    tpm = (base * rng.lognormal(0, 0.1, size=(60, 200))).astype(np.float32)
    library = rng.normal(5e7, 1e6, 60)
    detected = rng.normal(15000, 100, 60).astype(int)
    library[3] = 2e5  # tiny library
    tpm[10] = rng.permutation(tpm[10])  # scrambled profile -> low correlation
    tpm[20, 5] = np.nan  # missing value
    table = qcm.sample_qc(tpm, library, detected, mad_threshold=4.0)
    assert table.loc[3, "flag_library_size"]
    assert table.loc[10, "flag_low_correlation"]
    assert table.loc[20, "flag_missing"]
    assert table["flag_any"].sum() <= 6  # almost all clean samples stay unflagged


def test_batch_summary_separates_projects_but_not_random_tss() -> None:
    rng = np.random.default_rng(1)
    project = np.repeat(["P1", "P2"], 40)
    emb = rng.normal(0, 1, (80, 5))
    emb[project == "P2"] += 8.0
    tss = rng.choice(["a", "b", "c"], size=80)
    summary = qcm.batch_effect_summary(emb, project, tss)
    assert summary["silhouette_project"] > 0.6
    assert summary["silhouette_tss_mean"] < 0.2
    assert qcm.silhouette(emb, np.zeros(80)) is None  # one group -> undefined, not an error
